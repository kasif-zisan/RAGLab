"""Phase 1: parse every document in the manifest with Docling, checkpointed to S3.

REUSABLE SKELETON. Everything here except parse_one() is generic to any batch job
that is a list of independent units. See infra/README.md for the full playbook.

CHANGE FOR A NEW PROJECT: BUCKET, REGION, the unit list, and parse_one(). The
checkpoint logic, the pool, the progress estimate and the sizing all carry over.

WHY IT LOOKS LIKE THIS - each point cost us something:

  * THE OUTPUT FILE IS THE CHECKPOINT. parsed/<stem>.json existing means that
    document finished. No state file, no database, no lock. Therefore it is
    written only AFTER success - write it first and an interruption mid-document
    leaves a checkpoint claiming "done" for half-finished work, which a re-run
    then skips forever without error.
  * FAILURES GO TO A DIFFERENT PREFIX. A failure written under parsed/ would read
    as "done" and never be retried.
  * S3 PutObject is atomic, so no temp-then-rename is needed. On a LOCAL
    filesystem it would be: a half-written file looks exactly like a complete one.
  * PAGE PROVENANCE IS NOT RECOVERABLE LATER. export_to_markdown() returns one
    string with no page boundaries; passing page_break_placeholder is the only
    way to keep them, and without them the best citation you can offer a user is
    "somewhere in this 91-page document", which is not a citation. We threw away
    35 documents of work relearning this.
  * PROGRESS IS ESTIMATED FROM PER-UNIT CORE-TIME, not from wall-clock
    throughput. Both `elapsed/completed x remaining` and `remaining_work /
    (work_done / elapsed)` are biased the same way: with N workers, N-1 units are
    IN FLIGHT and partly done, and neither formula can see that work. Our first
    attempt said 555 min against a real 168; the second said 37 HOURS. The fix is
    to divide each finished unit's OWN measured duration by its size - that rate
    is a property of the work, not of how many workers exist or how recently they
    started - then scale by total work and divide by the worker count.
  * THE PARSER'S OWN WARNINGS ARE A FREE CONFIDENCE SIGNAL. Docling silently
    drops table cells that do not fit its predicted grid and says so only in a
    log line. In an interleaved N-worker log those cannot be attributed to a
    document, so we capture them per unit here instead.
  * WORKER COUNT COMES FROM RAM, NOT CORES, for any job holding model weights per
    process - then measure actual RSS and re-size. We assumed ~3 GB per worker,
    measured 2.8-5.0 GB, and 14 workers left 9 GB free on a 64 GB box. An OOM kill
    breaks ProcessPoolExecutor and ends the WHOLE run, not one unit.
  * torch.set_num_threads(1) IS NOT ENOUGH. OpenCV and ONNX Runtime keep their own
    pools; OpenCV is handled here, the rest are exported in bootstrap.sh before
    python starts, because those libraries read the environment at import time.
  * boto3 clients are NOT fork-safe - each worker process builds its own.
  * The converter is a per-process global built on first use, so model weights
    load once per worker rather than once per unit.
"""

import json
import logging
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BUCKET = "raglab-zisan-apne1"
REGION = "ap-northeast-1"
WORK = Path("/opt/raglab")
PAGEBREAK = "<!-- PAGEBREAK -->"

MAX_WORKERS = 10   # measured: each worker held 2.8-5.0 GB, not the ~3 GB assumed.
                   # 14 workers left only 9 GB free on a 64 GB box, and an OOM kill
                   # breaks the pool and takes the whole run down with it.

_CONV = None       # one DocumentConverter per worker process, built on first use

# "253 of 387 pdf cells matched neither a row nor a column band of the 27x24 grid"
DROP_RE = re.compile(r"(\d+) of (\d+) pdf cells matched neither")


class WarningCollector(logging.Handler):
    """Captures docling's WARNING records so they can be attributed to a document."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        try:
            self.messages.append(record.getMessage())
        except Exception:
            pass


def workers() -> int:
    """RAM caps us, not cores - docling holds several GB of weights per process."""
    mem_gb = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9)
    return max(1, min(MAX_WORKERS, os.cpu_count() or 1, int(mem_gb / 5.5)))


def page_count(pdf: Path) -> int:
    import pymupdf
    with pymupdf.open(pdf) as d:
        return d.page_count


def parse_one(stem: str) -> dict:
    global _CONV
    # Each process gets ONE torch thread. Without this, N processes each spawn N
    # threads and the box spends its time context-switching instead of parsing.
    import torch
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(0)      # OpenCV keeps its own pool, separate from torch
    except Exception:
        pass

    if _CONV is None:
        from docling.document_converter import DocumentConverter
        _CONV = DocumentConverter()

    collector = WarningCollector()
    logging.getLogger().addHandler(collector)
    try:
        t0 = time.time()
        res = _CONV.convert(str(WORK / "pdf" / f"{stem}.pdf"))
        doc = res.document
        # page_break_placeholder is the ONLY way to keep page boundaries; without
        # it the whole document comes back as one undifferentiated string.
        md = doc.export_to_markdown(page_break_placeholder=PAGEBREAK)
        elapsed = time.time() - t0
    finally:
        logging.getLogger().removeHandler(collector)

    pages = md.split(PAGEBREAK)
    drops = [(int(a), int(b)) for a, b in DROP_RE.findall("\n".join(collector.messages))]

    return {
        "stem": stem,
        "pages": len(doc.pages),
        "page_segments": len(pages),   # should equal pages; a mismatch is a red flag
        "tables": len(doc.tables),
        "chars": len(md),
        "elapsed_s": round(elapsed, 1),
        # one markdown string per page. Index position n == page n (0-based).
        # This is the join key for citations and for small-to-big retrieval.
        "page_markdown": pages,
        # free confidence signal: cells docling discarded because they did not
        # fit its predicted grid. High ratio == this document's tables are suspect.
        "tables_with_drops": len(drops),
        "cells_dropped": sum(a for a, _ in drops),
        "cells_in_dropped_tables": sum(b for _, b in drops),
        "warnings": collector.messages[:200],
    }


def task(stem: str) -> tuple[str, bool, int, float, str]:
    """Runs in a worker process. Writes its own result to S3 and reports back."""
    s3 = boto3.client("s3", region_name=REGION)   # clients are not fork-safe
    try:
        out = parse_one(stem)
        out["docling"] = __import__("docling").__version__
        s3.put_object(
            Bucket=BUCKET, Key=f"parsed/{stem}.json",
            Body=json.dumps(out).encode(), ContentType="application/json",
        )
        drop = ""
        if out["cells_dropped"]:
            drop = (f" DROPPED {out['cells_dropped']}/{out['cells_in_dropped_tables']}"
                    f" cells in {out['tables_with_drops']} tables")
        warn = "" if out["page_segments"] == out["pages"] else \
               f" !! {out['page_segments']} segments vs {out['pages']} pages"
        return (stem, True, out["pages"], out["elapsed_s"],
                f"pages={out['pages']} tables={out['tables']} {out['elapsed_s']}s{drop}{warn}")
    except Exception:
        tb = traceback.format_exc()
        # NOT parsed/ - a key there would mean "done" and a re-run would skip it.
        s3.put_object(
            Bucket=BUCKET, Key=f"errors/{stem}.json",
            Body=json.dumps({"stem": stem, "error": tb}).encode(),
            ContentType="application/json",
        )
        return stem, False, 0, 0.0, tb.strip().splitlines()[-1][:160]


def done_already(s3, stem: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=f"parsed/{stem}.json")
        return True
    except ClientError:
        return False


def hms(minutes: float) -> str:
    return f"{int(minutes//60)}h{int(minutes%60):02d}m" if minutes >= 60 else f"{minutes:.0f}m"


def main() -> int:
    s3 = boto3.client("s3", region_name=REGION)
    stems = json.loads((WORK / "manifest.json").read_text())["corpus_docs"]

    todo = [s for s in stems if not done_already(s3, s)]
    nw = workers()
    print(f"manifest: {len(stems)} documents | already done: {len(stems)-len(todo)} "
          f"| to parse: {len(todo)}")
    print(f"workers: {nw}  (cpus={os.cpu_count()})", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    # Page counts up front (cheap - pymupdf reads the xref, not the content) so
    # progress can be measured in work done rather than units done.
    t0 = time.time()
    total_pages = sum(page_count(WORK / "pdf" / f"{s}.pdf") for s in todo)
    print(f"total pages: {total_pages:,}  (counted in {time.time()-t0:.1f}s)", flush=True)

    t0, ok, bad, pages_done, core_s = time.time(), 0, [], 0, 0.0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futures = {pool.submit(task, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            stem, good, npages, secs, note = fut.result()
            ok += good
            pages_done += npages
            core_s += secs
            if not good:
                bad.append(stem)
            el = (time.time() - t0) / 60
            # s/page measured from each finished document's OWN duration. This is
            # unbiased from the first completion: it does not depend on worker
            # count or on how much in-flight work is invisible. Wall-clock
            # throughput (pages_done/elapsed) IS biased that way - it reported
            # 37 hours against a real 2h47m.
            if pages_done:
                spp = core_s / pages_done
                left = hms((total_pages - pages_done) * spp / nw / 60)
                rate = f"{spp:.2f} s/pg, {nw*60/spp:.0f} pg/min"
            else:
                left, rate = "estimating", "-"
            print(f"[{i}/{len(todo)}] {'OK  ' if good else 'FAIL'} {stem:12s} {note} "
                  f"| {pages_done:,}/{total_pages:,} pages, {rate}, ~{left} left", flush=True)

    mins = (time.time() - t0) / 60
    print(f"\n=== {ok}/{len(todo)} parsed, {pages_done:,} pages in {hms(mins)} "
          f"({core_s/max(pages_done,1):.2f} s/page across {nw} workers) ===")
    if bad:
        print(f"FAILED ({len(bad)}): {bad}")
        print("re-run this job to retry them; successful documents are skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
