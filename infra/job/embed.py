"""Phase 7: embed one arm's chunks to sharded vectors in S3.

RUN:  ARM=B1_3200_ov889 python embed.py

WHY IT LOOKS LIKE THIS - the parse job's rules, carried over:

  * THE SHARD FILE IS THE CHECKPOINT. vectors/<arm>/<model>/shard-00042.npy
    existing means chunks [42*SHARD, 43*SHARD) are done. No state file, no lock.
    It is uploaded only AFTER the rows are complete, because a shard written
    first would claim "done" for work that never finished, and a re-run would
    skip it forever without error.
  * FAILURES GO TO A DIFFERENT PREFIX. A failure under vectors/ would read as
    "done" and never be retried.
  * RE-RUNNING IS ALWAYS SAFE. It does whatever is not done yet. That is what
    makes a spot interruption a non-event: you lose only the shard in flight.
  * NO PARTIAL-SHARD FLUSH ON INTERRUPTION, deliberately. A partial shard is a
    file whose length disagrees with its name, which is a whole class of bug for
    the sake of saving ~60 seconds of work. The spot poller instead stops
    cleanly and says why, and the next run redoes that one shard.
  * META.JSON IS WRITTEN FIRST, not last. It records the model, dtype and the
    exact doc/query prefixes. Query time MUST read its prefix from here - a
    mismatch between index time and query time raises no error anywhere, it only
    lowers recall. Writing it first means an interrupted run still leaves behind
    the description of what its shards contain.
  * fp16 ON DISK. 137,138 x 768 is 211 MB at fp16 against 421 at fp32, the
    ranking is unaffected, and the dtype is recorded so query time matches.
"""
import gzip, json, os, pathlib, subprocess, sys, threading, time, urllib.request

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BUCKET = "raglab-zisan-apne1"
WORK = pathlib.Path("/opt/raglab")
ARM = os.environ["ARM"]                                  # e.g. B1_3200_ov889
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-Embedding-0.6B")
SHARD = int(os.environ.get("SHARD", "2000"))
BATCH = int(os.environ.get("BATCH", "32"))

# Prefixes are defined by the model's own training and are NOT interchangeable.
# gte-modernbert uses none; the others are here so a swap is a one-line change
# rather than a rediscovery.
PREFIXES = {
    "Alibaba-NLP/gte-modernbert-base":         ("", ""),
    # Qwen3 ships a QUERY instruction in config_sentence_transformers.json and
    # sets default_prompt_name=null, so sentence-transformers does NOT apply it
    # unless asked. Documents get "". Recorded verbatim here and copied into
    # meta.json; query time must use the string from meta.json, not this file.
    # It touches only the query side, so it can be re-tuned later WITHOUT
    # re-embedding the corpus.
    "Qwen/Qwen3-Embedding-0.6B": (
        "",
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery:",
    ),
    "Snowflake/snowflake-arctic-embed-m-v2.0": ("", "query: "),
    "nomic-ai/nomic-embed-text-v1.5":          ("search_document: ", "search_query: "),
}
DOC_PREFIX, QUERY_PREFIX = PREFIXES[MODEL]
SLUG = MODEL.split("/")[-1]
DEST = f"s3://{BUCKET}/vectors/{ARM}/{SLUG}"

_stop = threading.Event()


def spot_watch():
    """Poll the instance metadata service for the spot interruption notice.

    This endpoint is reachable ONLY from inside the instance. Nothing outside -
    no watcher, no console - can see the two-minute warning; from outside the
    first visible sign is the instance already shutting down. So the only place
    this check can live is here.

    404 is the normal answer and means 'not interrupted'.
    """
    url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    while not _stop.is_set():
        try:
            tok = urllib.request.urlopen(urllib.request.Request(
                "http://169.254.169.254/latest/api/token", method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}), timeout=2).read()
            r = urllib.request.urlopen(urllib.request.Request(
                url, headers={"X-aws-ec2-metadata-token": tok}), timeout=2)
            print(f"\n!!! SPOT INTERRUPTION NOTICE: {r.read().decode()} - "
                  f"finishing current shard and stopping", flush=True)
            _stop.set()
            return
        except Exception:
            pass                                   # 404 = not interrupted
        _stop.wait(5)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def done_shards():
    """Which shards already exist. `aws s3 ls` exits 1 on an empty prefix, which
    is the normal state at the start of a run - so this must not treat that as
    an error."""
    out = sh(f"aws s3 ls {DEST}/ --region ap-northeast-1 2>/dev/null || true")
    return {int(w.split("shard-")[1].split(".")[0])
            for line in out.splitlines() for w in line.split() if "shard-" in w}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "refusing to run on CPU - this would be ~20x slower and silent"
    src = WORK / "chunks" / f"{ARM}.jsonl.gz"
    texts = [json.loads(l)["text"] for l in gzip.open(src, "rt", encoding="utf-8")]
    n = len(texts)
    nshard = (n + SHARD - 1) // SHARD
    have = done_shards()
    print(f"arm={ARM} model={MODEL} chunks={n:,} shards={nshard} already done={len(have)}",
          flush=True)

    m = SentenceTransformer(MODEL, device=dev, trust_remote_code=False,
                            model_kwargs={"torch_dtype": torch.float16})
    dims = (m.get_embedding_dimension() if hasattr(m, "get_embedding_dimension")
            else m.get_sentence_embedding_dimension())

    meta = {
        "arm": ARM, "model": MODEL, "dims": int(dims),
        "max_seq_length": int(m.max_seq_length),
        "doc_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX,
        "normalize": True, "dtype": "float16",
        "shard_size": SHARD, "chunks": n, "shards": nshard,
        # Passed in from the launching machine: the instance has no git repo, so
        # resolving it here always yielded null and silently lost the provenance
        # of which code produced these vectors.
        "git_sha": os.environ.get("GIT_SHA"),
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Write it only if absent. Two boxes worked arm A at once and the second
    # starter overwrote the first's meta.json - including its git_sha, which it
    # did not have. meta.json describes the vectors, and every shard of a given
    # arm+model is identical whoever computed it, so the FIRST description wins
    # and a later run must not clobber it.
    if sh(f"aws s3 ls {DEST}/meta.json --region ap-northeast-1 2>/dev/null || true").strip():
        # Resuming. SHARD must match what the existing shards were written with:
        # shard 5 covers chunks 10000-11999 at SHARD=2000 but 5000-5999 at
        # SHARD=1000, so a changed value makes done_shards() skip ranges that
        # were never embedded while the file present holds different rows.
        # Nothing errors anywhere - the index is just quietly wrong.
        sh(f"aws s3 cp {DEST}/meta.json {WORK}/prev.json --region ap-northeast-1")
        prev = json.loads((WORK / "prev.json").read_text())
        for k, mine in (("shard_size", SHARD), ("model", MODEL), ("chunks", n)):
            if prev.get(k) != mine:
                sys.exit(f"REFUSING to resume: existing shards have {k}={prev.get(k)!r}, "
                         f"this run has {mine!r}. Delete {DEST}/ or match the old value.")
        print("meta.json already present and consistent - leaving it alone", flush=True)
    else:
        (WORK / "meta.json").write_text(json.dumps(meta, indent=2))
        sh(f"aws s3 cp {WORK}/meta.json {DEST}/meta.json --region ap-northeast-1")
    print(json.dumps(meta, indent=2), flush=True)

    threading.Thread(target=spot_watch, daemon=True).start()

    t0, tdone = time.time(), 0
    for s in range(nshard):
        if s in have:
            continue
        if _stop.is_set():
            print("stopping before shard %05d on interruption notice" % s, flush=True)
            break
        lo, hi = s * SHARD, min((s + 1) * SHARD, n)
        try:
            v = m.encode([DOC_PREFIX + t for t in texts[lo:hi]], batch_size=BATCH,
                         normalize_embeddings=True, show_progress_bar=False)
            v = np.asarray(v, dtype=np.float16)
            assert v.shape == (hi - lo, dims), f"shape {v.shape} != {(hi-lo, dims)}"
            p = WORK / f"shard-{s:05d}.npy"
            np.save(p, v)
            # Upload only AFTER the rows are complete - this is the checkpoint.
            sh(f"aws s3 cp {p} {DEST}/shard-{s:05d}.npy --region ap-northeast-1 --only-show-errors")
            p.unlink()
        except Exception as e:
            err = {"shard": s, "lo": lo, "hi": hi, "error": f"{type(e).__name__}: {e}"}
            (WORK / "err.json").write_text(json.dumps(err, indent=2))
            sh(f"aws s3 cp {WORK}/err.json "
               f"s3://{BUCKET}/errors/embed/{ARM}/{SLUG}/shard-{s:05d}.json --region ap-northeast-1")
            print(f"  shard {s:05d} FAILED: {err['error']}", flush=True)
            continue
        tdone += 1
        el = time.time() - t0
        left = (nshard - len(have) - tdone) * (el / tdone)
        print(f"  shard {s:05d}  rows {hi-lo:>5}  {el/60:5.1f}m elapsed  "
              f"~{left/60:5.1f}m left", flush=True)

    _stop.set()
    final = done_shards()
    print(f"\n{len(final)}/{nshard} shards present. "
          f"{'COMPLETE' if len(final) == nshard else 'INCOMPLETE - re-run to resume'}",
          flush=True)
    sys.exit(0 if len(final) == nshard else 2)


if __name__ == "__main__":
    main()
