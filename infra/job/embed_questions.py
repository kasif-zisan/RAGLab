"""Phase 7 step 1: embed the 393 FinQA questions.

RUN:  python embed_questions.py        (MODEL optional, defaults to the indexed one)

Stored ONCE, under vectors/questions/<model>/, not per arm. The questions and
the model are identical for both arms, so one copy makes it structurally
impossible for the two arms to have been queried with different vectors - which
is the comparison the whole experiment rests on.

WHY IT LOOKS LIKE THIS:

  * THE QUERY PREFIX IS READ FROM meta.json, NOT TYPED HERE. Qwen3 prepends an
    instruction to queries and nothing to documents. If index time and query
    time disagree on that string, nothing errors - recall is just quietly worse.
    So the string comes from the file the index wrote.

  * IT CROSS-CHECKS BOTH ARMS' meta.json AND THE MODEL'S OWN CONFIG. Three
    independent sources must agree on model, dims, normalize and the query
    prompt. Any disagreement means the two indexes were not built the same way,
    and comparing them would be meaningless. Refuse rather than produce a
    number nobody can trust.

  * NO SHARDING. 393 vectors is 0.8 MB; sharding exists to make interruption
    cheap, and there is nothing here to lose. Still written only after it is
    complete, and skipped if already present - the same checkpoint rule.

  * POSITION IS IDENTITY. Row i of q.npy is qid i in questions/finqa.jsonl,
    which is row i of load_gold(). No ids are stored in the .npy.
"""
import json, os, pathlib, subprocess, sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BUCKET = "raglab-zisan-apne1"
WORK = pathlib.Path("/opt/raglab")
ARMS = ["B1_3200_ov889", "A_2400_ov360"]


def sh(cmd, check=False):
    """capture_output swallows stderr, so a failed `aws s3 cp` used to return an
    empty string and the script only noticed later, as a missing file. That cost
    a whole instance boot to discover a one-line IAM problem. `check=True`
    surfaces the real AWS error at the point of failure."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"COMMAND FAILED ({r.returncode}): {cmd}\n{r.stderr.strip()}")
    return r.stdout


def preflight(paths):
    """Check every S3 path this job needs BEFORE spending minutes on setup.
    Every failure here so far has been a prefix missing from the instance role,
    and the whole point is to find that in ten seconds, not in six minutes."""
    bad = []
    for p in paths:
        r = subprocess.run(f"aws s3 ls {p} --region ap-northeast-1",
                           shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"  {p}\n    {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'not found'}")
    if bad:
        sys.exit("PREFLIGHT FAILED - cannot read:\n" + "\n".join(bad))
    print(f"preflight: all {len(paths)} S3 paths readable")


def main():
    preflight([f"s3://{BUCKET}/questions/finqa.jsonl",
               f"s3://{BUCKET}/vectors/{ARMS[0]}/",
               f"s3://{BUCKET}/vectors/{ARMS[1]}/"])

    # CPU is allowed HERE, unlike embed.py. That assert exists because running
    # 43M corpus tokens on CPU would be ~20x slower and look merely slow. This
    # job is 393 questions, about 10k tokens - seconds either way. The vectors
    # are identical; only the speed differs.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    # --- the contract: both arms must agree, or the comparison is void --------
    metas = {}
    for arm in ARMS:
        dst = WORK / f"meta_{arm}.json"
        sh(f"aws s3 cp s3://{BUCKET}/vectors/{arm}/*/meta.json {dst} --region ap-northeast-1")
        if not dst.exists():                       # wildcards do not work in s3 cp
            slug = sh(f"aws s3 ls s3://{BUCKET}/vectors/{arm}/ --region ap-northeast-1"
                      ).split()[-1].strip("/")
            sh(f"aws s3 cp s3://{BUCKET}/vectors/{arm}/{slug}/meta.json {dst} "
               f"--region ap-northeast-1")
        metas[arm] = json.loads(dst.read_text())

    a, b = metas[ARMS[0]], metas[ARMS[1]]
    for k in ("model", "dims", "normalize", "dtype", "query_prefix", "doc_prefix"):
        if a[k] != b[k]:
            sys.exit(f"ARMS DISAGREE on {k}: {ARMS[0]}={a[k]!r} {ARMS[1]}={b[k]!r}. "
                     f"The two indexes were not built the same way; refusing.")
    MODEL, DIMS, QP = a["model"], a["dims"], a["query_prefix"]
    SLUG = MODEL.split("/")[-1]
    DEST = f"s3://{BUCKET}/vectors/questions/{SLUG}"
    print(f"both arms agree: model={MODEL} dims={DIMS} normalize={a['normalize']}")
    print(f"query_prefix from meta.json: {QP!r}")

    if sh(f"aws s3 ls {DEST}/q.npy --region ap-northeast-1 2>/dev/null || true").strip():
        print("q.npy already present - nothing to do"); return

    # --- questions ------------------------------------------------------------
    sh(f"aws s3 cp s3://{BUCKET}/questions/finqa.jsonl {WORK}/q.jsonl --region ap-northeast-1", check=True)
    rows = [json.loads(l) for l in (WORK / "q.jsonl").open(encoding="utf-8")]
    assert [r["qid"] for r in rows] == list(range(len(rows))), "qids must be 0..n-1 in order"
    print(f"questions: {len(rows)}")

    m = SentenceTransformer(MODEL, device=dev, trust_remote_code=False,
                            model_kwargs={"torch_dtype": torch.float16} if dev == "cuda" else {})

    # third source: the model's own configured prompt
    cfg = (m.prompts or {}).get("query")
    if cfg is not None and cfg != QP:
        sys.exit(f"meta.json query_prefix {QP!r} != model's configured query prompt "
                 f"{cfg!r}. One of them is wrong; refusing to guess.")
    print(f"model's own configured query prompt matches meta.json: {cfg is not None}")

    v = m.encode([QP + r["question"] for r in rows], batch_size=32,
                 normalize_embeddings=a["normalize"], show_progress_bar=False)
    v = np.asarray(v, dtype=np.float16)
    assert v.shape == (len(rows), DIMS), f"{v.shape} != {(len(rows), DIMS)}"

    n = np.linalg.norm(v.astype(np.float32), axis=1)
    print(f"shape {v.shape} dtype {v.dtype}  norms min {n.min():.4f} max {n.max():.4f}")
    print(f"NaN {int(np.isnan(v).sum())}  all-zero rows {int((n == 0).sum())}")
    assert not np.isnan(v).any() and (n > 0).all(), "bad vectors - refusing to upload"

    np.save(WORK / "q.npy", v)
    meta = {
        "kind": "questions", "dataset": "finqa", "source": "questions/finqa.jsonl",
        "model": MODEL, "dims": int(DIMS), "normalize": a["normalize"],
        "dtype": "float16", "query_prefix": QP, "questions": len(rows),
        "device": dev,
        "row_i_is": "qid i in questions/finqa.jsonl = row i of load_gold()",
        "arms_verified": ARMS, "git_sha": os.environ.get("GIT_SHA"),
    }
    (WORK / "qmeta.json").write_text(json.dumps(meta, indent=2))
    # upload only AFTER both are complete and checked
    sh(f"aws s3 cp {WORK}/q.npy {DEST}/q.npy --region ap-northeast-1 --only-show-errors")
    sh(f"aws s3 cp {WORK}/qmeta.json {DEST}/meta.json --region ap-northeast-1 --only-show-errors")
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {DEST}/q.npy  ({v.nbytes/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
