"""Measure embedding throughput for the candidate models on the REAL chunks.

This is not the embedding run. It exists to turn an estimate into a measurement
before we commit to a full pass, because the only honest way to size a job is to
run a slice of the actual work.

WHY IT LOOKS LIKE THIS:
  * It reads the real chunks, not synthetic text. Transformer cost scales with
    TOKENS, and our chunks are long (p50 well under the cap, tail near 800
    tokens). Benchmarking on short strings would overstate throughput several
    times over.
  * It reports tokens/sec, not chunks/sec. Chunks/sec is not transferable
    between arms - A averages 1,280 chars and B1 averages 2,392.
  * It records max_seq_length per model. A model silently truncating our input
    is the failure that produces no error and a mediocre recall@k weeks later,
    so the number goes in the output where it can be read back.
  * fp16 on GPU: a T4 has tensor cores and half the memory traffic. Accuracy
    impact on embeddings is negligible; speed impact is not.
"""
import gzip, json, os, pathlib, time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BUCKET = "raglab-zisan-apne1"
WORK = pathlib.Path("/opt/raglab")
SAMPLE = int(os.environ.get("SAMPLE", "2000"))

# (hf id, doc prefix, query prefix, needs trust_remote_code)
# Prefixes are NOT interchangeable between models - each is defined by that
# model's training. They live here, next to the model they belong to, and get
# copied into meta.json so query time reads the same strings index time used.
CANDIDATES = [
    ("Snowflake/snowflake-arctic-embed-m-v2.0", "",                   "query: ",        True),
    ("nomic-ai/nomic-embed-text-v1.5",          "search_document: ",  "search_query: ", True),
    ("Alibaba-NLP/gte-modernbert-base",         "",                   "",               False),
    ("Qwen/Qwen3-Embedding-0.6B",               "",                   "",               False),
]


def _mean_len(path, probe=500):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        ls = [len(json.loads(l)["text"]) for l, _ in zip(f, range(probe))]
    return sum(ls) / len(ls)


def load_sample(path, n):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if len(out) >= n:
                break
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  {torch.cuda.get_device_name(0) if dev=='cuda' else ''}", flush=True)

    files = sorted(WORK.glob("chunks/*.jsonl.gz"))
    print(f"chunk files: {[f.name for f in files]}", flush=True)
    # Benchmark on the arm with the LONGEST CHUNKS, not the biggest FILE. The
    # first version picked by file size and chose A - whose file is larger only
    # because it has 2.6x more chunks, while its chunks are HALF the length.
    # Attention cost grows faster than linearly in sequence length, so sizing a
    # run from short chunks overstates throughput for long ones.
    pick = os.environ.get("ARM")
    if pick:
        src = next(f for f in files if f.name.startswith(pick))
    else:
        src = max(files, key=_mean_len)
    texts = load_sample(src, SAMPLE)
    chars = sum(len(t) for t in texts)
    print(f"sample: {len(texts):,} chunks from {src.name}, {chars/1e6:.1f}M chars "
          f"(~{chars/4/1e6:.1f}M tokens)\n", flush=True)

    results = {}
    for hf_id, doc_p, qry_p, remote in CANDIDATES:
        print(f"=== {hf_id} ===", flush=True)
        try:
            t0 = time.time()
            m = SentenceTransformer(hf_id, device=dev, trust_remote_code=remote,
                                    model_kwargs={"torch_dtype": torch.float16} if dev == "cuda" else {})
            load_s = time.time() - t0
            maxlen = m.max_seq_length
            dims = (m.get_embedding_dimension() if hasattr(m, 'get_embedding_dimension')
                    else m.get_sentence_embedding_dimension())

            tok = m.tokenizer
            ntok = sum(len(tok(t, truncation=True, max_length=maxlen)["input_ids"]) for t in texts)
            over = sum(1 for t in texts
                       if len(tok(t, truncation=False)["input_ids"]) > maxlen)

            t0 = time.time()
            v = m.encode([doc_p + t for t in texts], batch_size=32,
                         normalize_embeddings=True, show_progress_bar=False)
            dt = time.time() - t0

            r = {
                "model": hf_id, "dims": int(dims), "max_seq_length": int(maxlen),
                "doc_prefix": doc_p, "query_prefix": qry_p, "trust_remote_code": remote,
                "load_s": round(load_s, 1), "sample_chunks": len(texts),
                "sample_tokens": int(ntok), "encode_s": round(dt, 1),
                "tokens_per_s": round(ntok / dt),
                "chunks_per_s": round(len(texts) / dt, 1),
                "truncated_chunks": int(over),
                "full_run_est_min_54M": round(54.4e6 / (ntok / dt) / 60, 1),
                "dtype": "float16" if dev == "cuda" else "float32",
                "vec_dtype": str(v.dtype), "vec_shape": list(v.shape),
            }
            results[hf_id] = r
            print(json.dumps(r, indent=2), flush=True)
            if over:
                print(f"  !! {over}/{len(texts)} sample chunks EXCEED max_seq_length "
                      f"={maxlen} and were truncated", flush=True)
            del m
            torch.cuda.empty_cache()
        except Exception as e:                      # one bad model must not end the run
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results[hf_id] = {"model": hf_id, "error": f"{type(e).__name__}: {e}"}

    out = WORK / "bench.json"
    out.write_text(json.dumps(results, indent=2))
    os.system(f"aws s3 cp {out} s3://{BUCKET}/bench/candidates.json")
    print("\n=== summary ===", flush=True)
    for k, r in results.items():
        if "error" in r:
            print(f"{k:45} FAILED")
        else:
            print(f"{k:45} {r['tokens_per_s']:>7,} tok/s  "
                  f"{r['dims']:>5}d  maxlen={r['max_seq_length']:<6} "
                  f"trunc={r['truncated_chunks']:<4} full run ~{r['full_run_est_min_54M']} min")


if __name__ == "__main__":
    main()
