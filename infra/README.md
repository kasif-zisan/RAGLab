# Batch job on EC2 spot — a reusable pattern

These five files run a long, embarrassingly-parallel batch job on a throwaway EC2
instance and get the results into S3. They were written for parsing 23,000 pages
of PDF with Docling, but the shape is generic: **anything that is a list of
independent units of work, where each unit takes seconds to minutes.**

Document parsing, image processing, transcoding, bulk embedding, scraping a URL
list, running a model over a dataset — all the same shape.

If you are a Claude instance picking this up on a new project: read this file
first. Most of what is below was learned by something failing, and several of the
failures were silent.

---

## The architecture

```
S3  s3://<bucket>/
      input/<unit>            the work items
      job/run.py              the job — NOT baked into user-data
      models/<cache>.tar      heavy downloads, cached after the first run
      output/<unit>.json      one object per completed unit = output AND checkpoint
      errors/<unit>.json      failures, deliberately a DIFFERENT prefix
      logs/<instance-id>.log  pushed every 2 min, final copy on exit

EC2 spot instance
      instance profile (IAM role) — never access keys
      user-data = bootstrap.sh
      ProcessPoolExecutor over units
      self-terminates when done; deadman timer as backstop
```

### The one idea everything else hangs off

**The output file IS the checkpoint.** `output/<unit>.json` existing means that
unit finished. There is no separate state file, no database, no lock.

Three consequences:

1. **Write it only AFTER the unit completes.** Write it first and an interruption
   mid-unit leaves a checkpoint saying "done" for work that is half-finished — and
   a re-run skips it forever, silently.
2. **Failures go to a different prefix.** A failure written to `output/` would be
   read as "done" and never retried.
3. **Re-running is always safe.** It does whatever is not done yet. This is what
   makes spot interruptions a non-event: you lose only the units in flight.

On S3 this is enough. `PutObject` is atomic — a reader never sees a partial
object. **On a local filesystem it is not**: a half-written file looks identical
to a complete one, so there you need write-to-temp-then-rename.

---

## What to change for a new project

| File | Change |
|---|---|
| all | `BUCKET`, `REGION`, the `--profile` in `P=(...)` |
| `bootstrap.sh` | the `pip install` line; the model-cache block if your job has no big download |
| `launch.sh` | `AMI`, `SG`, `SUBNET`, instance profile name, instance type |
| `watch.sh` | the total unit count (hardcoded `196`), prefix names |
| `run.py` | `parse_one()` — everything else is scaffolding |

Resolve the AMI from an SSM public parameter rather than hardcoding an ID; a
pinned AMI silently ages out of patch coverage and is region-specific.

---

## Lessons, each one learned the hard way

### Cost and safety

- **`shutdown -h now` at the end of the script is not enough.** It only runs if
  the script *reaches* the end. A hung job never does, and the box bills until
  someone notices. Schedule `shutdown -h +N` at boot as well — it fires
  regardless of what the job does.
- **Terminate on shutdown.** Set `--instance-initiated-shutdown-behavior
  terminate` or the instance stops rather than terminating, and the EBS volume
  keeps billing.
- **Check for orphans after a run**, not just for running instances: unattached
  EBS volumes and still-active spot requests both bill with no instance in sight.

### Credentials

- **An EC2 instance cannot use an IAM user.** Users and roles are different
  principals. The instance needs a role delivered via an instance profile; boto3
  then picks up temporary, auto-rotating credentials from the metadata service
  with nothing on disk. Putting long-lived access keys on a spot instance is the
  thing to never do.
- Your own user needs **`iam:PassRole`** to attach a role to an instance. It is a
  separate permission and the error message does not say so clearly.
- **SSM Session Manager instead of SSH.** Attach `AmazonSSMManagedInstanceCore`
  and the security group needs **zero inbound rules**. No key pair to manage.

### Visibility — the expensive one

- **A user-data script that fails takes its error message to the grave.** You
  cannot SSH into a box that already shut down, and cloud-init's log dies with the
  disk. Put an `EXIT` trap on the script that uploads the log to S3 on every path,
  including failure. This is the single highest-value line in `bootstrap.sh`.
- **The trap only fires at exit.** On a job you walk away from, that is no
  visibility for hours. Push the log to S3 on a background timer too.
- **Print a banner before the first command that can fail.** A script that exits
  silently is unreportable.

### Bash traps that cause silent death

- **`aws s3 ls` exits 1 on an empty prefix.** Under `set -euo pipefail` that kills
  the script instantly, before any output. Cost us a launch that returned a bare
  prompt with no message.
  - Inside `if aws s3 ls ...; then` it is safe — `set -e` does not fire in a
    condition. Bare, it is not.
  - Guard with `$( { cmd || true; } | wc -l )`. **Put `|| true` inside the pipe.**
    `cmd | wc -l || echo 0` makes `wc` print `0` and then `echo` print `0` again,
    giving a two-line value.
- **`--max-items` appends the pagination token to `--output text`.** You get a
  stray `None` line in your variable. Use `| head -1`, or the API's own
  `--max-results`.

### Python / ML runtime

- **`libGL.so.1: cannot open shared object file`** — OpenCV on a headless server
  image. `apt-get install libgl1 libglib2.0-0t64`. Hits anything importing `cv2`,
  which includes most document and vision pipelines.
- **Install CPU-only torch explicitly** (`--index-url
  https://download.pytorch.org/whl/cpu`) *before* the package that depends on it.
  The default wheel pulls ~2.5 GB of CUDA onto a box with no GPU.
- **`torch.set_num_threads(1)` is not sufficient.** ONNX Runtime, OpenBLAS, MKL
  and OpenCV each keep their own pool. They size it **at import time from the
  environment**, so the limits must be `export`ed in the shell before Python
  starts. Symptom: load average 124 on a 16-vCPU box — saturated, but spending the
  time context-switching.
- **Size workers by RAM, not cores**, for any job holding model weights per
  process. Then *measure* RSS and re-size: we assumed ~3 GB per worker, measured
  2.8–5.0 GB, and 14 workers left 9 GB free on a 64 GB box. An OOM kill breaks
  `ProcessPoolExecutor` and ends the **whole run**, not one unit.
- **boto3 clients are not fork-safe.** Create one inside each worker process.
- **Cache heavy model downloads in S3**, but **guard the seed with a size check**.
  We nearly uploaded an empty 204 KB tar, which every future instance would have
  "restored" successfully and silently fallen back to re-downloading. Restoring
  nothing is not an error.

### Keep what you cannot recover later

- **Provenance is discarded by default and cannot be rebuilt.** Docling's
  `export_to_markdown()` returns one string per document with no page
  boundaries; `page_break_placeholder="<!-- PAGEBREAK -->"` is the only way to
  keep them. Without page numbers the best citation you can give a user is
  "somewhere in this 91-page document" - and citation is the single feature that
  makes an imperfect RAG system trustworthy, because it lets the user verify in
  five seconds. We threw away 35 documents of work relearning this.
  **Before a long run, ask what the output cannot tell you afterwards.**
- **Capture the library's own warnings per unit.** Parsers usually say when they
  are unsure - docling logs every table cell it drops. In an interleaved
  N-worker log those lines cannot be attributed to a unit, so attach a logging
  handler inside the worker and store the result with that unit's output. It is
  a free per-unit confidence score and it costs nothing to collect.

### Progress estimates

- **Estimate from work done, not units done.** `elapsed / completed x remaining`
  assumes units finish one after another. With N workers, *nothing* can complete
  for the first full unit-time, so the early estimate is enormous - ours reported
  555 minutes against a real 168. Units also vary in size (our documents ranged
  26 to 182 pages), so unit count is a biased measure regardless.
- Count the total work up front when it is cheap (page counts come from the PDF
  xref in milliseconds) and report `remaining_work / (work_done / elapsed)`.

### Method

- **Validate on one unit before scaling.** A fifteen-minute instance parsing two
  pages caught: the libGL error, a wrong CLI command, a bug in our own exit code,
  and proved the trap and self-termination worked. All four would otherwise have
  surfaced hours into the real run.
- **Score against a label if you have one.** Do not eyeball output and call it
  good. And make the scoring function *break* when the thing you care about
  breaks — a loose string match passes on garbage.
- **Instance size trades time, not money.** For a fixed parallelisable job you pay
  per vCPU-hour, so a bigger box costs about the same total and finishes sooner.
  The only knob that changes the bill is spot vs on-demand (~3x).
- **Parallelism stops scaling at the number of units.** With 196 documents and 32
  workers you end up waiting on whichever worker drew the long ones while the rest
  idle. Past that point you need to split work *within* a unit, which breaks the
  one-file-per-unit checkpoint.
