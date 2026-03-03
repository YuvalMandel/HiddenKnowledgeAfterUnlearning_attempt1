#!/usr/bin/env python3
"""
run_70b_distributed.py — parallel Llama-3-70B inference with file-locked work queue.

Multiple SLURM workers load the model independently and claim chunks of work
from a shared queue stored under checkpoints/llama70b_dist/.  fcntl.flock()
prevents two workers from claiming the same chunk.  Stale claims (>3 h) are
automatically reclaimed so preempted workers don't block the queue.

Modes
-----
  --init [--chunk N]   Build the work queue from current dataset splits (run once).
  --worker             Load model, process chunks until queue exhausted or preempted.
  --merge              Combine all chunk results → checkpoints/llama70b_results.json.
  --status             Print queue progress. Exit 0 = all done, 1 = work remains.
"""

import argparse
import fcntl
import json
import os
import pathlib
import random
import signal
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Lazy import of the main module — safe to do here because all side-effecting
# code in hidden_knowledge_after_unlearning.py is guarded by __name__=="__main__".
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hidden_knowledge_after_unlearning as hk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DIST_DIR    = hk.CHECKPOINT_DIR / "llama70b_dist"
QUEUE_PATH  = DIST_DIR / "queue.json"
LOCK_PATH   = DIST_DIR / "queue.lock"
RESULTS_DIR = DIST_DIR / "results"

CHUNK_SIZE    = 100        # examples per work unit (tune as needed)
STALE_SECONDS = 3 * 3600  # reclaim claims older than 3 hours

TASK_NAMES = ["bio_gen", "cyber_gen", "bio_mcq", "cyber_mcq", "bio_logit", "cyber_logit"]


# ===========================================================================
# Queue helpers
# ===========================================================================

def _load_queue() -> dict:
    return json.loads(QUEUE_PATH.read_text())


def _save_queue(q: dict):
    tmp = QUEUE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(q, indent=2))
    tmp.replace(QUEUE_PATH)


class _QueueLock:
    """Context manager that holds an exclusive flock on LOCK_PATH."""
    def __enter__(self):
        self._f = open(LOCK_PATH, "a")
        fcntl.flock(self._f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_):
        fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close()


def claim_next_chunk(worker_id: str) -> dict | None:
    """
    Return the next pending (or stale) chunk, marking it claimed in the queue.
    Returns None when no work remains.
    """
    with _QueueLock():
        q   = _load_queue()
        now = time.time()
        for task in TASK_NAMES:
            for chunk in q.get(task, []):
                if chunk["status"] == "done":
                    continue
                is_stale = (
                    chunk["status"] == "claimed"
                    and now - (chunk.get("claimed_at") or 0) > STALE_SECONDS
                )
                if chunk["status"] == "pending" or is_stale:
                    if is_stale:
                        print(f"[queue] Reclaiming stale chunk {chunk['id']} "
                              f"({(now - chunk['claimed_at']) / 3600:.1f}h old, "
                              f"was held by {chunk['worker']})")
                    chunk["status"]     = "claimed"
                    chunk["worker"]     = worker_id
                    chunk["claimed_at"] = now
                    _save_queue(q)
                    return dict(chunk)
    return None


def release_chunk(chunk_id: str):
    """Return a claimed chunk to pending (called on preemption before dying)."""
    with _QueueLock():
        q = _load_queue()
        for task in TASK_NAMES:
            for chunk in q.get(task, []):
                if chunk["id"] == chunk_id and chunk["status"] == "claimed":
                    chunk["status"]     = "pending"
                    chunk["worker"]     = None
                    chunk["claimed_at"] = None
                    _save_queue(q)
                    return


def mark_done(chunk_id: str):
    with _QueueLock():
        q = _load_queue()
        for task in TASK_NAMES:
            for chunk in q.get(task, []):
                if chunk["id"] == chunk_id:
                    chunk["status"] = "done"
                    _save_queue(q)
                    return


def queue_counts() -> tuple[int, int, int, int]:
    """Return (total, done, claimed, pending)."""
    q = _load_queue()
    total = done = claimed = pending = 0
    for task in TASK_NAMES:
        for chunk in q.get(task, []):
            total += 1
            s = chunk["status"]
            if s == "done":       done    += 1
            elif s == "claimed":  claimed += 1
            else:                 pending += 1
    return total, done, claimed, pending


# ===========================================================================
# --init
# ===========================================================================

def cmd_init(chunk_size: int):
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    LOCK_PATH.touch()

    print("Loading dataset splits to determine queue size...")
    rng = random.Random(hk.RANDOM_SEED)
    np.random.seed(hk.RANDOM_SEED)

    _, _, test_q     = hk.load_datasets(rng)
    csv_result       = hk.load_tf_pairs_from_csv()
    test_pairs       = (csv_result[2] if csv_result is not None
                        else hk.make_forget_pairs(test_q, rng))
    _, _, cyber_test = hk.load_cyber_tf_pairs(rng)
    bio_mcq_pairs    = hk.make_mcq_pairs(test_q)
    cyber_mcq_pairs  = hk.make_cyber_mcq_pairs()

    sizes = {
        "bio_gen":     len(test_pairs),
        "cyber_gen":   len(cyber_test),
        "bio_mcq":     len(bio_mcq_pairs),
        "cyber_mcq":   len(cyber_mcq_pairs),
        "bio_logit":   len(test_pairs),
        "cyber_logit": len(cyber_test),
    }
    print(f"Sizes: {sizes}")

    def _make_chunks(task: str, n: int) -> list[dict]:
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunks.append({
                "id":         f"{task}_{start}_{end}",
                "task":       task,
                "start":      start,
                "end":        end,
                "status":     "pending",
                "worker":     None,
                "claimed_at": None,
            })
        return chunks

    if QUEUE_PATH.exists():
        # Preserve completed chunks; reset stale claims to pending.
        existing = _load_queue()
        done_ids = {c["id"] for t in TASK_NAMES for c in existing.get(t, [])
                    if c["status"] == "done"}
        print(f"Existing queue found — {len(done_ids)} chunks already done (preserved).")
        queue = existing
        for task in TASK_NAMES:
            existing_ids = {c["id"] for c in existing.get(task, [])}
            for c in _make_chunks(task, sizes[task]):
                if c["id"] not in existing_ids:
                    queue.setdefault(task, []).append(c)
        # Reset any lingering claims so fresh workers can start immediately.
        for task in TASK_NAMES:
            for c in queue.get(task, []):
                if c["status"] == "claimed":
                    c.update(status="pending", worker=None, claimed_at=None)
    else:
        queue = {task: _make_chunks(task, sizes[task]) for task in TASK_NAMES}

    _save_queue(queue)
    total = sum(len(v) for v in queue.values())
    done  = sum(1 for t in TASK_NAMES for c in queue[t] if c["status"] == "done")
    print(f"Queue ready: {done}/{total} done, {total - done} pending.")


# ===========================================================================
# --worker
# ===========================================================================

_SHUTDOWN    = False
_ACTIVE_CHUNK: str | None = None  # chunk currently being processed


def _handle_preemption(signum, frame):
    global _SHUTDOWN
    print(f"\n[worker] Signal {signum} received (preemption/termination). "
          f"Will stop after current chunk.", flush=True)
    _SHUTDOWN = True


def cmd_worker():
    signal.signal(signal.SIGUSR1, _handle_preemption)
    signal.signal(signal.SIGTERM, _handle_preemption)

    worker_id = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    print(f"[worker {worker_id}] Starting", flush=True)

    # Load all dataset splits once — cheap, no GPU needed.
    rng = random.Random(hk.RANDOM_SEED)
    np.random.seed(hk.RANDOM_SEED)
    _, _, test_q     = hk.load_datasets(rng)
    csv_result       = hk.load_tf_pairs_from_csv()
    test_pairs       = (csv_result[2] if csv_result is not None
                        else hk.make_forget_pairs(test_q, rng))
    _, _, cyber_test = hk.load_cyber_tf_pairs(rng)
    bio_mcq_pairs    = hk.make_mcq_pairs(test_q)
    cyber_mcq_pairs  = hk.make_cyber_mcq_pairs()

    all_pairs = {
        "bio_gen":     test_pairs,
        "bio_logit":   test_pairs,
        "cyber_gen":   cyber_test,
        "cyber_logit": cyber_test,
        "bio_mcq":     bio_mcq_pairs,
        "cyber_mcq":   cyber_mcq_pairs,
    }

    tok = model = true_ids = false_ids = None

    try:
        while not _SHUTDOWN:
            chunk = claim_next_chunk(worker_id)
            if chunk is None:
                print(f"[worker {worker_id}] Queue exhausted — nothing more to do.", flush=True)
                break

            global _ACTIVE_CHUNK
            _ACTIVE_CHUNK = chunk["id"]

            task  = chunk["task"]
            start = chunk["start"]
            end   = chunk["end"]
            pairs = all_pairs[task][start:end]

            result_path = RESULTS_DIR / f"{chunk['id']}.npy"
            if result_path.exists():
                print(f"[worker {worker_id}] {chunk['id']}: result already on disk, skipping.")
                mark_done(chunk["id"])
                _ACTIVE_CHUNK = None
                continue

            print(f"[worker {worker_id}] {chunk['id']}: {len(pairs)} examples", flush=True)
            t0 = time.time()

            # Lazy model load — deferred until first real work unit.
            if model is None:
                print(f"[worker {worker_id}] Loading {hk.LLAMA70B_MODEL}...", flush=True)
                tok, model = hk.load_model_and_tokenizer(hk.LLAMA70B_MODEL)
                true_ids, false_ids = hk.get_tf_token_ids(tok)

            if task in ("bio_gen", "cyber_gen", "bio_mcq", "cyber_mcq"):
                result = hk.batch_generate(
                    model, tok, pairs,
                    hk.LLAMA70B_GEN_BATCH_SIZE,
                    f"w{worker_id}/{chunk['id']}",
                )
                np.save(result_path, np.array(result, dtype=object), allow_pickle=True)

            elif task in ("bio_logit", "cyber_logit"):
                result = hk.logit_tf_scores(
                    model, tok, pairs,
                    hk.LLAMA70B_LOGIT_BATCH_SIZE,
                    true_ids, false_ids,
                    f"w{worker_id}/{chunk['id']}",
                )
                np.save(result_path, np.array(result, dtype=float))

            mark_done(chunk["id"])
            _ACTIVE_CHUNK = None
            print(f"[worker {worker_id}] {chunk['id']}: done "
                  f"({time.time() - t0:.0f}s)", flush=True)

    except KeyboardInterrupt:
        print(f"[worker {worker_id}] KeyboardInterrupt.", flush=True)
    finally:
        if _ACTIVE_CHUNK:
            print(f"[worker {worker_id}] Releasing unfinished chunk {_ACTIVE_CHUNK}.")
            release_chunk(_ACTIVE_CHUNK)
        if model is not None:
            print(f"[worker {worker_id}] Unloading model.", flush=True)
            hk.unload_model(model)
            del tok, model

    total, done, _, pending = queue_counts()
    print(f"[worker {worker_id}] Exit. Queue: {done}/{total} done, {pending} pending.",
          flush=True)


# ===========================================================================
# --merge
# ===========================================================================

def cmd_merge():
    total, done, _, pending = queue_counts()
    if pending > 0:
        print(f"[merge] WARNING: {pending} chunks still pending — results will be incomplete.")
    else:
        print(f"[merge] All {done}/{total} chunks done. Merging...")

    q = _load_queue()

    def load_task(task: str) -> list:
        chunks = sorted(q[task], key=lambda c: c["start"])
        parts  = []
        for c in chunks:
            p = RESULTS_DIR / f"{c['id']}.npy"
            if not p.exists():
                print(f"[merge] MISSING result: {p}")
                continue
            arr = np.load(p, allow_pickle=True)
            parts.append(arr)
        if not parts:
            return []
        return np.concatenate(parts).tolist()

    rng = random.Random(hk.RANDOM_SEED)
    np.random.seed(hk.RANDOM_SEED)
    _, _, test_q     = hk.load_datasets(rng)
    csv_result       = hk.load_tf_pairs_from_csv()
    test_pairs       = (csv_result[2] if csv_result is not None
                        else hk.make_forget_pairs(test_q, rng))
    _, _, cyber_test = hk.load_cyber_tf_pairs(rng)
    bio_mcq_pairs    = hk.make_mcq_pairs(test_q)
    cyber_mcq_pairs  = hk.make_cyber_mcq_pairs()

    bio_gen_ans    = load_task("bio_gen")
    cyber_gen_ans  = load_task("cyber_gen")
    bio_mcq_ans    = load_task("bio_mcq")
    cyber_mcq_ans  = load_task("cyber_mcq")
    bio_logit      = load_task("bio_logit")
    cyber_logit    = load_task("cyber_logit")

    bio_gen_s     = hk.generation_stats(bio_gen_ans,   test_pairs)
    cyber_gen_s   = hk.generation_stats(cyber_gen_ans, cyber_test)
    bio_logit_s   = hk.logit_stats(bio_logit,          test_pairs)
    cyber_logit_s = hk.logit_stats(cyber_logit,        cyber_test)
    bio_mcq_s     = hk.mcq_gen_stats(bio_mcq_ans,      bio_mcq_pairs)
    cyber_mcq_s   = hk.mcq_gen_stats(cyber_mcq_ans,    cyber_mcq_pairs)

    results = {
        "bio_gen_stats":    bio_gen_s,
        "bio_logit_stats":  bio_logit_s,
        "bio_mcq_stats":    bio_mcq_s,
        "cyber_gen_stats":  cyber_gen_s,
        "cyber_logit_stats": cyber_logit_s,
        "cyber_mcq_stats":  cyber_mcq_s,
    }

    out = hk.CHECKPOINT_DIR / "llama70b_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[merge] Saved → {out}")
    print(f"  bio   gen={bio_gen_s['accuracy']:.3f}  "
          f"logit={bio_logit_s['accuracy']:.3f}  mcq={bio_mcq_s['accuracy']:.3f}")
    print(f"  cyber gen={cyber_gen_s['accuracy']:.3f}  "
          f"logit={cyber_logit_s['accuracy']:.3f}  mcq={cyber_mcq_s['accuracy']:.3f}")


# ===========================================================================
# --status
# ===========================================================================

def cmd_status(quiet: bool) -> int:
    if not QUEUE_PATH.exists():
        if not quiet:
            print("[status] Queue not initialised. Run --init first.")
        return 1

    total, done, claimed, pending = queue_counts()
    pct = 100 * done / total if total else 0

    if not quiet:
        print(f"Queue: {done}/{total} done ({pct:.1f}%)  "
              f"claimed={claimed}  pending={pending}")
        q = _load_queue()
        for task in TASK_NAMES:
            chunks = q.get(task, [])
            t = len(chunks)
            d = sum(1 for c in chunks if c["status"] == "done")
            c = sum(1 for c in chunks if c["status"] == "claimed")
            bar = "#" * d + "~" * c + "." * (t - d - c)
            print(f"  {task:14s}  {d:3d}/{t}  [{bar}]")

    return 0 if (pending == 0 and claimed == 0) else 1


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--init",   action="store_true", help="Initialise work queue")
    grp.add_argument("--worker", action="store_true", help="Run as an inference worker")
    grp.add_argument("--merge",  action="store_true", help="Merge results into checkpoint")
    grp.add_argument("--status", action="store_true", help="Print queue progress")
    ap.add_argument("--chunk", type=int, default=CHUNK_SIZE,
                    help=f"Examples per chunk (--init only, default {CHUNK_SIZE})")
    ap.add_argument("--quiet", action="store_true", help="Suppress output (--status only)")
    args = ap.parse_args()

    if args.init:
        cmd_init(args.chunk)
    elif args.worker:
        cmd_worker()
    elif args.merge:
        cmd_merge()
    elif args.status:
        sys.exit(cmd_status(args.quiet))


if __name__ == "__main__":
    main()
