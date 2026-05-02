#!/usr/bin/env python3
"""Batch download -> extract -> probe for 13 missing models."""
import subprocess, time, datetime
from pathlib import Path
from huggingface_hub import snapshot_download

REPO   = Path("/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1")
PYTHON = "/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT = str(REPO / "inside_out_knowledge.py")
LOGS   = str(REPO / "inside_out_logs")
PROBE_FLAGS = "--domains bio --clfs LR --lcs layer --no_cv --no_cross_probe --n_jobs 4 --out_suffix layer"
CONDA  = "source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh && conda activate unlearning"

SLUGS = {
    "GradDiff": "graddiff", "PB_J": "pbj", "RMU": "rmu",
    "RMU-LAT": "rmu-lat", "ELM": "elm", "RR": "rr", "TAR": "tar",
}
BATCH1 = ["GradDiff_ck5", "PB_J_ck2",    "PB_J_ck5",  "RMU_ck1", "RMU_ck3", "RMU_ck8"]
BATCH2 = ["RMU-LAT_ck3",  "RMU-LAT_ck4", "ELM_ck4",   "RR_ck7",  "TAR_ck3", "TAR_ck4", "TAR_ck7"]

def model_repo(mid):
    method, ck = mid.rsplit("_ck", 1)
    return f"LLM-GAT/llama-3-8b-instruct-{SLUGS[method]}-checkpoint-{ck}"

def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)

def download(mid):
    repo = model_repo(mid)
    log(f"Downloading {mid} ({repo})")
    snapshot_download(repo)
    log(f"  OK: {mid}")

def sbatch_extract(mid):
    nm = mid.replace("-", "_")
    wrap = (f"set -e; {CONDA}; export HF_HOME=$HOME/.cache/huggingface; "
            f"cd {REPO}; echo Extract {mid}; "
            f"{PYTHON} {SCRIPT} --stage extract --model_id {mid} --domains bio; "
            f"echo Done {mid}")
    r = subprocess.check_output([
        "sbatch", "--parsable",
        f"--job-name=io_ext_{nm}",
        f"--output={LOGS}/ext_{nm}_%j.out",
        f"--error={LOGS}/ext_{nm}_%j.err",
        "--time=02:00:00", "--mem=32G", "--cpus-per-task=4",
        "--gres=gpu:L40:1", "--partition=public",
        f"--wrap={wrap}",
    ], text=True).strip()
    log(f"  {mid} ext={r}")
    return r

def sbatch_probe(mid, dep):
    nm = mid.replace("-", "_")
    wrap = (f"set -e; {CONDA}; export HF_HOME=$HOME/.cache/huggingface; "
            f"cd {REPO}; echo Probe {mid}; "
            f"{PYTHON} {SCRIPT} --stage probe --model_id {mid} {PROBE_FLAGS}; "
            f"echo Done {mid}")
    r = subprocess.check_output([
        "sbatch", "--parsable",
        f"--job-name=io_prb_{nm}",
        f"--output={LOGS}/prb_lyr_{nm}_%j.out",
        f"--error={LOGS}/prb_lyr_{nm}_%j.err",
        "--time=01:00:00", "--mem=32G", "--cpus-per-task=4",
        "--partition=public", f"--dependency=afterok:{dep}",
        f"--wrap={wrap}",
    ], text=True).strip()
    log(f"  {mid} prb={r}")
    return r

def wait_for_jobs(jids):
    id_str = ",".join(jids)
    log(f"Waiting for extract jobs: {id_str}")
    while True:
        out = subprocess.run(
            ["squeue", "-j", id_str, "-h"],
            capture_output=True, text=True
        ).stdout.strip()
        if not out:
            break
        time.sleep(60)
    log("Batch extracts complete.")

def process_batch(name, models):
    log(f"=== {name}: downloading {len(models)} models ===")
    for mid in models:
        download(mid)
    log(f"=== {name}: submitting SLURM jobs ===")
    ext_jids = []
    for mid in models:
        jext = sbatch_extract(mid)
        sbatch_probe(mid, jext)
        ext_jids.append(jext)
    wait_for_jobs(ext_jids)

process_batch("Batch1", BATCH1)
process_batch("Batch2", BATCH2)
log(f"All done. Run: {PYTHON} {SCRIPT} --stage aggregate")
