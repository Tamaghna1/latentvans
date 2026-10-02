"""Non-fatal W&B logging for one-shot jobs (analysis, diagnostics).

Same rule as train_stage0_latent_grounding.py: W&B is an extra view, never the
source of truth (every job still writes its own JSON), and any W&B failure is a
warning, not a crash. Requires `wandb login` on the cluster (already done; the key
lives in ~/.netrc) or WANDB_API_KEY in the job environment.
"""
import os
import time


def log_summary(project, run_name, config, summary, tables=None, job_type=None):
    """Create one W&B run holding `summary` (flat dict of scalars) and optional
    `tables` ({name: (columns, rows)}), then finish it. Returns the run URL or None."""
    if not project:
        return None
    try:
        import wandb

        name = run_name or f"{job_type or 'job'}-{os.environ.get('SLURM_JOB_ID', time.strftime('%Y%m%d-%H%M%S'))}"
        run = wandb.init(project=project, name=name, config=config, job_type=job_type, reinit=True)
        run.summary.update(summary)
        run.log(summary)
        for table_name, (columns, rows) in (tables or {}).items():
            run.log({table_name: wandb.Table(columns=columns, data=rows)})
        url = run.url
        run.finish()
        print(f"W&B run logged: {url}", flush=True)
        return url
    except Exception as e:  # never let W&B take a finished job down with it
        print(f"WARNING: W&B logging failed ({e}); results are still in the job's JSON output.", flush=True)
        return None


def flatten(d, prefix=""):
    """{'a': {'b': 1}} -> {'a/b': 1}, keeping only numbers (for run.summary)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out
