# latentvans

Training and data scripts for the latentvans project (stage-0 latent grounding on Panda-70M clips).

## Layout

- `scripts/`: data download and staging, clip screening and verification, frame extraction, stage-0 training and evaluation (Slurm `.sh` wrappers + Python)

Not tracked (see `.gitignore`): `data/` (~61 GB), `checkpoints/` (~17 GB), `logs/`, `envs/`, wandb run dirs, `cookies.txt`.

## Upstream code

`code/VANS` is an unmodified clone of https://github.com/KlingAIResearch/VANS at commit `4a931a8`:

```bash
git clone https://github.com/KlingAIResearch/VANS.git code/VANS && git -C code/VANS checkout 4a931a8
```
