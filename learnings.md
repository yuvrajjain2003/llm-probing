# Learnings

Things we've learned and mistakes not to repeat. Add new entries at the bottom with a date.

- 2026-10-03: `sinfo`/`scontrol` hide GPU partitions by default. Use `sinfo -a` / `scontrol --all show partition h100_all`.
- 2026-10-03: `batch` partition needs ≥48 CPUs and `small` needs ≥2, or the job is rejected (`QOSMinCpuNotSatisfied`).
- 2026-10-03: Use `h100_all`, not `h100` (the latter gives "Invalid qos specification").
- 2026-10-03: From `runs/goal2/*/_verify/cmp.txt`: EasySteer activations are shifted vs HuggingFace — decoder layer L → ES index L-1, token position t → ES position t+1. Final layer differs (pre- vs post-norm).
- 2026-10-03: The project folder is owned by `yuvraj`, so git needed `git config --global --add safe.directory /proj/ssrivalab/projects/llm-probing` (done).
- 2026-10-03: `env.sh` defaults to `/work/users/r/i/rishp` (doesn't exist for us) and silently falls back to `<repo>/.work`. Check `WORK_DIR`/`RUNS_ROOT` before launching anything.
- 2026-10-03: The repo's `.claude/STATE.md` describes the old UNC box (2026-09-18). The Sycamore `gpqa25` runs in `runs/` are newer and not described there.
