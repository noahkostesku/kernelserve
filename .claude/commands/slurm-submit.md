---
description: Submit a benchmark or serving job to the Alliance HPC cluster (Narval)
argument-hint: [bench|serving] [--dry-run]
allowed-tools: [Bash, Read, Edit]
---

# /slurm-submit — SLURM Job Submitter

## Steps

1. If $ARGUMENTS does not specify `bench` or `serving`, ask:
   "What type of job? (bench / serving)"

2. Ask: "How many GPUs? (1 / 2 / 4 / 8)" — default 1.

3. Ask: "Time limit?" — default `2:00:00` for bench, `12:00:00` for serving.
   Warn before accepting any value over `3:00:00` for bench or `24:00:00` for serving.

4. Fill the appropriate template:
   - bench → `slurm/templates/bench_job.sh`
   - serving → `slurm/templates/serving_job.sh`

   Substitute `{{NGPUS}}` and `{{TIMELIMIT}}` in the template.
   Always include `--account=def-cbravo` and module loads:
   ```bash
   module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0 python/3.11
   ```

5. Print the filled script in full and ask: "Submit this job? (y/n)"
   Do not run sbatch unless the user confirms.

6. If `--dry-run` is in $ARGUMENTS, run syntax check instead and stop:
   ```bash
   bash -n slurm/templates/<type>_job.sh
   ```

7. Submit:
   ```bash
   sbatch slurm/templates/<type>_job.sh
   ```
   Capture and print the job ID from the output line `Submitted batch job <ID>`.

8. Show queue position:
   ```bash
   squeue -u $USER -j <job_id>
   ```

9. Print the commands to follow the job:
   ```
   squeue -u $USER && sattach <job_id>.0
   ```
   Remind: run `seff <job_id>` after completion to review CPU/GPU efficiency.

## Guardrails

- If the target cluster appears to be Nibi (H100 / sm_90), warn and abort — Phase 2 is not active.
- Never hardcode `sm_80` or `sm_90` in scripts; read GPU arch from SLURM env vars.
