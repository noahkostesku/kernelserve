---
description: Submit a SLURM job to Narval, verify queue position, and set up output streaming
argument-hint: <bench|serving> [--dry-run]
allowed-tools: [Bash, Read]
---

# /slurm-submit — SLURM Job Submitter

Job type: $ARGUMENTS

## Steps

1. Parse job type from $ARGUMENTS: `bench` → `slurm/bench_job.sh`, `serving` → `slurm/serving_job.sh`
2. If `--dry-run` is present, run `bash -n slurm/<job>.sh` to syntax-check only and print the `#SBATCH` directives
3. Otherwise:
   a. Run `sbatch slurm/<job>.sh` and capture the job ID
   b. Run `squeue -u $USER -j <job_id>` to show initial queue position
   c. Print: `tail -f $SCRATCH/kernelserve-logs/<job_id>-out.txt` as the command to stream live output
4. Remind the user to run `seff <job_id>` after the job completes to check CPU/GPU efficiency

## Guardrails

- If the script targets sm_80 only and the cluster appears to be Nibi (H100), warn before submitting
- Never submit with `--time` > 3:00:00 without prompting for explicit confirmation
- Always show the estimated queue wait time if `squeue` output is available
