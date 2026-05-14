# KernelServe Git Workflow

Reference for branch naming, commit format, PR rules, and release tagging.
Injected into Claude's context — keep rules machine-parseable: use flat headers,
explicit examples, and avoid prose-only explanations.

---

## 1. Branch Naming

Format: `<prefix>/<short-description>`

| Prefix | Use case | Example |
|---|---|---|
| `feat/` | New features | `feat/rms-norm-cuda-oxide` |
| `fix/` | Bug fixes | `fix/triton-backend-batch-size` |
| `bench/` | Benchmark runs and analysis | `bench/rms_norm-cuda-oxide` |
| `obs/` | Observability changes | `obs/otel-latency-histogram` |
| `chore/` | Scaffolding, deps, config | `chore/uv-lockfile-update` |

**Never work directly on `main`.**

---

## 2. Commit Message Format

Follows a Conventional Commits variant:

```
<type>(<scope>): <imperative description under 72 chars>
```

### Types

`feat` `fix` `bench` `obs` `chore` `docs` `test`

### Scopes

`kernel` `serving` `slurm` `otel` `mlflow` `ci`

### Examples

```
feat(kernel): add fused rms-norm cuda-oxide implementation
fix(serving): resolve triton backend max_batch_size misconfiguration
bench(rms_norm): log cuda-oxide vs triton p99 latency to mlflow
obs(otel): add histogram span for kernel dispatch latency
chore(ci): pin cuda/12.2 module load in bench-report workflow
docs(kernel): document correctness threshold for rms-norm
test(kernel): add fp32 max-abs-error assertions for rms-norm
```

### Rules

- Description: imperative mood, lowercase, no trailing period
- No `Co-Authors` lines — `settings.json` enforces this at commit time
- Body is optional; use it only for non-obvious WHY (constraint, workaround, incident)

---

## 3. Pull Request Rules

### All PRs

- Must pass all CI checks: lint (`ruff`, `clippy`), type check (`mypy`), unit tests (`pytest -m "not gpu"`)
- Squash merge to `main`; delete branch after merge

### Benchmark PRs (`bench/`)

- Must include the MLflow comparison table in the PR body
- Table is auto-posted by `bench-report.yml` when a `bench/` branch pushes benchmark results
- Do not merge if the MLflow experiment name doesn't follow the five-segment format:
  `kernelserve/<kernel>/<backend>/<cluster>/<YYYY-MM>`

### Serving PRs (`feat/` or `fix/` touching `serving/`)

- Must include a locust load test result in the PR body
- Show at minimum: RPS, p50, p99, error rate

---

## 4. Worktree Workflow

Worktrees let Claude (and humans) work on multiple branches simultaneously without
stashing or switching.

```bash
# Start a worktree session for a task
claude --worktree <branch-name>

# Clean up stale worktrees (run weekly)
git worktree prune
```

**Limits and rules:**

- Run at most **3 parallel worktree sessions** (API rate-limit sanity)
- `.claude/worktrees/` is gitignored — never commit worktree directories
- Subagent delegation already uses worktrees; see CLAUDE.md → Subagent delegation

---

## 5. Using Claude with Git History

Claude can read `git log` and `git diff` output directly. Pass the raw command
output in the prompt or ask Claude to run it via Bash.

### Example prompts

```
Read git log --oneline -20 and summarize what changed in the kernel layer.
```

```
git diff main..HEAD -- kernels/ and explain the changes.
```

```
Find the commit that changed rms_norm.rs and explain why.
```

```
Summarize all bench/ commits merged since 2026-04-01.
```

### Useful one-liners to paste into context

```bash
git log --oneline -20
git diff main..HEAD -- kernels/
git log --all --oneline -- kernels/cuda_oxide/src/rms_norm.rs
git log --since="2026-04-01" --oneline --grep="bench"
```

---

## 6. Release Tagging

Format: `v<phase>.<milestone>.<patch>`

| Segment | Meaning | Example |
|---|---|---|
| `phase` | Project phase (1 = Narval A100) | `1` |
| `milestone` | Milestone number within phase | `2` |
| `patch` | Patch / hotfix increment | `0` |

**Example:** `v1.2.0` — Phase 1, milestone 2, initial release.

### Tagging procedure

1. Confirm all milestone kernels are benchmarked and merged to `main`
2. Verify MLflow experiment runs are present for every kernel/backend pair
3. Create an annotated tag:
   ```bash
   git tag -a v1.2.0 -m "phase 1 milestone 2: rms-norm and softmax benchmarked on narval"
   git push origin v1.2.0
   ```
4. Do **not** tag pre-benchmark or draft benchmark branches

---

## Quick Reference

```
Branch   feat/ fix/ bench/ obs/ chore/
Commit   <type>(<scope>): <imperative, <72 chars>
Merge    squash → main; delete branch
CI gate  ruff + clippy + mypy + pytest -m "not gpu"
Worktree max 3 parallel; prune weekly
Tags     v<phase>.<milestone>.<patch> after milestone benchmarked + merged
```
