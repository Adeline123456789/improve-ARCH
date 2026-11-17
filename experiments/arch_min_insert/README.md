# ARCH INSERT minimum experiment

This is an isolated low-level PPO experiment. It reuses the installed Isaac Sim 4.0 / Isaac Lab 1.1 environment, original UR10e/Hexagon assets, native differential IK and joint drives. No high-level DiT is trained.

**Execution status:** see [background_status.json](../../artifacts/arch_minexp/background_status.json), [screen_job.json](../../artifacts/arch_minexp/screen_job.json), and the generated [report](../../artifacts/arch_minexp/report.md). A running job is not a completed result.

The pretraining audit is [repo_audit.md](../../artifacts/arch_minexp/repo_audit.md). The [protocol](../../artifacts/arch_minexp/frozen_protocol.yaml) and validation/test manifests were frozen before main training. B0 uses world geometry/actions, B1 local geometry/actions, B2 additionally uses aligned tracking residuals. All have the same 27D actor/shared critic, 6D tip actions and two-layer 256-unit LSTM. There is no force input in this simulator policy/control chain.

Common corrections are limited to the new environment: one grasp constraint per environment; PhysX tensor updates for a kinematic board retaining its CAD/SDF; CAD-centered per-environment targets; isolated resets; a CAD-relative success evaluator. Old native R0 runs retain their identity. Recorded failures during preflight are excluded from main results.

From the repository root, using the existing `.venv-arch` (do not reinstall Isaac):

```bash
.venv-arch/bin/python experiments/arch_min_insert/test_geometry.py
bash experiments/arch_min_insert/run_smoke.sh --group B2 --num_envs 4 --steps 192 \
  --headless --output "$PWD/artifacts/arch_minexp/manual_smoke_new"
bash experiments/arch_min_insert/run_train.sh --group B0 --seed 0 --num_envs 512 \
  --transitions 1996800 --headless --output "$PWD/artifacts/arch_minexp/manual_train_new"
```

Output directories must not exist; this preserves previous results. The launcher sets CWD/PYTHONPATH, eight CPU threads and disabled W&B. One complete rollout is 51,200 transitions at 512 environments. Native MyA2CAgent is the actual optimizer, with eight PPO mini-epochs, horizon 100, minibatch 12,800 and sequence length 4.

After the corresponding checkpoint exists:

```bash
bash experiments/arch_min_insert/run_eval.sh --group B0 --seed 0 --num_envs 100 \
  --checkpoint "$PWD/artifacts/arch_minexp/checkpoints/B0_seed0_1996800.pth" \
  --manifest "$PWD/artifacts/arch_minexp/evaluation_manifest.jsonl" \
  --headless --output "$PWD/artifacts/arch_minexp/manual_eval_new"
.venv-arch/bin/python experiments/arch_min_insert/summarize_results.py
```

Evaluation uses deterministic mean actions, frozen normalization, identical manifest cases and per-episode RNN resets. All test trajectories remain local in each test run; fixed cases and the first up to five successes/failures per suite are exported under `artifacts/arch_minexp/traces/`. `actor_observation` is deployed input; `diagnostic_*` arrays are logging-only truth. Quaternions are wxyz; distances m, angles rad, times s, net forces N. Torque is unavailable. The NPZ contains explicit command/measurement times and case IDs.

The background driver runs equal seed-0 screening, validation, a budget-gated continuation, fixed final tests, result plots/tables and GitHub publication. If basic learning exists but full 45M confirmation cannot fit the six-hour wall deadline, it can add seeds 1/2 at the same 1,996,800 snapshot. If all nominal validation policies fail, it can instead perform a common longer screening from scratch, with separate identities. These are explicitly reported as budget-limited, not full 5M-per-seed confirmation.

Native checkpoints contain model, optimizer and normalization, but not complete live PhysX/contact and recurrent rollout state. `--checkpoint` can restore those stored training components; it restarts episodes and is not bitwise full-state continuation. The automatic experiment does not silently treat it as such.

`publish.py` stages only explicit experiment/evidence paths in a temporary Git index and pushes a new snapshot to the requested `improve-ARCH` repository. It leaves the user's staging area, existing modifications, original remote and checkout HEAD intact. The three preexisting logging changes are included in the published source snapshot for provenance; their original diff is preserved. It refuses to overwrite unexpected remote changes.
