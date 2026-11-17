# INSERT repository audit (2026-11-17, Asia/Shanghai)

## Identity and preservation

Root: `/media/midea/61500ece-df8d-46ce-be73-e04b2378079c/ARCH`.
Initial HEAD: `a6a224b79f57e0b6c149e5b0f9d51a9b56ca131e`, branch `main`.
No registered Git submodules. Original remote: `https://github.com/Jiankai-Sun/ARCH.git`.
Requested publication target `Adeline123456789/improve-ARCH` was empty at inspection.
No on-disk AGENTS.md found within this checkout; the user's supplied AGENTS instructions apply.
No `.codegraph/` exists; graph MCP search/index/coverage tools are unavailable in this session.
Evidence below uses bounded source reads and direct simulator instrumentation, not a claimed graph index.

The preexisting tracked changes are three added lines: writer.flush in
`multi-robot-assembly/multi_arm_assembly/rl_components/my_agent.py` and optional
log-directory prefix in `train_rl.py`. They remain intact. See `preexisting.patch`
and `preexisting_status.txt` for the initial snapshot. Existing untracked scripts,
environment definitions, editor settings and previous outputs are preserved.
New code is isolated in `experiments/arch_min_insert/`; original task code is unchanged.

## Existing environment

`environment.json` records the live Python/package/GPU values. Warm-reused
`.venv-arch`: Python 3.10.20, PyTorch 2.3.0 / CUDA 12.1, NumPy 1.26.4,
Isaac Sim 4.0.0.0, Isaac Lab 1.1.0 (`omni-isaac-lab` 0.20.0), rl-games 1.6.1,
RTX 3090 24 GiB. Declarative environment hash is
`da783fc48273953c137d2ae0b939d69b12e1af600f58c49da8f5ea2583ecb902`, matching
`.aris/compute/local.md`. No dependency install, Isaac replacement or high-level training.
Existing IsaacLab source pin in environment spec: `454905bf08c374b6162e6c4c64a320aa6261fbe6`;
rsl_rl pin: `73fd7c621bf63104a8a7eb0c168df16c0ee65908` (not the active PPO runner).

Existing launcher `scripts/run_sim.sh` uses the venv, eight CPU threads,
`WANDB_MODE=disabled` and CWD `multi-robot-assembly/multi_arm_assembly`.
New `experiments/arch_min_insert/launch.sh` makes this explicit without relying on
untracked user scripts for published reproduction.

## Actual entry points

- `direct_isaac_lab_position.py:2471`: Gym `AssemblyInsert-v0` -> `InsertEnv`,
  `InsertEnvCfg`, and `assembly_insert_env_cfg.yaml`.
- `utils.py:1497` `AssemblyInsert(selected_obj_id=0)`: FMB Hexagon and board,
  one UR10e, already held via fixed joint, 60 mm retreat from the seated root pose.
- `train_rl.py:81` main: registers MyNetworkBuilder/MyA2CAgent, wraps with
  MyRlGamesVecEnvWrapper, trains with rl-games Runner. Native `--play -c ...`
  and `scripts/sim_smoke.py --checkpoint ...` replay the native policy.
- Research `run.py --mode train`: same actual MyA2CAgent and network builder,
  isolated MinInsertEnv. `--mode eval` uses MyPpoPlayerContinuous mean actions,
  model.eval and frozen normalization; no optimization.
- High-level DiT collection/training chain is unrelated and was not run.

Native reproduction used:

```
bash scripts/run_sim.sh ../../experiments/arch_min_insert/probe.py --headless \
  --mode reference --num_envs 20 --checkpoint "$PWD/outputs/ppo-100k/checkpoint.pth" \
  --output "$PWD/artifacts/arch_minexp/r0_native_v2"
```

## Timing and control

`direct_isaac_lab_position.py:146-150,548-549`: physics 1/120 s, decimation 5,
policy 1/24 s. INSERT horizon 4 s (96 policy steps). Native `_get_dones:1862`
sets BOTH terminated and truncated at `episode_length_buf >= max_episode_length-1`;
there is no success predicate, giving the reference 95 steps per episode.
Research uses 96 steps and separate success labels; fixed episode durations keep
rollout accounting simple. Reset initialization is not a training transition.

Native `_pre_physics_step_sr:1643` interprets actions as **pose increments**, not
m/s: XYZ times 0.05 m, Euler XYZ increments times 0.1 rad, orientation right
multiplied around tool0. It allocates seven action channels even when gripper
control is disabled; channel 7 is unused. The controller receives absolute tool0
pose via `DifferentialIKController.set_command`. `_apply_action:1790` recomputes
damped least-squares joint IK at each physics substep and calls
`set_joint_position_target`. No force feedback is used by this simulator controller.
Arm implicit drives: stiffness 400, damping 800, effort limit 100, velocity limit 20.
The arm and peg have gravity disabled. Pose writes happen in initialization/reset,
not each policy execution step. Contact can impede execution (paired probe).

Research actions have six channels, equal Euclidean displacement/rotation limits,
SO(3) left world increment, and a common tip-to-tool converter. B0 expresses the
increment in W; B1/B2 in estimated S. The actual absolute Cartesian command is
stored before the five physical substeps; endpoint measurement produces the
residual. Per-substep joint targets are also available in native reference traces.
There is no target filter/interpolator in the reused native controller.

## Actor, critic, force and memory

Native actual observation is 15D (verified checkpoint load): 3D translation and
first two rotation-matrix columns (6D) from `des_world_T_tool * inverse(measured_tool)`;
plus 6D `J.T @ projected_joint_forces`. It is NOT a physically validated wrist
wrench: general recovery of Cartesian wrench does not equal multiplication by J.T.
Position units are m, rotation columns dimensionless; the latter six quantities
must not be labeled calibrated N/Nm. No velocity, q/dq, target history or gripper
channel is enabled in this native configuration. The relative transform is not
simply target-minus-position, and its translation depends on the reference origin.
Native target comes directly from fixed WORLD_T_HOLE_START, so layout randomization
would need a common correction. Native reward `insert_relative_pose:2130` also
uses this fixed target and negative norm of weighted position/axis-angle error,
clipped to [-2,2]; no force penalty. Native done uses time only.

Research actor (27D): position error 3 [m], SO3 error 3 [rad], tip linear velocity
3 [m/s], angular velocity 3 [rad/s], socket world rotation columns 6, gravity
unit vector 3, tracking residual 6 [m/rad]. All geometry uses measured robot FK
and X_hat, estimated socket, and fixed CAD target. Velocity includes omega cross
lever arm. Ground-truth peg/socket/grasp/contact are used solely for physical reset,
reward, success and logs; no privileged critic. The wrapper has no `critic` field,
and the shared actor/critic network therefore gets exactly these deployable sources.
B0/B1 residual channels are zeros. All groups retain identical global socket
orientation channels; full actor input is not claimed SE(3)-invariant.

`assembly_insert_env_cfg.yaml`: two-layer LSTM, 256 units, before MLP,
concat_input=true, MLP 256/128/64 with ELU, sequence length 4. `my_agent.py`
`zero_rnn_on_done` defaults true and clears each ended environment's state;
evaluation explicitly does the same. Input/value running normalization is trained
on current PPO rollouts; evaluation player is in eval mode. All research policies
start from scratch, with the same network dimensions and optimizer configuration.
Research minibatch = 25*num_envs (1600 at 64 envs), horizon 100, eight mini-epochs;
this common adjustment avoids native 400 optimizer steps per rollout.

The separate real-robot `araas_environment.py` has `update_force_real`, measured
force sensors and compliance equations (around lines 340-536 and 1098-1116).
It is NOT this simulator entry point. This experiment can establish an Actor and
simulator execution path without FT; it cannot establish a tested FT-free real robot.

## Randomization, physics and common fixes

Native easy reset keeps target/layout fixed, zeroes grasp perturbations and updates
all fixed joints regardless of env_ids. Native reset also advances the whole scene
while resetting a subset, invalidating simple per-env timestamps. Research reset
writes only requested env IDs, does not advance global physics, and masks residual
history. The first physical substep holds reset joints while FK/Jacobians refresh.
A reusable PyBullet UR10e IK solver replaces native solve_ik for reset only;
the native helper opens connections without disconnecting and reseeds global RNGs.

A new pretraining audit found 63 grasp fixed joints for 32 native-style cloned
environments: every clone inherited env_0's joint, then native code added `_1`.
Evidence: `smoke32_v1/fixed_joints.json`. With grasp bias, these two constraints can
conflict. Research overrides creation to define exactly one per environment, with
per-env local-frame attributes. Pre-fix fixtures/smoke are excluded from results.

True board layout, episode-fixed socket bias and true grasp bias are independently
sampled by local RNG in `scenarios.py`. X_hat stays nominal. Startup is estimated
socket * preinsert relation * inverse(X_hat), and physical peg uses X_true. Bias
is right-multiplied local translation plus SO3 rotation (explicitly not a coupled
six-dimensional SE3 exponential). True layout is applied to the static board USD
parent; actual reset q, FK, all transforms and validity are saved in manifests.

CAD `asset_audit.json` and extracted vertices: hexagon spans root z +/-50 mm;
board spans z -25.640 to +24.360 mm. Rounded hex hole center is
(-68.1833,-88.5441) mm in board coordinates, versus native target (-69.5,-89.5) mm.
Research corrects this ~1.63 mm target offset identically for all groups.
Socket functional origin is on the board top, +Z points into it. P is CAD bottom
root z=-50 mm, oriented into insertion; seated depth 44.360 mm. Native PEG_T_TIP
(-10 mm) is a controller convenience and is not the CAD bottom.

## Success and data

Native checkpoint `outputs/ppo-100k/checkpoint.pth` has 100,000 transitions,
training seed 42; SHA256 `8fee6540281152bda36a04164e6c00b77d48b97af9d4448f036a159fe1e8656d`.
Its 15D/7D native semantics are incompatible with research 27D/6D. R0 is not B0.
20 native scenes were replayed with the unchanged policy/control semantics.
Provisional pose-success count was 0/20; no validated physical SR is assigned to R0.
They are deterministic repeated native scenarios with saved actual states, not
20 independent randomized layouts. Every trajectory and native observations,
actual Cartesian/joint commands, measurements and diagnostic forces are retained.

Research success requires CAD-relative depth within 1 mm, lateral error <=0.75 mm
(under half the ~1.49 mm side clearance), orientation error <=2 degrees under C6
symmetry, and five steps (=0.2083 s). Invalid overshoot >3 mm rejects success.
Physical seating/edge/penetration fixtures must pass before training. Initial
invalid states are recorded before a penetrating object can be expelled by PhysX.
Net force is a 3D world-frame peg contact measurement; torque is unavailable.
Its net magnitude does not quantify all local squeezing forces.

Existing keyboard demonstrations are segmented NPZs with state/action/skill and
optional RGB/depth (see existing KEYBOARD_COLLECTION.md). They were not used for
PPO, which samples fresh online rollouts. They do not establish insertion success.

## Final pretraining revision v3

The layout fixture disproved the earlier static-parent update assumption: a known
seated peg at a rotated/translated hole generated >2 kN contact and was expelled.
`fixtures_layout_v2` is failed diagnostic evidence, not a main result. The board
is now a kinematic rigid body with its original CAD/SDF; reset updates its actual
PhysX root tensor. Reward/success read this actual physical board state. The same
layout fixture passes in `fixtures_layout_v3`. Peg-board contact filtering uses
the rigid board root; the previous mesh-path filter returned zeros and is excluded.

Final paired contact (`contact_common_v3`): free motion 2.86 mm versus blocked
0.27 mm, residual ~1.971 versus 1.998 mm, free board force 0 and blocked peak
1.33–1.35 N. True fixed-grasp agreement in `smoke64_v3` has position-error median
~0.006 mm; this checks that sampled grasp biases reach physical constraints.

Final compute choice: 512 environments, minibatch 12,800, all other common PPO
settings unchanged. `smoke512_v3` ran 98,304 transitions / 1,024 completed episodes;
`ppo_pilot_v3` actually trained 51,200 transitions with seed 999 at ~1,618
transitions/s. Pilot is excluded from the main comparison. Native limits, gravity,
friction and mass are preserved; measured peg mass 0.1155683 kg, static/dynamic
friction 0.5/0.5, restitution 0. `physics_snapshot.json` in each main run records
actual properties. The native grasp helper and board-transform defects are fixed
only in the research environment. Original tracked modifications remain untouched.
