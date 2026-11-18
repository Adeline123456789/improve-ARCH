# 估计位姿几何反馈在 G3 开发场景中的表现

现有几何反馈在四分之一强度开发集成功 100/100，在半强度成功 93/100。两组各 100 个场景全部通过已实现的物理有效性检查。该控制器已经能完成这两档的大部分插入。半强度仍有 7 个失败，本次实验覆盖范围限于这两份开发清单。

| 扰动倍率 | scripted | B0 该档最佳开发结果 | 两者都成功 | 仅 scripted 成功 | 仅 B0 成功 | 两者均失败 |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 | 100/100 | 59/100（3,686,400 步） | 59 | 41 | 0 | 0 |
| 0.5 | 93/100 | 48/100（4,710,400 步） | 45 | 48 | 3 | 4 |

B0 的最佳检查点按对应开发集分数选取。与半强度末次 B0（5,939,200 步，26/100）相比，脚本多成功 67 个百分点。comparison.json 记录与两档全部 16 个 B0 检查点的场景配对比较。

成功要求截止时连续 5 步满足横向、深度和对称姿态三项阈值。半强度 7 个失败均为超时。末步横向误差为 0.777 至 1.125 mm，超过 0.75 mm 阈值，深度与姿态均在容差内。脚本动作由实测末端位姿、估计 grasp、估计 socket 和 CAD 装配目标构造。对两组共 19,200 次实际动作，以记录的估计误差复算，最大绝对差小于 1.5×10⁻⁸。GT 用于物理场景设置及评分。两次评估均无 checkpoint，奖励、动作限幅、成功判据和 96 步时限与对应 B0 一致。

rung_0、rung_1 分别保存逐回合 episodes.jsonl、完整轨迹 trajectories.npz、运行配置和离线几何复核 grasp_verification.json。frozen_checkpoints/index.json 记录名义通过、半强度最佳及最后四个退化检查点的路径与 SHA-256。模型复制后逐个校验并在本机设为只读。此次新增训练步数为零，G4 未启动。

在仓库根目录复现，两组使用相同的仿真种子 101。输出目录应为新路径。

```bash
set -e
src="$PWD/artifacts/arch_minexp_v2/run_20261118_alignment"
out=$(mktemp -d "$PWD/artifacts/arch_minexp_v2/scripted_replay.XXXXXX")
for rung in 0 1; do
  bash experiments/arch_min_insert/launch.sh --mode eval --group B0 --scripted \
    --seed 101 --headless --num_envs 100 --protocol "$src/g3_rung_${rung}.yaml" \
    --manifest "$src/manifests/g3_validation_r${rung}.jsonl" --output "$out/rung_${rung}"
done
```
