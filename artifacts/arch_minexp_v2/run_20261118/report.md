# ARCH v2 名义插入验证进度

初始本地 HEAD 为 a6a224b79f57e0b6c149e5b0f9d51a9b56ca131e。工作树包含已有修改和未跟踪实验源码，记录见 source_snapshot.json 和 preexisting_status.txt。发布分支基于目标仓库的 2e4cc7a。

G0 在正常孔外起点使用实际执行器完成 20/20 次脚本插入。20 次属于同一名义几何的工程回归。物理步长 1/120 秒，decimation 5，控制周期 1/24 秒，共 96 步。以实测位姿为基准，平移、旋转目标偏移上限分别为 50 mm、0.1 rad。六个臂关节实测驱动参数为 400、800、100。每回合截止时必须连续 5 步满足横向误差 ≤0.75 mm、深度误差 ≤1 mm、六重对称姿态误差 ≤2°，且全回合无已知无效物理状态。轨迹和离线复算见 g0_scripted 和 g0.json。

G0 自由空间中，沿世界 Z 轴给定 ±2 mm 目标偏移，实测每控制步平均位移为 +0.051 mm 和 −0.026 mm。同为 −2 mm 的持续命令在偏置板面夹具上施加 96 步，总轴向位移约 −0.276 mm，板面接触力峰值为 1.35 N。固定绝对目标的命令位置变化小于 0.00012 mm，最终平移跟踪误差约 2.49 mm。三类实测响应见 g0_probe/trajectories.npz 和 diagnostics/g0_free_response.csv。训练和验证清单各有 20 个不同 ID，各自仅有一个物理名义场景。

G1 使用 B0 seed 101，共 3,200 个训练环境步，完成两个 rollout 和 64 次小批量更新。PPO 回放结果与均值头、探索尺度的更新量见 g1/ppo_updates.jsonl。短诊断模型重载后名义成功 0/20，同输入的动作和命令最大差均为 0。G1 用于验证学习链。两轮采样的动作饱和率较高，G2 保留原配置建立对照。

G2 首次 512 环境在 PhysX 复制初始化阶段以 SIGABRT 退出，日志包含 malloc(): invalid size。未进入策略 rollout。并行度改为 128，B0 从头训练，实际 rollout 为 12,800 个环境步。每 16 个 rollout 保存预定检查点，共 7 次，在独立进程重置并验证。检查点间隔 204,800 步，首轮总预算 1,433,600 步。G3/G4 等待基础学习门槛。最终保留集尚未启用。

G2 的前两个预定检查点均为 0/20。204,800 步模型在接近孔口后退出，最接近目标时的三维距离中位数为 45.38 mm，末段为 81.64 mm。409,600 步对应为 53.50 mm 和 113.08 mm。两次验证都未达到完整几何条件。完整记录见各自的 g2_eval 目录，汇总距离由 trajectory_diagnosis.py 从实测轨迹复算。

614,400 步检查点同样为 0/20。作为额外诊断，从 6 条当前 G0 脚本轨迹的 27 维部署观测和有效六维动作拟合同容量 B0 网络，另用 2 条名义轨迹核对，离线动作 RMSE 为 0.0453。该 BC 模型独立闭环评估达到 14/20，全部曾到达几何范围，6 次最终失败的深度误差绝对值中位数为 1.53 mm。该模型只用于诊断，PPO 训练仍从随机初始化开展。详情见 diagnostics/bc/fit.json、bc_replay/episodes.jsonl。

首轮完整结果为七个检查点各 0/20。最后两次检查均到达孔口，最大深度中位数为 1.85 和 1.65 mm，末段姿态误差分别为 10.12° 和 10.53°。两次峰值板面接触力中位数为 80.47 N 和 76.65 N。这里能确认接触板面和进度受限，不能据此判定孔内卡死。曲线见 diagnostics/g2_curve.png。本配置结论为 G2 FAIL；第二配置将独立记录。G1 与首轮 G2 合计新增 1,436,800 个训练步。

在仓库根目录查看后台进展

```bash
tail -F artifacts/arch_minexp_v2/run_20261118/training.log
tail -F artifacts/arch_minexp_v2/run_20261118/g2_nominal/episodes.jsonl
```

重新执行完整关卡链请使用新输出目录。已完成且匹配的子任务可用其原输出目录继续。

```bash
.venv-arch/bin/python -u experiments/arch_min_insert/gate_v2.py \
  --protocol experiments/arch_min_insert/configs/protocol_v2.yaml \
  --output artifacts/arch_minexp_v2/reproduction
```
