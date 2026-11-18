# ARCH v2 名义插入验证进度

初始本地 HEAD 为 a6a224b79f57e0b6c149e5b0f9d51a9b56ca131e。工作树包含已有修改和未跟踪实验源码，记录见 source_snapshot.json 和 preexisting_status.txt。发布分支基于目标仓库的 2e4cc7a。

G0 在正常孔外起点使用实际执行器完成 20/20 次脚本插入。20 次属于同一名义几何的工程回归。物理步长 1/120 秒，decimation 5，控制周期 1/24 秒，共 96 步。以实测位姿为基准，平移、旋转目标偏移上限分别为 50 mm、0.1 rad。六个臂关节实测驱动参数为 400、800、100。每回合截止时必须连续 5 步满足横向误差 ≤0.75 mm、深度误差 ≤1 mm、六重对称姿态误差 ≤2°，且全回合无已知无效物理状态。轨迹和离线复算见 g0_scripted 和 g0.json。

G1 使用 B0 seed 101，共 3,200 个训练环境步，完成两个 rollout 和 64 次小批量更新。PPO 回放结果与均值头、探索尺度的更新量见 g1/ppo_updates.jsonl。短诊断模型重载后名义成功 0/20，同输入的动作和命令最大差均为 0。G1 用于验证学习链。两轮采样的动作饱和率较高，G2 保留原配置建立对照。

G2 首次 512 环境在 PhysX 复制初始化阶段以 SIGABRT 退出，日志包含 malloc(): invalid size。未进入策略 rollout。并行度改为 128，B0 从头训练，实际 rollout 为 12,800 个环境步。每 16 个 rollout 保存预定检查点，共 7 次，在独立进程重置并验证。检查点间隔 204,800 步，首轮总预算 1,433,600 步。G3/G4 等待基础学习门槛。最终保留集尚未启用。

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
