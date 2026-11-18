# B0 名义插入，第二个训练配置

首轮七个检查点各 0/20。第二配置使用同一正常孔外起点、真实执行器和有限时域终点，将位置代价分为绝对深度误差除以 60 mm 加横向误差除以 12 mm，初始高斯探索标准差由 1 降为 0.25。其余奖励项、保持时间、成功阈值沿用首轮协议，两项调整的影响无法单独归因。

实测上游初始化给出 1/120 s 物理步长、5 倍 decimation、4 s 时限及 400/800/100 的臂关节驱动参数。协议在初始化之后再次设定这些数值，有限时域标记由 false 改为 true，成功终奖只在第 96 个控制步结算。真实 deadline 的后续价值为零，普通未结束 rollout 的边界保留 bootstrap。平移与旋转目标偏移上限是 50 mm 与 0.1 rad。

G0 脚本 20/20 成功。20 个案例重复同一个物理名义几何。G1 完成 3,200 个训练步、64 次优化器更新，真实采样初始标准差为 0.25，两个 rollout 的最大概率回放误差均为 1.91×10⁻⁶。训练和重载评估对相同输入、LSTM 状态和统计量的执行命令差为 0。

G2 从头训练 B0 seed 101。819,200 到 1,024,000 步间，验证的末段三维位置误差中位数由 18.39 mm 降至 1.86 mm，之后在 1,228,800 步达到 1.59 mm。该检查点 20 次均到达孔口，末段横向、深度、姿态各自符合阈值的回合数为 2、1、0，末段稳定成功为 0/20。G2 的完整更新情况见 gate_report.json 和 diagnostics/g2_curve.json。

在仓库根目录实时查看训练与验证

```bash
tail -F artifacts/arch_minexp_v2/run_20261118_alignment/g2_nominal/episodes.jsonl
tail -F artifacts/arch_minexp_v2/run_20261118_alignment/diagnostics/checkpoints.log
```

从头复现本配置并重跑受影响关卡

```bash
.venv-arch/bin/python -u experiments/arch_min_insert/gate_v2.py \
  --protocol experiments/arch_min_insert/configs/protocol_v2_alignment.yaml \
  --output artifacts/arch_minexp_v2/alignment_reproduction
```

1,433,600 步时，末段横向与深度各 20/20 合格，姿态 0/20 合格，角误差中位数 2.87°。按原配置从该检查点续训到 1,843,200 步；另计新增 409,600 步。恢复时核对模型、标准化器、全部 19 份优化器状态及优化器步数 3,584，逐项相等。仿真与 LSTM 从新回合起步，依据见 g2_continue_to_1843200/resume_check.json。

本次续跑命令（已有任务可查看对应目录的 episodes.jsonl）

```bash
.venv-arch/bin/python -u experiments/arch_min_insert/gate_v2.py \
  --protocol experiments/arch_min_insert/configs/protocol_v2_alignment.yaml \
  --output artifacts/arch_minexp_v2/run_20261118_alignment \
  --extend-nominal-checks 2
```
