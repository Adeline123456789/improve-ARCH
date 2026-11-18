# ARCH v2 插入实验

总状态 **IN_PROGRESS**。G0 PASS，G1 PASS，G2 RUNNING，G3 NOT_RUN，G4 NOT_RUN。

## 控制链与学习链

控制频率 24 Hz，物理步长 1/120 s，decimation 5。每回合 96 个控制步，共 4 s。从孔口外约 15.64 mm 出发，到 CAD 装配目标的轴向行程为 60 mm。每步以实测位姿为基准，平移和旋转目标偏移的范数上限分别为 50 mm、0.1 rad。6 个臂关节的刚度、阻尼、力矩上限实际读为 400、800、100。

最后 5 个控制步均须满足横向误差 ≤0.75 mm、深度误差绝对值 ≤1 mm、按六重对称计算的姿态误差 ≤2°，并且全回合无已记录的无效状态。上游初始化完成后重新注入并回读参数，固定时域标记从 false 改为 true，截止时清零 bootstrap，普通 rollout 边界保留后续价值。

两个版本的脚本均在同一名义几何的 20 个正常起点回合获得 20/20，逐步轨迹离线复算与在线标签一致。每版 G1 各 3,200 个训练环境步，真实 PPO 回放、均值头、探索参数、重载动作和物理命令均已核对。固定绝对目标的末段跟踪残差为 2.49 mm。[G0 控制证据](run_20261118_alignment/g0.json) · [G1 更新证据](run_20261118_alignment/g1/ppo_updates.jsonl)

## 名义 B0

初始配置 B0 seed 101 在 1,433,600 步内的七个预定检查点均为 0/20，512 环境曾在启动阶段中止。6 条脚本轨迹训练的独立 BC 诊断闭环为 14/20。第二配置调整横向与深度代价的相对尺度，将初始高斯探索标准差从 1 降到 0.25，PPO 从头训练。两项调整的效果无法单独归因。[初始配置报告](run_20261118/report.md) · [修复依据](run_20261118_alignment/repair.json)

第二配置各预定检查点的累计名义环境步与成功数如下。

204,800 → 0/20, 409,600 → 0/20, 614,400 → 0/20, 819,200 → 0/20, 1,024,000 → 0/20, 1,228,800 → 0/20, 1,433,600 → 0/20, 1,638,400 → 20/20, 1,843,200 → 18/20, 2,048,000 → 20/20, 2,252,800 → 16/20。

连续两个检查点至少 19/20 加独立重载的门槛尚未通过。

20 个名义回合是相同语义任务的工程回归。逐回合结果含失败、到位后失稳、误差和饱和率。[第二配置结果](run_20261118_alignment/results.csv) · [名义曲线](run_20261118_alignment/diagnostics/g2_curve.png)

## 轻度扰动

预定完整范围为板布局 X、Y 各 ±20 mm、偏航 ±15°，socket 估计 X、Y 各 ±1 mm，grasp 平移三轴各 ±0.5 mm，两者三轴旋转估计各 ±0.5°。横向误差容限为 0.75 mm，完整估计偏差可以跨越这一容限。课程分 0.25、0.5、1.0 倍。

尚无通过物理核验的轻扰动开发分数。

## 正式比较

| 训练种子 | 组 | N | T_id | T_layout | T_bias |
|---|---|---|---|---|---|
| 0 | B0 | 未运行 | 未运行 | 未运行 | 未运行 |
| 0 | B1 | 未运行 | 未运行 | 未运行 | 未运行 |
| 0 | B2 | 未运行 | 未运行 | 未运行 | 未运行 |
| 1 | B0 | 未运行 | 未运行 | 未运行 | 未运行 |
| 1 | B1 | 未运行 | 未运行 | 未运行 | 未运行 |
| 1 | B2 | 未运行 | 未运行 | 未运行 | 未运行 |
B1−B0、B2−B1、B2−B0 及其工程信号均无实测值。

B1−B0 同时改变观测与动作坐标；B2−B1 才能检验因果残差的增量。两个训练种子对学习稳定性的代表性有限。

## 资源与复现

已完成的在线训练共 3,692,800 环境步。GPU 占用区间取并集约 1.572 小时，含初始化及验证。首个仿真至最后一次记录约 1.727 小时。未结束任务的步数下界和各进程计时来源见账本。[预算](budget_ledger.json) · [软件与资产](environment_provenance.json)

若名义姿态在 2° 边界附近反复失稳，可对已保存的成功和失败轨迹比较首次到位后的角误差与真实控制响应。若名义持续合格而轻扰动欠佳，按真实布局与 socket/grasp 估计偏差分层检查首次进入、末段保持与实际抓取变换。

在仓库根目录查看实时进度或复现已测检查点

```bash
o="$PWD/artifacts/arch_minexp_v2/run_20261118_alignment"
proto="$PWD/experiments/arch_min_insert/configs/protocol_v2_alignment.yaml"
python3 experiments/arch_min_insert/status_v2.py "$o" --follow
# 查看原始后台日志
tail -F "$o/training.log"
```

```bash
bash experiments/arch_min_insert/launch.sh --mode eval --group B0 --seed 101 --num_envs 20 --headless \
  --protocol "$proto" --manifest "$o/manifests/validation_nominal.jsonl" \
  --checkpoint "$o/g2_continue_to_2252800/runner/run/nn/last_apa_impedance_ep_176_rew_-19.127636.pth" --output "$o/replay_2252800_$(date +%s)"
```

本机 128 环境名义训练吞吐约 715 环境步/s，续训一个 204,800 步区间的纯训练约 4.8 分钟，另计启动、独立评估。

以下顺序复现第二配置已完成的名义续训段。新输出目录会从头执行 G0/G1。G3、G4 的入口分别验证上游门槛。

```bash
new="$PWD/artifacts/arch_minexp_v2/alignment_reproduction"
py=.venv-arch/bin/python
gate=experiments/arch_min_insert/gate_v2.py
$py -u "$gate" --protocol "$proto" --output "$new"
$py -u "$gate" --protocol "$proto" --output "$new" --extend-nominal-checks 2
$py -u "$gate" --protocol "$proto" --output "$new" --extend-nominal-checks 4
$py -u experiments/arch_min_insert/curriculum_v2.py --protocol "$proto" --output "$new"
$py -u experiments/arch_min_insert/formal_v2.py --protocol "$proto" --output "$new"
```

正式固定课程 B 与轻扰动吞吐尚未同时获得。六个模型仅重训到第二配置最近完成的名义步数，纯训练估计至少 5.25 GPU 小时，轻度课程与测试另计。

