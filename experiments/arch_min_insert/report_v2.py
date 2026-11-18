"""Build a report from observed gate results, including all unsuccessful attempts."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from budget_v2 import budget

def read(path):
    return json.loads(path.read_text())
def dump(path,record):
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
def report(out):
    parent=out.parent
    previous=parent/'run_20261118'
    r=read(out/'gate_report.json')
    led=budget([previous,out])
    dump(parent/'budget_ledger.json',led)
    g2,g3,g4=(r[k] for k in ('G2','G3','G4'))
    overall=r.get('overall_status','IN_PROGRESS')
    chosen=g2['curve'][-1] if g2['status']=='PASS' else None
    reload=read(out/f"g2_reloaded_{chosen['frame']}/completion.json") if chosen else None
    compact={k:v for k,v in led.items() if k!='jobs'}
    meta=dict(status=overall,first_attempt=read(previous/'gate_report.json')['G2']['status'],
              current={k:r[k] for k in ('G0','G1','G2','G3','G4')},budget=compact,formal=[])
    observed=g4.get('runs',[])
    pairs={}
    suites=['test_E0','test_E1','test_E2','test_E3']
    labels=['N','T_id','T_layout','T_bias']
    for seed in [0,1]:
        pairs[seed]={}
        for group in ['B0','B1','B2']:
            row=next((x for x in observed if x['seed']==seed and x['group']==group),None)
            values=row['counts'] if row else {}
            meta['formal'].append(dict(seed=seed,group=group,status='COMPLETED' if row else 'NOT_RUN',
                suites={label:values.get(suite) for label,suite in zip(labels,suites)}))
            if row:pairs[seed][group]=[100*values[k]['successes']/values[k]['episodes'] for k in suites]
    deltas={}
    for upper,lower in [('B1','B0'),('B2','B1'),('B2','B0')]:
        key=upper+'-'+lower
        deltas[key]={str(seed):[a-b for a,b in zip(groups[upper],groups[lower])] if upper in groups and lower in groups else None for seed,groups in pairs.items()}
        one,two=deltas[key]['0'],deltas[key]['1']
        mean=[(a+b)/2 for a,b in zip(one,two)] if one and two else None
        deltas[key]['two_seed_mean_pp']=mean
        idx={'B1-B0':2,'B2-B1':3}.get(key)
        deltas[key]['engineering_signal']=(one[idx]>0 and two[idx]>0 and mean[idx]>=10 and mean[0]>=-5) if idx and mean else None
    meta['formal_differences_pp_in_N_Tid_Tlayout_Tbias_order']=deltas
    dump(parent/'gate_report.json',meta)
    stats=led['completed_training_steps']
    t=led['gpu_wall_seconds_union_estimate']/3600
    c=g2['curve']
    nominal=', '.join(f"{entry['frame']:,} → {entry['successes']}/20" for entry in c)
    lines=['# ARCH v2 插入实验',f"总状态 **{overall}**。G0 {r['G0']['status']}，G1 {r['G1']['status']}，G2 {g2['status']}，G3 {g3['status']}，G4 {g4['status']}。",'## 控制链与学习链',
        '控制频率 24 Hz，物理步长 1/120 s，decimation 5。每回合 96 个控制步，共 4 s。从孔口外约 15.64 mm 出发，到 CAD 装配目标的轴向行程为 60 mm。每步以实测位姿为基准，平移和旋转目标偏移的范数上限分别为 50 mm、0.1 rad。6 个臂关节的刚度、阻尼、力矩上限实际读为 400、800、100。',
        '最后 5 个控制步均须满足横向误差 ≤0.75 mm、深度误差绝对值 ≤1 mm、按六重对称计算的姿态误差 ≤2°，并且全回合无已记录的无效状态。上游初始化完成后重新注入并回读参数，固定时域标记从 false 改为 true，截止时清零 bootstrap，普通 rollout 边界保留后续价值。',
        '两个版本的脚本均在同一名义几何的 20 个正常起点回合获得 20/20，逐步轨迹离线复算与在线标签一致。每版 G1 各 3,200 个训练环境步，真实 PPO 回放、均值头、探索参数、重载动作和物理命令均已核对。固定绝对目标的末段跟踪残差为 2.49 mm。[G0 控制证据](run_20261118_alignment/g0.json) · [G1 更新证据](run_20261118_alignment/g1/ppo_updates.jsonl)',
        '## 名义 B0',
        '初始配置 B0 seed 101 在 1,433,600 步内的七个预定检查点均为 0/20，512 环境曾在启动阶段中止。6 条脚本轨迹训练的独立 BC 诊断闭环为 14/20。第二配置调整横向与深度代价的相对尺度，将初始高斯探索标准差从 1 降到 0.25，PPO 从头训练。两项调整的效果无法单独归因。[初始配置报告](run_20261118/report.md) · [修复依据](run_20261118_alignment/repair.json)',
        '第二配置各预定检查点的累计名义环境步与成功数如下。',
        nominal+'。',
        (f"在 {chosen['frame']:,} 步满足连续两个检查点至少 19/20，选定模型在独立进程重载后达到 {reload['successes']}/20。模型 SHA-256 为 `{chosen['sha256']}`。" if chosen else '连续两个检查点至少 19/20 加独立重载的门槛尚未通过。'),
        '20 个名义回合是相同语义任务的工程回归。逐回合结果含失败、到位后失稳、误差和饱和率。[第二配置结果](run_20261118_alignment/results.csv) · [名义曲线](run_20261118_alignment/diagnostics/g2_curve.png)',
        '## 轻度扰动',
        '预定完整范围为板布局 X、Y 各 ±20 mm、偏航 ±15°，socket 估计 X、Y 各 ±1 mm，grasp 平移三轴各 ±0.5 mm，两者三轴旋转估计各 ±0.5°。横向误差容限为 0.75 mm，完整估计偏差可以跨越这一容限。课程分 0.25、0.5、1.0 倍。']
    for entry in g3.get('curve',[]):
        block=g3['curve'].index(entry)+1
        lines.append(f"- {entry['frame']:,} 步，倍率 {entry['scale']}，名义 {entry['nominal_successes']}/20，独立轻扰动 {entry['mild_successes']}/100。[物理抓取与轨迹核对](run_20261118_alignment/g3_mild_b{block}/grasp_verification.json)")
    if not g3.get('curve'):lines.append('尚无通过物理核验的轻扰动开发分数。')
    for p in sorted((out/'manifests').glob('g3_validation_r*_audit.json')):
        v=read(p)
        lines.append(f"{p.stem} 中按估计值准确到达时，有 {v['ideal_estimated_goal_outside_true_geometry']}/100 个场景仍超出真实几何条件，横向误差第 90 百分位为 {v['p90_ideal_goal_lateral_mm']:.3f} mm。")
    if g3.get('reason'):lines.append('G3 停止原因为 '+g3['reason']+'。')
    lines+=['## 正式比较','| 训练种子 | 组 | N | T_id | T_layout | T_bias |','|---|---|---|---|---|---|']
    for entry in meta['formal']:
        vals=[f"{v['successes']}/{v['episodes']}" if v else '未运行' for v in entry['suites'].values()]
        lines.append('| '+' | '.join(map(str,[entry['seed'],entry['group'],*vals]))+' |')
    if not observed:lines.append('B1−B0、B2−B1、B2−B0 及其工程信号均无实测值。')
    else:
        for key,values in deltas.items():
            lines.append(f"{key} 百分点差值（N、T_id、T_layout、T_bias），种子 0 为 {values['0']}，种子 1 为 {values['1']}，两种子均值为 {values['two_seed_mean_pp']}。工程信号为 {values['engineering_signal']}。缺失项以 None 标记。")
    lines+=['B1−B0 同时改变观测与动作坐标；B2−B1 才能检验因果残差的增量。两个训练种子对学习稳定性的代表性有限。',
        '## 资源与复现',
        f"已完成的在线训练共 {stats:,} 环境步。GPU 占用区间取并集约 {t:.3f} 小时，含初始化及验证。首个仿真至最后一次记录约 {led['elapsed_first_to_last_simulator_seconds']/3600:.3f} 小时。未结束任务的步数下界和各进程计时来源见账本。[预算](budget_ledger.json) · [软件与资产](environment_provenance.json)",
        '若名义姿态在 2° 边界附近反复失稳，可对已保存的成功和失败轨迹比较首次到位后的角误差与真实控制响应。若名义持续合格而轻扰动欠佳，按真实布局与 socket/grasp 估计偏差分层检查首次进入、末段保持与实际抓取变换。',
        '在仓库根目录查看实时进度或复现已测检查点',
        '\n'.join(['```bash','o="$PWD/artifacts/arch_minexp_v2/run_20261118_alignment"','proto="$PWD/experiments/arch_min_insert/configs/protocol_v2_alignment.yaml"','python3 experiments/arch_min_insert/status_v2.py "$o" --follow','# 查看原始后台日志','tail -F "$o/training.log"','```'])]
    if c:
        q=chosen or c[-1]
        lines.append('\n'.join(['```bash','bash experiments/arch_min_insert/launch.sh --mode eval --group B0 --seed 101 --num_envs 20 --headless \\',
            '  --protocol "$proto" --manifest "$o/manifests/validation_nominal.jsonl" \\',
            f'  --checkpoint "$o/{q["checkpoint"]}" --output "$o/replay_{q["frame"]}_$(date +%s)"','```']))
    names=['g2_nominal',*[p.name for p in out.glob('g2_continue_to_*') if (p/'completion.json').is_file()]]
    estimate=[read(out/name/'completion.json') for name in names]
    speed=sum(x['actual_env_transitions'] for x in estimate)/sum(x['wall_time_s'] for x in estimate)
    lines.append(f"本机 128 环境名义训练吞吐约 {speed:.0f} 环境步/s，续训一个 204,800 步区间的纯训练约 {204800/speed/60:.1f} 分钟，另计启动、独立评估。")
    lines.append('以下顺序复现第二配置已完成的名义续训段。新输出目录会从头执行 G0/G1。G3、G4 的入口分别验证上游门槛。')
    cmds=['```bash','new="$PWD/artifacts/arch_minexp_v2/alignment_reproduction"','py=.venv-arch/bin/python','gate=experiments/arch_min_insert/gate_v2.py','$py -u "$gate" --protocol "$proto" --output "$new"']
    for name in sorted(names[1:],key=lambda x:int(x.split('_')[-1])):
        extra=(int(name.split('_')[-1])-1433600)//204800
        cmds.append(f'$py -u "$gate" --protocol "$proto" --output "$new" --extend-nominal-checks {extra}')
    cmds+=['$py -u experiments/arch_min_insert/curriculum_v2.py --protocol "$proto" --output "$new"','$py -u experiments/arch_min_insert/formal_v2.py --protocol "$proto" --output "$new"','```']
    lines.append('\n'.join(cmds))
    if g4.get('cost_estimate'):lines.append('从头六个正式模型的实测外推约 %.2f GPU 小时。'%(g4['cost_estimate']['seconds_for_six_models']/3600))
    else:lines.append('正式固定课程 B 与轻扰动吞吐尚未同时获得。六个模型仅重训到第二配置最近完成的名义步数，纯训练估计至少 %.2f GPU 小时，轻度课程与测试另计。'%(6*sum(x['actual_env_transitions'] for x in estimate)/speed/3600))
    # Join adjacent table lines without blank lines so Markdown renders a single table.
    body=''
    for line in lines:
        body+=line+('\n' if line.startswith('|') else '\n\n')
    (parent/'report.md').write_text(body)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    report(p.parse_args().output.resolve())
