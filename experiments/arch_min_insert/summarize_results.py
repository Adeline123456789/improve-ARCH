"""Generate tables/figures from completed runs only. No inferred successes."""
import csv,json,math,hashlib,statistics
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/arch_minexp';PLOTS=OUT/'plots';PLOTS.mkdir(exist_ok=True)

def write_csv(path,rows,fields=None):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def load_rows(path):
    if not path.exists():return []
    content=path.read_text();lines=content.splitlines()
    if content and not content.endswith('\n'):lines=lines[:-1]  # append in progress
    return [json.loads(x) for x in lines]
def summarize():
    selection=json.loads((OUT/'final_selection.json').read_text()) if (OUT/'final_selection.json').exists() else []
    episodes=[];table=[];failures=[];by_policy={};manifest={x['case_id']:x for x in load_rows(OUT/'evaluation_manifest.jsonl')}
    traces_dir=OUT/'traces';traces_dir.mkdir(exist_ok=True)
    for item in selection:
        directory=Path(item['test_dir']);rows=load_rows(directory/'episodes.jsonl')
        if not (directory/'completion.json').exists():continue
        # Fixed manifest is the denominator. Duplicates/padding must never count.
        ids=[r['case_id'] for r in rows];assert len(ids)==len(set(ids))==400,(directory,len(ids))
        assert set(ids)==set(manifest)
        by_policy[(item['group'],item['seed'])]={r['case_id']:r for r in rows}
        episodes.extend(rows)
        for suite in ['E0','E1','E2','E3']:
            rs=[r for r in rows if r['suite']==suite];n=len(rs);k=sum(r['success'] for r in rs)
            table.append({'group':item['group'],'training_seed':item['seed'],'training_transitions':item['transitions'],
              'suite':suite,'successes':k,'episodes':n,'success_rate':k/n,'checkpoint_sha256':item['checkpoint_sha256'],
              'success_completion_time_s':statistics.mean(r['completion_time_s'] for r in rs if r['success']) if k else None})
            for reason,count in Counter(r['termination_reason'] for r in rs).items():
                failures.append({'group':item['group'],'training_seed':item['seed'],'suite':suite,'reason':reason,'count':count})
        # All evaluations are retained locally. Publish fixed + first five per outcome/suite.
        chosen=[]
        for suite in ['E0','E1','E2','E3']:
            sr=[r for r in rows if r['suite']==suite]
            chosen.append(sr[0]['case_id'])
            for success in [True,False]:chosen.extend(r['case_id'] for r in [x for x in sr if x['success']==success][:5])
        chosen=list(dict.fromkeys(chosen))
        raw=np.load(directory/'trajectories.npz');sample={}
        for key in raw.files:
            if key=='case_id':continue
            per=[]
            for case in chosen:
                mask=raw['case_id']==case;arr=raw[key][mask];assert len(arr)==96,(case,key,arr.shape);per.append(arr)
            sample[key]=np.stack(per)
        sample['case_id']=np.array(chosen)
        np.savez_compressed(traces_dir/f'{item["group"]}_seed{item["seed"]}_representative.npz',**sample)
    (OUT/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in episodes))
    write_csv(OUT/'results.csv',table);write_csv(OUT/'failure_counts.csv',failures)
    stratified=[]
    for (group,seed),data in by_policy.items():
        for label,predicate in [('E2_translation',lambda c:c['suite']=='E2' and c['stratum']=='translation'),('E2_yaw',lambda c:c['suite']=='E2' and c['stratum']=='yaw'),('E3_outside_training',lambda c:c['suite']=='E3' and c['error_outside_training'])]:
            rr=[row for case,row in data.items() if predicate(manifest[case])]
            stratified.append({'group':group,'seed':seed,'stratum':label,'successes':sum(r['success'] for r in rr),'episodes':len(rr)})
    write_csv(OUT/'stratified_results.csv',stratified)
    paired=[]
    for label,left,right,suites in [('H1','B1','B0',['E2']),('H2','B2','B1',['E3']),('overall','B2','B0',['E1','E2','E3']),('E0_change','B2','B0',['E0'])]:
        seeds=sorted({s for g,s in by_policy if g==left}&{s for g,s in by_policy if g==right})
        for seed in seeds:
            x=by_policy[(left,seed)];y=by_policy[(right,seed)];assert set(x)==set(y)
            diffs=[int(x[k]['success'])-int(y[k]['success']) for k in x if x[k]['suite'] in suites]
            paired.append({'contrast':label,'seed':seed,'delta_percentage_points':100*statistics.mean(diffs),'paired_cases':len(diffs)})
    write_csv(OUT/'paired_deltas.csv',paired)
    # Training data are separate from evaluation. Plot actual produced transitions.
    curves=[]
    for directory in sorted(OUT.iterdir()):
        if not directory.is_dir() or not any(directory.name.startswith(x) for x in ['screen_B','confirm_B','extended_B']):continue
        rs=load_rows(directory/'episodes.jsonl');bins=defaultdict(list)
        for r in rs:bins[(r['transitions']-1)//51200].append(r)
        for bucket,rr in bins.items():curves.append({'run':directory.name,'group':rr[0]['group'],'seed':rr[0]['training_seed'],
             'transitions':max(r['transitions'] for r in rr),'mean_return':statistics.mean(r['return'] for r in rr),
             'train_success_rate':statistics.mean(int(r['success']) for r in rr),'episodes':len(rr)})
    write_csv(OUT/'learning_curves.csv',curves)
    colors={'B0':'#536878','B1':'#247ba0','B2':'#d1495b'}
    if curves:
        fig,axes=plt.subplots(1,2,figsize=(10,3.4))
        for run in sorted({r['run'] for r in curves}):
            rr=[r for r in curves if r['run']==run];g=rr[0]['group'];x=np.array([r['transitions'] for r in rr])/1e6
            axes[0].plot(x,[r['mean_return'] for r in rr],color=colors[g],label=run,alpha=.8)
            axes[1].plot(x,[100*r['train_success_rate'] for r in rr],color=colors[g],alpha=.8)
        for ax in axes:ax.set_xlabel('Training transitions (million)');ax.grid(alpha=.2)
        axes[0].set_ylabel('Mean episode return');axes[1].set_ylabel('Training success (%)');axes[0].legend(fontsize=7)
        fig.tight_layout();fig.savefig(PLOTS/'learning_curves.png',dpi=180);fig.savefig(PLOTS/'learning_curves.pdf');plt.close(fig)
    if table:
        fig,ax=plt.subplots(figsize=(7.5,4));xs=np.arange(4)
        for gi,g in enumerate(['B0','B1','B2']):
            means=[];std=[]
            for suite in ['E0','E1','E2','E3']:
                values=[100*r['success_rate'] for r in table if r['group']==g and r['suite']==suite]
                means.append(statistics.mean(values) if values else np.nan);std.append(statistics.stdev(values) if len(values)>1 else 0.)
            ax.bar(xs+(gi-1)*.25,means,width=.24,yerr=std,color=colors[g],label=g,capsize=3)
        ax.set_xticks(xs,['E0','E1','E2','E3']);ax.set_ylabel('Success (%)');ax.set_ylim(0,105);ax.legend();ax.grid(axis='y',alpha=.2)
        ax.set_title('Mean across training seeds; error bars = sample SD')
        fig.tight_layout();fig.savefig(PLOTS/'success_rates.png',dpi=180);fig.savefig(PLOTS/'success_rates.pdf');plt.close(fig)
    # Contact signal diagnostic: two free/blocked pairs, not independent training seeds.
    path=OUT/'contact_common_v3/trajectories.npz'
    if path.exists():
        z=np.load(path);fig,axes=plt.subplots(1,3,figsize=(11,3.3));t=z['time_s'][:,0]
        for name,indices,color in [('free',[0,2],'#247ba0'),('blocked',[1,3],'#d1495b')]:
            axes[0].plot(t,1000*np.linalg.norm(z['tracking_residual_world'][:,indices,:3],axis=-1).mean(1),label=name,color=color)
            axes[1].plot(t,np.linalg.norm(z['diagnostic_board_contact_force_world_N'][:,indices],axis=-1).mean(1),color=color)
            zz=z['diagnostic_peg_root'][:,indices,2];axes[2].plot(t,1000*(zz[0]-zz).mean(1),color=color)
        for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
        axes[0].set_ylabel('Position tracking residual (mm)');axes[1].set_ylabel('Peg-board net contact force (N)');axes[2].set_ylabel('Downward displacement (mm)');axes[0].legend()
        fig.tight_layout();fig.savefig(PLOTS/'contact_probe.png',dpi=180);fig.savefig(PLOTS/'contact_probe.pdf');plt.close(fig)
    # Evidence-focused Chinese report. Missing evaluations are explicitly unrun.
    training_records=[]
    for job in ['screen','confirm','extended']:
        path=OUT/(job+'_job.json')
        if path.exists():training_records.extend(json.loads(path.read_text()).get('runs',[]))
    (OUT/'training_inventory.json').write_text(json.dumps(training_records,indent=2)+'\n')
    rows=['| 模型 | 训练种子数 | 每种子 transitions | E0 SR | E1 SR | E2 SR | E3 SR | E1/E2/E3 均值 | 主要失败 | 执行器 FT |',
          '|---|---:|---|---|---|---|---|---|---|---|']
    for g in ['B0','B1','B2']:
        gt=[r for r in table if r['group']==g];seeds=sorted({r['training_seed'] for r in gt});rates=[]
        for suite in ['E0','E1','E2','E3']:
            rr=[r for r in gt if r['suite']==suite];values=[r['success_rate']*100 for r in rr]
            rates.append(('；'.join(f"s{r['training_seed']}: {r['successes']}/{r['episodes']}" for r in rr)+f"；均值 {statistics.mean(values):.1f}%"+(f" ± {statistics.stdev(values):.1f}" if len(values)>1 else '（无种子方差估计）')) if rr else '未运行')
        pressure=[r['success_rate']*100 for r in gt if r['suite']!='E0']
        reasons=Counter(r['termination_reason'] for r in episodes if r['group']==g and not r['success'])
        count=', '.join(str(x) for x in sorted({r['training_transitions'] for r in gt})) or '未完成'
        rows.append(f"| {g} | {len(seeds)} | {count} | "+' | '.join(rates)+f" | {statistics.mean(pressure):.1f}% | {dict(reasons)} | 否（仅本仿真链） |" if pressure else f"| {g} | {len({r['seed'] for r in training_records if r['group']==g and 'actual_transitions' in r})} | {sorted({r['actual_transitions'] for r in training_records if r['group']==g and 'actual_transitions' in r}) or '训练尚未完成'} | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 否（仅本仿真链） |")
    findings=[]
    for label in ['H1','H2','overall','E0_change']:
        rr=[r for r in paired if r['contrast']==label];v=[r['delta_percentage_points'] for r in rr]
        if v:findings.append(f"- {label}：各种子差值 {v} 个百分点；均值 {statistics.mean(v):.2f}"+(f"，样本标准差 {statistics.stdev(v):.2f}。" if len(v)>1 else '；只有一个训练种子。'))
        else:findings.append(f'- {label}：尚无完整成对测试，不能给出收益结论。')
    overall=[r['delta_percentage_points'] for r in paired if r['contrast']=='overall'];e0=[r['delta_percentage_points'] for r in paired if r['contrast']=='E0_change']
    qualifies=len(overall)==3 and statistics.mean(overall)>=10 and all(v>0 for v in overall) and statistics.mean(e0)>=-5
    basic=any(r['suite']=='E0' and r['successes']>0 for r in table)
    conclusion='达到预先设定的工程筛查门槛，值得进入下一阶段；仍是有限预算证据。' if qualifies else '当前配置下尚无明确收益；未达到完整工程筛查门槛。' if basic else '未完成有效检验：尚未获得学会基础插入的学习策略证据，不能据此宣布算法无效。'
    state=json.loads((OUT/'background_status.json').read_text()) if (OUT/'background_status.json').exists() else {}
    report=f'''# ARCH INSERT 最小实验报告

更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。

**{conclusion}**

'''+ '\n'.join(rows)+ '\n\n'+ '\n'.join(findings)+f'''

原始逐回合记录在 `episodes.jsonl`，逐种子/逐 suite 原始计数在 `results.csv`，成对差值在 `paired_deltas.csv`，布局与超训练边界误差分层在 `stratified_results.csv`。误差条仅为训练种子间样本标准差，不把 episode 数当成训练重复次数。E0 无偏置名义场景是确定性重复，不为它报告独立二项置信区间。

## 管线证据

- R0：既有 seed=42、100,000 transitions checkpoint，20 个原生固定场景共 1,900 次交互。候选位姿条件 0/20；原生没有成功检测器，故不填正式物理 SR。R0 与 B0 身份不同。
- 原生重复抓取约束修复：32 环境曾有 63 个固定关节；新环境每环境只有一个。静态插座仅改 USD 在布局夹具中失败，改为原 CAD/SDF 的运动学刚体并用 PhysX 张量更新后通过。
- 夹具验证：正确就位为成功，孔沿与悬空不成功，穿透为异常物理。不同布局也通过。原生控制器参数保持 400/800；未重新安装 Isaac，未训练高层 DiT。
- 简单闭环控制在无误差名义验证场景 4/4 成功，约 2.625 秒完成；这不是 PPO 成功率。
- 最终接触对照：自由下压约 2.86 mm，孔沿约 0.27 mm；相同 2 mm 命令的位置残差均值约 1.971 与 1.998 mm，差异仅约 27 微米。自由场景板接触力为 0，受阻峰值约 1.33–1.35 N。残差信号较弱且控制滞后占主导。
- 64/256/512 环境检查均通过；512 环境预检共 98,304 次交互。坐标等变、尖端作用点、环境原点、真值不进入 Actor、异步 reset、时间戳检查和 checkpoint 重载/冻结归一化检查有真实记录。

## 范围与限制

B0/B1 比较的是坐标和动作表达的组合；B1/B2 比较显式执行残差特征。三组 Actor 与共享 Critic 都无力输入，本次仿真执行器也无 FT；独立真机代码仍有力反馈路径，本实验没有验证真机去力传感器。局部组保留共同全局插座朝向，不宣称完整 Actor 输入严格 SE(3) 不变。净接触力不等于六维腕部 wrench，力矩 unavailable。

固定物理参数、CAD/阈值、预算及 SHA256 见 `frozen_protocol.yaml`、`frozen_hashes.json`。预算不足时只比较相同 transitions 快照；未达到推荐的每组每种子 500 万步时，不宣称充分收敛或完成完整确认。物理引擎非确定性和浮点环境原点误差仍存在。初态检查包括 IK、关节限位和保守插头间隙，没有穷尽整臂自碰撞检查。固定抓取不能评价真实掉落/滑移。

后台决策与完成状态：`background_status.json`。当前记录的决策：{state.get('decision','尚未完成阶段决策')}。

最值得做的一项后续工作：在冻结验证场景上检查强偏置下的接触残差可分性，并检验更长共同训练预算是否改善学习；只有现有对照出现可重复收益后才扩展传感或动力学模块。
'''
    (OUT/'report.md').write_text(report)
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'checkpoints').glob('*.pth')}
    (OUT/'checkpoint_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return {'completed_policies':len(by_policy),'test_episodes':len(episodes),'engineering_gate_met':qualifies,'conclusion':conclusion}
if __name__=='__main__':print(json.dumps(summarize(),ensure_ascii=False))
