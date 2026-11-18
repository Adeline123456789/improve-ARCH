"""Continue a validated nominal B0 through 1/4, 1/2 and full mild perturbations."""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
import torch
import yaml
from protocol import load_protocol,evidence_binding
from budget_v2 import budget
from verify_mild_v2 import verify as verify_mild
from scenarios import make_case
from geometry import compose,inverse,assembly_errors

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--diagnostic-limit',type=int,help='Explicitly authorized cumulative G1-G3 training limit')
p.add_argument('--plan',action='store_true',help='Print the continuation and resource estimate without changing files or launching simulation')
a=p.parse_args()
out=a.output.resolve()
protocol=a.protocol.resolve()
cfg=load_protocol(protocol)
def read(path):
    return json.loads(path.read_text())
def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(path)
report=read(out/'gate_report.json')
if report['G2']['status']!='PASS':
    raise SystemExit('G3 requires a passed G2 nominal screen and independent reload.')
assert read(out/'evidence_binding.json')==evidence_binding(protocol)
chosen=report['G2']['curve'][-1]
frame=chosen['frame']
assert read(out/f'g2_reloaded_{frame}/completion.json')['successes']>=19
checkpoint=out/chosen['checkpoint']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==chosen['sha256']
diagnostic_limit=cfg['budget']['diagnostic'] if a.diagnostic_limit is None else a.diagnostic_limit
if diagnostic_limit>cfg['budget']['total']:
    raise ValueError('Diagnostic allocation exceeds the total training budget')
seed=cfg['evaluation']['diagnostic_seed']
n=read(out/'g2_nominal/identity.json')['num_envs']
rollout=n*cfg['ppo']['horizon_length']
interval=round(cfg['evaluation']['interval']/rollout)*rollout
used=report.get('budget',{}).get('prior_training_steps',0)
if (out/'repair.json').exists():used=read(out/'repair.json')['first_config_interactions']
for name in ['g1','g2_nominal',*[p.name for p in out.glob('g2_continue_to_*')]]:
    used+=read(out/name/'completion.json')['actual_env_transitions']
prior=out.parent/'run_20261118'
budget_roots=[prior,out] if (out/'repair.json').exists() and prior!=out else [out]
def remaining_wall():
    return cfg['budget']['wall_seconds']-budget(budget_roots)['gpu_wall_seconds_union_estimate']
existing=report['G3'].get('curve',[])
if a.plan:
    last=existing[-1] if existing else None
    current_used=last['diagnostic_total'] if last else used
    completed=[read(p/'completion.json') for p in out.glob('g3_train_b*_f*') if (p/'completion.json').exists()]
    speed=sum(r['actual_env_transitions'] for r in completed)/sum(r['wall_time_s'] for r in completed) if completed else None
    blocks=max(0,(diagnostic_limit-current_used)//interval)
    print(json.dumps(dict(checkpoint=last['checkpoint'] if last else str(checkpoint.relative_to(out)),
        restored_frame=last['frame'] if last else frame,stage_scale=last['scale'] if last else .25,
        diagnostic_used=current_used,proposed_diagnostic_limit=diagnostic_limit,
        interval=interval,available_new_blocks=blocks,new_steps=blocks*interval,
        estimated_training_seconds=blocks*interval/speed if speed else None,
        remaining_gpu_seconds=remaining_wall(),formal_each_limit=cfg['budget']['formal_each'],
        action='plan only; no simulation started'),indent=2))
    raise SystemExit(0)
if report['G3']['status']=='PASS':
    raise SystemExit('G3 already passed. Continue with formal_v2.py.')
if a.diagnostic_limit is not None:
    history=out/'diagnostics'/f'before_g3_limit_{diagnostic_limit}.json'
    if not history.exists():write(history,report)
report['budget']['diagnostic_limit']=diagnostic_limit
report['overall_status']='IN_PROGRESS'
g3=report['G3']={'status':'RUNNING','curve':[],'starting_frame':frame}
write(out/'gate_report.json',report)
def execute(mode,name,*args):
    directory=out/name
    if not (directory/'completion.json').exists():
        cmd=['bash',str(HERE/'launch.sh'),'--mode',mode,'--protocol',str(active_protocol),'--group','B0',
             '--seed',str(seed),'--output',str(directory),'--headless',*map(str,args)]
        print(shlex.join(cmd),flush=True)
        remaining=remaining_wall()
        if remaining<=0:raise TimeoutError('shared wall budget exhausted')
        subprocess.run(cmd,check=True,timeout=remaining)
    ident=read(directory/'identity.json')
    assert ident['mode']==mode and ident['protocol_sha256']==hashlib.sha256(active_protocol.read_bytes()).hexdigest()
    for flag,key in [('--checkpoint','checkpoint_sha256'),('--manifest','manifest_sha256')]:
        assert ident[key]==hashlib.sha256(Path(args[args.index(flag)+1]).read_bytes()).hexdigest()
    return read(directory/'completion.json')

def manifest(path,source):
    text=''.join(json.dumps(row)+'\n' for row in source)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    return path
def eval_model(name,manifest,count):
    r=execute('eval',name,'--checkpoint',checkpoint,'--manifest',manifest,'--num_envs',count)
    assert r['num_complete_episodes']==count
    actual=[json.loads(line)['case_id'] for line in (out/name/'episodes.jsonl').read_text().splitlines()]
    assert actual==[json.loads(line)['case_id'] for line in manifest.read_text().splitlines()]
    return r['successes']

nominal=out/'manifests/validation_nominal.jsonl'
block=0
for rung,scale in enumerate([.25,.5,1.]):
    scene=dict(cfg['scene'])
    for key in ['layout_xy_m','yaw_deg','socket_xy_m','grasp_xyz_m','bias_deg']:scene[key]*=scale
    scene['train_suite']='E1'
    active_protocol=out/f'g3_rung_{rung}.yaml'
    settings=dict(cfg,scene=scene,curriculum={'mild_fraction':scale,'nominal_train_fraction':.25,
                                             'train_manifest_pattern':'manifests/g3_train_b{block}.jsonl',
                                             'validation_manifest':f'manifests/g3_validation_r{rung}.jsonl'})
    active_protocol.write_text(yaml.safe_dump(settings))
    val=manifest(out/f'manifests/g3_validation_r{rung}.jsonl',
                 (make_case(570000+rung*1000+i,'E1','validation',scene) for i in range(100)))
    val_rows=[json.loads(line) for line in val.read_text().splitlines()]
    distinct={json.dumps([r['board_true'],r['X_true'],r['socket_estimated']]) for r in val_rows}
    assert len(distinct)==len(val_rows)==100
    poses=lambda key:torch.tensor([r[key] for r in val_rows],dtype=torch.float64)
    star=poses('T_SP_star')
    ee_at_estimate=compose(compose(poses('socket_estimated'),star),inverse(poses('X_hat')))
    actual_tip=compose(ee_at_estimate,poses('X_true'))
    ident=star.new_tensor([0,0,0,1,0,0,0]).expand_as(star)
    geometric=assembly_errors(actual_tip,ident,poses('socket_true'),star,**cfg['geometry'])
    write(out/f'manifests/g3_validation_r{rung}_audit.json',dict(scene=scene,realized_unique=len(distinct),
        sha256=hashlib.sha256(val.read_bytes()).hexdigest(),
        ideal_estimated_goal_outside_true_geometry=int((~geometric['candidate']).sum()),
        median_ideal_goal_lateral_mm=float(geometric['lateral'].median()*1000),
        p90_ideal_goal_lateral_mm=float(geometric['lateral'].quantile(.9)*1000)))
    consecutive=0
    while consecutive < (2 if scale==1. else 1):
        if used+interval>diagnostic_limit or remaining_wall()<=0:
            g3.update(status='FAIL',reason='shared diagnostic budget reached before consecutive checks')
            report['overall_status']='BUDGET_BLOCKED'
            write(out/'gate_report.json',report)
            raise SystemExit(0)
        block+=1
        count=(interval//n//cfg['episode_steps']+1)*n
        # Maintain nominal regression while introducing distinct mild training scenes.
        cases=(make_case(50000000+block*100000+i,'E0' if i%4==0 else 'E1','train',scene) for i in range(count))
        train=manifest(out/f'manifests/g3_train_b{block}.jsonl',cases)
        frame+=interval
        name=f'g3_train_b{block}_f{frame}'
        r=execute('train',name,'--num_envs',n,'--checkpoint',checkpoint,'--manifest',train,'--transitions',frame)
        assert r['actual_env_transitions']==interval
        used+=interval
        checkpoint=sorted((out/name/'runner/run/nn').glob(f'last_*_ep_{frame//rollout}_rew_*.pth'))[0]
        state=torch.load(checkpoint,map_location='cpu')
        assert state['frame']==frame and 'optimizer' in state
        nom=eval_model(f'g3_nominal_b{block}',nominal,20)
        mild=eval_model(f'g3_mild_b{block}',val,100)
        try:
            verify_mild(out/f'g3_mild_b{block}')
        except (AssertionError,RuntimeError) as exc:
            g3.update(status='FAIL',reason=str(exc),failed_check=f'g3_mild_b{block}')
            report['overall_status']='PIPELINE_BLOCKED'
            write(out/'gate_report.json',report)
            raise
        consecutive=consecutive+1 if nom>=18 and mild>=50 else 0
        g3['curve'].append(dict(frame=frame,new_steps=interval,diagnostic_total=used,scale=scale,
            nominal_successes=nom,mild_successes=mild,nominal_episodes=20,mild_episodes=100,
            checkpoint=str(checkpoint.relative_to(out)),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        write(out/'gate_report.json',report)
g3['status']='PASS'
write(out/'gate_report.json',report)
