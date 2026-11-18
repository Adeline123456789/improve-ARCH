"""Continue a validated nominal B0 through 1/4, 1/2 and full mild perturbations."""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
import torch
from protocol import load_protocol
from scenarios import make_case

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
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
chosen=report['G2']['curve'][-1]
frame=chosen['frame']
assert read(out/f'g2_reloaded_{frame}/completion.json')['successes']>=19
checkpoint=out/chosen['checkpoint']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==chosen['sha256']
seed=cfg['evaluation']['diagnostic_seed']
n=read(out/'g2_nominal/identity.json')['num_envs']
rollout=n*cfg['ppo']['horizon_length']
interval=round(cfg['evaluation']['interval']/rollout)*rollout
used=report.get('budget',{}).get('prior_training_steps',0)
if (out/'repair.json').exists():used=read(out/'repair.json')['first_config_interactions']
for name in ['g1','g2_nominal',*[p.name for p in out.glob('g2_continue_to_*')]]:
    used+=read(out/name/'completion.json')['actual_env_transitions']
start=read(out/'budget_start.json')['unix']
g3=report['G3']={'status':'RUNNING','curve':[],'starting_frame':frame}
def execute(mode,name,*args):
    directory=out/name
    if not (directory/'completion.json').exists():
        cmd=['bash',str(HERE/'launch.sh'),'--mode',mode,'--protocol',str(protocol),'--group','B0',
             '--seed',str(seed),'--output',str(directory),'--headless',*map(str,args)]
        print(shlex.join(cmd),flush=True)
        remaining=cfg['budget']['wall_seconds']-(time.time()-start)
        if remaining<=0:raise TimeoutError('shared wall budget exhausted')
        subprocess.run(cmd,check=True,timeout=remaining)
    ident=read(directory/'identity.json')
    assert ident['mode']==mode and ident['protocol_sha256']==hashlib.sha256(protocol.read_bytes()).hexdigest()
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
    return r['successes']

nominal=out/'manifests/validation_nominal.jsonl'
block=0
for rung,scale in enumerate([.25,.5,1.]):
    scene=dict(cfg['scene'])
    for key in ['layout_xy_m','yaw_deg','socket_xy_m','grasp_xyz_m','bias_deg']:scene[key]*=scale
    val=manifest(out/f'manifests/g3_validation_r{rung}.jsonl',
                 (make_case(570000+rung*1000+i,'E1','validation',scene) for i in range(100)))
    distinct={json.dumps([r['board_true'],r['X_true'],r['socket_estimated']]) for r in map(json.loads,val.read_text().splitlines())}
    assert len(distinct)==100
    consecutive=0
    while consecutive < (2 if scale==1. else 1):
        if used+interval>cfg['budget']['diagnostic'] or time.time()-start>=cfg['budget']['wall_seconds']:
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
        consecutive=consecutive+1 if nom>=18 and mild>=50 else 0
        g3['curve'].append(dict(frame=frame,new_steps=interval,diagnostic_total=used,scale=scale,
            nominal_successes=nom,mild_successes=mild,nominal_episodes=20,mild_episodes=100,
            checkpoint=str(checkpoint.relative_to(out)),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        write(out/'gate_report.json',report)
g3['status']='PASS'
write(out/'gate_report.json',report)
