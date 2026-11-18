"""Run the nominal learning screen after validating physical and PPO evidence."""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
import torch
from verify_g0 import verify
from protocol import load_protocol

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--output',required=True,type=Path)
p.add_argument('--protocol',type=Path,default=HERE/'configs/protocol_v2.yaml')
a=p.parse_args()
out=a.output.resolve()
protocol=a.protocol.resolve()
cfg=load_protocol(protocol)
def load(path):
    return json.loads((out/path).read_text())
def save(path,data):
    path=out/path
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(path)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
out.mkdir(parents=True,exist_ok=True)
seed=cfg['evaluation']['diagnostic_seed']
# Count elapsed time conservatively, including initialization and gaps between commands.
clock=out/'budget_start.json'
if not clock.exists():
    legacy=out/'legacy_regression.log'
    save('budget_start.json',{'unix':legacy.stat().st_ctime-30 if legacy.exists() else time.time()})
start=load('budget_start.json')['unix']
def run(mode,name,*args,run_seed=None):
    cmd=['bash',str(HERE/'launch.sh'),'--protocol',str(protocol),'--group','B0','--seed',str(seed if run_seed is None else run_seed),
         '--mode',mode,'--headless','--output',str(out/name),*map(str,args)]
    if (out/name/'completion.json').is_file():
        ident=load(name+'/identity.json')
        for field in ('group','seed','mode','num_envs'):
            assert str(ident[field])==cmd[cmd.index('--'+field)+1]
        assert ident['protocol_sha256']==sha(protocol)
        for flag in ['--manifest','--checkpoint','--scripted','--ppo_audit','--response_probe','--transitions']:
            old=ident['command']
            assert (flag in cmd)==(flag in old)
            if flag in cmd and flag not in ('--scripted','--ppo_audit','--response_probe'):
                assert cmd[cmd.index(flag)+1]==old[old.index(flag)+1]
        return
    print(time.strftime('%F %T'),shlex.join(cmd),flush=True)
    remaining=cfg['budget']['wall_seconds']-(time.time()-start)
    if remaining<=0:
        raise RuntimeError('wall budget exhausted')
    subprocess.run(cmd,check=True,timeout=remaining)

from scenarios import make_case
manifests=out/'manifests'
manifests.mkdir(exist_ok=True)
for split,base in [('train',seed),('validation',700000)]:
    rows=[make_case(base+i,'E0',split,cfg['scene']) for i in range(cfg['evaluation']['nominal_count'])]
    path=manifests/(split+'_nominal.jsonl')
    content=''.join(json.dumps(row)+'\n' for row in rows)
    if path.exists():
        assert path.read_text()==content
    else:
        path.write_text(content)
run('fixtures','g0_probe','--response_probe','--steps',96,'--num_envs',4,run_seed=0)
run('eval','g0_scripted','--scripted','--num_envs',20,'--manifest',manifests/'validation_nominal.jsonl',run_seed=0)
g0=verify(out)
g1_frames=2*16*cfg['ppo']['horizon_length']
run('train','g1','--ppo_audit','--num_envs',16,'--transitions',g1_frames,'--manifest',manifests/'train_nominal.jsonl')
audit_ck,=(out/'g1/runner/run/nn').glob('last*.pth')
run('eval','g1_reload','--num_envs',20,'--checkpoint',audit_ck,'--manifest',manifests/'validation_nominal.jsonl')

for name in ['g0_scripted','g0_probe','g1','g1_reload']:
    ident=load(name+'/identity.json')
    assert ident['protocol_sha256']==sha(protocol)
    for file in ['min_env.py','scenarios.py','geometry.py']:
        assert ident['source_sha256'][file]==sha(HERE/file)
updates=[json.loads(x) for x in (out/'g1/ppo_updates.jsonl').read_text().splitlines()]
reload=load('g1_reload/checkpoint_checks.json')
assert len(updates)>=2 and all(max(u['actor_update_norm'].values())>0 and u['neglogp_max_error']<.001 for u in updates)
assert reload['command_max_error']<1e-5 and reload['action_max_error']<1e-5
assert load('g1_reload/completion.json')['num_complete_episodes']==20
seed=cfg['evaluation']['diagnostic_seed']
n=128
rollout=n*cfg['ppo']['horizon_length']
interval=max(1,round(cfg['evaluation']['interval']/rollout))*rollout
frames=cfg['budget']['nominal_first']//interval*interval
steps=list(range(interval,frames+1,interval))
report={'G0':g0,'G1':{'status':'PASS','updates':'g1/ppo_updates.jsonl','reload':reload},
        'G2':{'status':'RUNNING','seed':seed,'planned_checkpoints':steps,'curve':[]},
        'G3':{'status':'NOT_RUN'},'G4':{'status':'NOT_RUN'}}
save('gate_report.json',report)
run('train','g2_nominal','--num_envs',n,'--transitions',frames,'--manifest',out/'manifests/train_nominal.jsonl')
assert load('g2_nominal/completion.json')['actual_env_transitions']==frames
previous=False
for frame in steps:
    checkpoints=sorted((out/'g2_nominal/runner/run/nn').glob(f'last_*_ep_{frame//rollout}_rew_*.pth'))
    ck=checkpoints[0]
    assert torch.load(ck,map_location='cpu')['frame']==frame
    def evaluate(name):
        run('eval',name,'--num_envs',20,'--checkpoint',ck,'--manifest',out/'manifests/validation_nominal.jsonl')
        result=load(name+'/completion.json')
        assert result['num_complete_episodes']==20
        return result['successes']
    name=f'g2_eval_{frame}'
    successes=evaluate(name)
    report['G2']['curve'].append({'frame':frame,'successes':successes,'episodes':20,'checkpoint':str(ck.relative_to(out)),'sha256':sha(ck),'evaluation':name})
    save('gate_report.json',report)
    if successes>=19 and previous and evaluate(f'g2_reloaded_{frame}')>=19:
        report['G2']['status']='PASS'
        break
    previous=successes>=19
else:
    report['G2']['status']='FAIL'
save('gate_report.json',report)
