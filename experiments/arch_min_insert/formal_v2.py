"""Freeze the measured B0 curriculum, then run paired fresh PPO seeds within the shared budget."""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import torch
from budget_v2 import budget
from protocol import evidence_binding,load_protocol
from scenarios import make_case

HERE=Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text())
def write(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save_cases(path,rows):
    content=''.join(json.dumps(c)+'\n' for c in rows)
    if path.exists():
        assert path.read_text()==content
    else:
        path.write_text(content)
    return path

def main(out,protocol,additional_wall_seconds):
    report=read(out/'gate_report.json')
    if any(report[key]['status']!='PASS' for key in ('G0','G1','G2','G3')):
        raise SystemExit('Formal comparison requires G0 through G3 PASS.')
    cfg=load_protocol(protocol)
    binding=evidence_binding(protocol)
    assert binding==read(out/'evidence_binding.json')
    manifests=out/'manifests'
    history=[out.parent/'run_20261118',out] if (out/'repair.json').exists() else [out]
    limit=cfg['budget']['wall_seconds']+additional_wall_seconds
    def ledger():
        return budget(history)
    curve=report['G3']['curve']
    frames=report['G3']['starting_frame']
    # Preserve the nominal training restart boundaries as well as the perturbation schedule.
    ends=[read(out/'g2_nominal/completion.json')['actual_env_transitions']]
    ends+=sorted(int(d.name.rsplit('_',1)[1]) for d in out.glob('g2_continue_to_*') if (d/'completion.json').exists())
    ends=sorted({v for v in ends if v<frames}|{frames})
    stages=[(v,0.) for v in ends]+[(r['frame'],r['scale']) for r in curve]
    total=stages[-1][0]
    assert total<=cfg['budget']['formal_each']
    n=read(out/'g2_nominal/identity.json')['num_envs']
    rollout=n*cfg['ppo']['horizon_length']
    assert all(v%rollout==0 for v,_ in stages)
    freeze=dict(binding=binding,runner_sha256=sha(Path(__file__)),num_envs=n,seeds=[0,1],groups=['B0','B1','B2'],
                equal_training_steps=total,stages=stages,scene=cfg['scene'],
                stage_protocols={str(s):sha(protocol if s==0 else out/f'g3_rung_{[.25,.5,1.].index(s)}.yaml') for _,s in stages},
                nominal_fraction_on_mild_stages=.25,checkpoint_rule='final fixed training step only, same for every group',
                validation_frequency='at the fixed final checkpoint; nominal 20 and full mild 100',
                holdout={'N':[20,'nominal repeats'],'T_id':[100,'full mild'],'T_layout':[100,'50 unseen X offsets 20.001 to 40 mm, 50 unseen yaw 15.0001 to 30 degrees, E1 estimation error distribution'],'T_bias':[100,'E1 layout with doubled socket and grasp estimation ranges']},
                holdout_base_seed=81000000,validation_base_seed=82000000,training_base_seed=90000000,
                reachability='Same preinsertion geometry as G0/G3. Each reset checks full target start IK, joint bounds and CAD peg clearance before actions. Whole-robot collision enumeration unavailable.')
    freeze_path=out/'frozen_formal_protocol.json'
    if freeze_path.exists():assert read(freeze_path)==json.loads(json.dumps(freeze))
    else:write(freeze_path,freeze)
    splits={}
    def manifest(suite,split,start,count,scene,path):
        return save_cases(manifests/path,(make_case(start+i,suite,split,scene) for i in range(count)))
    for mode,base,suites in [('validation',82000000,['E0','E1']),('test',81000000,['E0','E1','E2','E3'])]:
        for i,suite in enumerate(suites):
            splits[mode,suite]=manifest(suite,mode,base+i*1000,20 if suite=='E0' else 100,cfg['scene'],f'formal_{mode}_{suite}.jsonl')
    def physical(row):
        return tuple(tuple(row[k]) for k in ('board_true','socket_estimated','X_true','X_hat'))
    unseen=set()
    for (split,suite),path in splits.items():
        cases=[json.loads(line) for line in path.read_text().splitlines()]
        keys={physical(c) for c in cases}
        if suite!='E0':
            assert len(keys)==len(cases) and not keys.intersection(unseen)
            unseen.update(keys)
    for path in manifests.glob('g3_*.jsonl'):
        with path.open() as f:
            assert all(physical(json.loads(l)) not in unseen for l in f)
    audit={p.name:sha(p) for p in splits.values()}
    write(out/'formal_manifest_audit.json',{'sha256':audit,'unique_non_nominal':len(unseen),'nominal_physical_cases':1})
    formal=report['G4']
    formal.update(status='RUNNING',protocol_sha256=sha(freeze_path))
    formal.setdefault('runs',[])
    def save():
        write(out/'gate_report.json',report)
    save()
    recorded=ledger()
    sample_names=['g2_nominal',*[p.name for p in out.glob('g3_train_b*_f*')]]
    costs=[j for j in recorded['jobs'] if Path(j['directory']).name in sample_names and Path(j['directory']).parent==out]
    throughput=sum(j['new_training_steps'] for j in costs)/sum(j['simulator_wall_seconds'] for j in costs)
    eval_costs=[j['simulator_wall_seconds'] for j in recorded['jobs'] if Path(j['directory']).name.startswith(('g3_nominal_b','g3_mild_b'))]
    per_model=total/throughput+6*sum(eval_costs)/len(eval_costs)
    formal['cost_estimate']={'seconds_per_model_including_six_evaluations':per_model,'training_steps_per_simulator_second':throughput,
                             'seconds_for_six_models':6*per_model,'wall_limit_seconds':limit}
    def run(mode,name,group,seed,active,manifest_path,count,checkpoint=None,target=None):
        directory=out/name
        args=['bash',str(HERE/'launch.sh'),'--mode',mode,'--group',group,'--seed',str(seed),'--protocol',str(active),
              '--output',str(directory),'--manifest',str(manifest_path),'--num_envs',str(count),'--headless']
        if checkpoint:args+=['--checkpoint',str(checkpoint)]
        if target:args+=['--transitions',str(target)]
        if not (directory/'completion.json').exists():
            left=limit-ledger()['gpu_wall_seconds_union_estimate']
            if left<=0:raise TimeoutError('GPU budget exhausted')
            print(shlex.join(args),flush=True)
            subprocess.run(args,check=True,timeout=left)
        identity=read(directory/'identity.json')
        assert (identity['group'],identity['seed'],identity['num_envs'],identity['mode'])==(group,seed,count,mode)
        assert identity['protocol_sha256']==sha(active) and identity['manifest_sha256']==sha(manifest_path)
        assert identity['checkpoint_sha256']==(sha(checkpoint) if checkpoint else None)
        return read(directory/'completion.json')
    for seed in freeze['seeds']:
        for group in freeze['groups']:
            if any(r['group']==group and r['seed']==seed for r in formal['runs']):continue
            recorded=ledger()
            stem=f'g4_s{seed}_{group}'
            new_so_far=sum(j['new_training_steps'] or 0 for j in recorded['jobs'] if Path(j['directory']).name.startswith(stem+'_train'))
            remaining=total-new_so_far
            if remaining/throughput+(per_model-total/throughput)>limit-recorded['gpu_wall_seconds_union_estimate'] or remaining+recorded['completed_training_steps']>cfg['budget']['total']:
                formal.update(status='NOT_RUN' if not formal['runs'] else 'PARTIAL',reason='insufficient shared budget for next complete fixed-budget model')
                report['overall_status']='SCREENING_ONLY' if len(formal['runs'])>=3 else 'BUDGET_BLOCKED'
                save()
                return
            checkpoint=None
            previous=0
            for i,(frame,scale) in enumerate(stages):
                active=protocol if not scale else out/f'g3_rung_{[.25,.5,1.].index(scale)}.yaml'
                scene=load_protocol(active)['scene']
                count=(1+(frame-previous)//n//cfg['episode_steps'])*n if scale else n
                base=90000000+seed*10000000+i*100000
                path=manifests/f'formal_train_s{seed}_stage{i}.jsonl'
                if not path.exists():
                    save_cases(path,(make_case(base+j,'E0' if not scale or j%4==0 else 'E1','train',scene) for j in range(count)))
                with path.open() as f:
                    assert all(physical(json.loads(l)) not in unseen for l in f)
                name=f'{stem}_train_f{frame}'
                comp=run('train',name,group,seed,active,path,n,checkpoint,frame)
                assert comp['actual_env_transitions']==frame-previous
                checkpoint=sorted((out/name/'runner/run/nn').glob(f'last_*_ep_{frame//rollout}_rew_*.pth'))[0]
                assert torch.load(checkpoint,map_location='cpu')['frame']==frame
                previous=frame
            row=dict(group=group,seed=seed,frame=total,checkpoint=str(checkpoint.relative_to(out)),sha256=sha(checkpoint),counts={})
            for (split,suite),path in splits.items():
                count=20 if suite=='E0' else 100
                name=f'{stem}_{split}_{suite}'
                comp=run('eval',name,group,seed,active,path,count,checkpoint)
                assert comp['num_complete_episodes']==count
                rows=list(map(json.loads,(out/name/'episodes.jsonl').read_text().splitlines()))
                assert [r['case_id'] for r in rows]==[json.loads(l)['case_id'] for l in path.read_text().splitlines()]
                row['counts'][split+'_'+suite]=dict(successes=comp['successes'],episodes=count,valid=sum(r['valid'] for r in rows))
            formal['runs'].append(row)
            save()
    formal['status']='PASS'
    report['overall_status']='VALID_COMPARISON_COMPLETED'
    save()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--additional-wall-seconds',type=int,default=0,help='Only for a separately authorized future budget extension')
    args=parser.parse_args()
    main(args.output.resolve(),args.protocol.resolve(),args.additional_wall_seconds)
