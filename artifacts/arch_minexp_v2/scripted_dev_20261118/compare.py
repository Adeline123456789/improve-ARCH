from pathlib import Path
import collections,csv,json,hashlib,statistics,math,sys
import numpy as np
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'experiments/arch_min_insert'))
from verify_mild_v2 import verify
import yaml
base=ROOT/'artifacts/arch_minexp_v2'
src=base/'run_20261118_alignment'
out=base/'scripted_dev_20261118'
def read(p):return json.loads(p.read_text())
def rows(p):return list(map(json.loads,p.read_text().splitlines()))
summary={}
paired=[]
for rung,blocks in [(0,range(1,6)),(1,range(6,17))]:
    d=out/f'rung_{rung}'
    check=verify(d)
    ident=read(d/'identity.json')
    comp=read(d/'completion.json')
    assert ident['checkpoint'] is None and '--scripted' in ident['command'] and '--checkpoint' not in ident['command']
    episodes=rows(d/'episodes.jsonl')
    cases=rows(src/f'manifests/g3_validation_r{rung}.jsonl')
    ids=[r['case_id'] for r in cases]
    assert len(set(ids))==len(episodes)==comp['num_complete_episodes']==100
    assert [r['case_id'] for r in episodes]==ids
    assert all(r['num_steps']==96 for r in episodes)
    cfg=yaml.safe_load((d/'effective_config.yaml').read_text())
    with np.load(d/'trajectories.npz') as tr:
        obs=tr['actor_observation']
        action=np.concatenate([20*obs[:,:,:3]/cfg['control']['translation_limit'],10*obs[:,:,3:6]/cfg['control']['rotation_limit']],axis=2)
        delta=float(np.abs(tr['action']-action.clip(-1,1)).max())
        assert delta<1.e-4,delta
    record=dict(successes=comp['successes'],valid=sum(r['valid'] for r in episodes),failed_by_reason=dict(collections.Counter(r['termination_reason'] for r in episodes if not r['success'])),
                median_lateral_mm=statistics.median(r['final_lateral_error_m']*1000 for r in episodes),median_abs_depth_error_mm=statistics.median(abs(r['final_depth_error_m'])*1000 for r in episodes),
                median_orientation_deg=statistics.median(math.degrees(r['final_rotation_error_rad']) for r in episodes),
                max_action_error_vs_estimate_feedback=delta,paired_vs_all_B0=[])
    for block in blocks:
        b=src/f'g3_mild_b{block}'
        baseline=rows(b/'episodes.jsonl')
        assert [r['case_id'] for r in baseline]==ids
        bi=read(b/'identity.json')
        assert all(bi[k]==ident[k] for k in ['seed','num_envs','episode_steps','manifest_sha256','source_sha256','protocol_sha256'])
        reference=yaml.safe_load((b/'effective_config.yaml').read_text())
        assert reference['run']['checkpoint'] and cfg['run']['checkpoint'] is None
        reference['run']['checkpoint']=None
        assert reference==cfg
        counts=collections.Counter((a['success'],v['success']) for a,v in zip(episodes,baseline))
        comparison=dict(block=block,B0_successes=sum(r['success'] for r in baseline),both=counts[True,True],scripted_only=counts[True,False],B0_only=counts[False,True],neither=counts[False,False])
        record['paired_vs_all_B0'].append(comparison)
        if block in [5,10,16]:
            for a,b0 in zip(episodes,baseline):
                paired.append(dict(rung=rung,case_id=a['case_id'],B0_block=block,scripted_success=a['success'],B0_success=b0['success'],scripted_valid=a['valid'],B0_valid=b0['valid']))
    summary[rung]=record
(out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'paired_cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=paired[0]);w.writeheader();w.writerows(paired)
for rung,r in summary.items():
 print(rung,{k:v for k,v in r.items() if k!='paired_vs_all_B0'},[x for x in r['paired_vs_all_B0'] if x['block'] in [5,10,16]])
