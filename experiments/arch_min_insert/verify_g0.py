"""Reconstruct the terminal labels from CAD transforms and the recorded physics trajectory."""
import json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from geometry import assembly_errors,compose,delta,reward_terms

def verify(run):
    d=run/'g0_scripted'
    p=yaml.safe_load((d/'effective_config.yaml').read_text())
    rows=[json.loads(x) for x in (d/'episodes.jsonl').read_text().splitlines()]
    cases=[json.loads(x) for x in (d/'realized_manifest.jsonl').read_text().splitlines()][:20]
    t=np.load(d/'trajectories.npz')
    root,sock=[torch.tensor(t[k],dtype=torch.float64) for k in ('diagnostic_peg_root','diagnostic_socket_true')]
    rt=torch.tensor([0,0,-.05,0,1,0,0],dtype=root.dtype).expand_as(root)
    target=torch.tensor([c['T_SP_star'] for c in cases],dtype=root.dtype).expand_as(root)
    m=assembly_errors(root,rt,sock,target,**p['geometry'])
    moved=delta(torch.tensor([.8,-.6,1.2,.4,.2,.1],dtype=root.dtype).expand(*root.shape[:-1],6))
    mm=assembly_errors(compose(moved,root),rt,compose(moved,sock),target,**p['geometry'])
    for a,b in zip(reward_terms(m,p['reward']),reward_terms(mm,p['reward'])):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    expected=torch.tensor([c['socket_true'] for c in cases],dtype=root.dtype)
    torch.testing.assert_close(expected[:,:3],sock[0,:,:3],atol=2e-6,rtol=0)
    assert (m['depth'][0]<0).all()
    cfg=json.loads((d/'runtime_config_check.json').read_text())
    success=m['candidate'][-cfg['hold_steps']:].all(0)&~m['invalid'].any(0)
    assert success.tolist()==[r['success'] for r in rows] and len(rows)==20 and success.sum()>=19
    joints=json.loads((d/'fixed_joints.json').read_text())
    assert len(joints)==20
    for i in range(20):
        prefix=f'/World/envs/env_{i}/'
        assert sum(j['body0']==[prefix+'robot1/tool0'] and j['body1'][0].startswith(prefix+'peg/') for j in joints)==1
    probe=np.load(run/'g0_probe/trajectories.npz')
    forces=np.linalg.norm(probe['diagnostic_board_contact_force_world_N'],axis=-1)
    fixed=probe['commanded_ee'][:,2,:3]
    assert np.ptp(fixed,axis=0).max()<2e-6 and forces[:,1].max()>.5
    return {'status':'PASS','scripted_successes':int(success.sum()),'episodes':len(rows),'offline_matches':True,
            'edge_peak_force_N':float(forces[:,1].max()),'fixed_target_range_m':float(np.ptp(fixed,axis=0).max()),
            'hold_tracking_error_final_m':float(np.linalg.norm(probe['tracking_residual_world'][-1,2,:3])),
            'probe_trace':'g0_probe/trajectories.npz','scripted_trace':'g0_scripted/trajectories.npz'}

if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1])),indent=2))
