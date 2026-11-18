"""Replay a fixed-case evaluation and measure where progress stopped."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from geometry import assembly_errors

def diagnose(directory):
    config=yaml.safe_load((directory/'effective_config.yaml').read_text())
    runtime=json.loads((directory/'runtime_config_check.json').read_text())
    episodes=[json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
    cases={r['case_id']:r for r in map(json.loads,(directory/'realized_manifest.jsonl').read_text().splitlines())}
    with np.load(directory/'trajectories.npz') as trace:
        # Parallel environments reset synchronously at the protocol cutoff.
        length=config['episode_steps']
        cut=lambda key:torch.as_tensor(np.concatenate(np.split(trace[key],trace[key].shape[0]//length),axis=1))
        peg=cut('diagnostic_peg_root').double()
        socket=cut('diagnostic_socket_true').double()
        ids=np.concatenate(trace['case_id'][::length])
        assert list(ids)==[r['case_id'] for r in episodes]
        tip=peg.new_tensor([0,0,-.05,0,1,0,0]).expand_as(peg)
        goal=peg.new_tensor([cases[i]['T_SP_star'] for i in ids]).expand_as(peg)
        met=assembly_errors(peg,tip,socket,goal,**config['geometry'])
        hold=torch.zeros_like(met['depth'][0],dtype=torch.long)
        recorded_hold=cut('hold')
        for k,candidate in enumerate(met['candidate']):
            hold=torch.where(candidate,hold+1,0)
            assert torch.equal(hold,recorded_hold[k])
        success=(hold>=runtime['hold_steps'])&~met['invalid'].any(0)
        assert success.tolist()==[r['success'] for r in episodes]
        entering=met['depth']>=0
        covered=entering.any(0)
        entry_lateral=met['lateral'].gather(0,entering.long().argmax(0)[None])[0,covered]
        dist=met['position_error'].norm(dim=-1)
        med=lambda x:float(x.quantile(.5)) if x.numel() else None
        return dict(episodes=len(episodes),successes=int(success.sum()),reached_opening_plane=int(covered.sum()),
                    median_lateral_at_first_opening_plane_mm=med(entry_lateral*1000),
                    median_peak_depth_mm=med(met['depth'].max(0).values*1000),
                    median_closest_tip_position_error_mm=med(dist.min(0).values*1000),
                    median_final_tip_position_error_mm=med(dist[-1]*1000),
                    median_final_lateral_mm=med(met['lateral'][-1]*1000),
                    median_final_depth_error_mm=med(met['depth_error'][-1]*1000),
                    median_final_tilt_and_yaw_symmetry_error_deg=med(met['rotation'][-1].rad2deg()),
                    ever_met_geometry=int(met['candidate'].any(0).sum()),
                    invalid_physics=int(met['invalid'].any(0).sum()),
                    median_peak_board_contact_N=med(cut('diagnostic_board_contact_force_world_N').norm(dim=-1).max(0).values),
                    mean_reward_terms={k:float(cut(k).mean()) for k in ['position_reward','rotation_reward','terminal_reward']})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    directory=parser.parse_args().directory
    result=diagnose(directory)
    (directory/'trajectory_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
