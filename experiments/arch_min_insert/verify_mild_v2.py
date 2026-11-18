"""Compare the physical grasp at approach with the distinct requested true grasps."""
import json
import numpy as np
import torch
from geometry import compose,inverse,log
from trajectory_diagnosis import diagnose

def verify(directory):
    replay=diagnose(directory)
    with np.load(directory/'trajectories.npz') as t:
        expected=torch.tensor(t['diagnostic_X_true'][0],dtype=torch.float64)
        # First simulated control interval, before board contact.
        observed=torch.tensor(t['diagnostic_X_measured'][0],dtype=torch.float64)
        ids=t['case_id'][0]
        force=np.linalg.norm(t['diagnostic_board_contact_force_world_N'][0],axis=-1)
    manifest={r['case_id']:r for r in map(json.loads,(directory/'realized_manifest.jsonl').read_text().splitlines())}
    nominal=torch.tensor([manifest[k]['X_hat'] for k in ids],dtype=torch.float64)
    requested=compose(inverse(nominal),expected)
    measured=compose(inverse(nominal),observed)
    def stats(requested,measured,unit):
        x=requested-requested.mean(0)
        y=measured-measured.mean(0)
        return {'slope':float((x*y).sum()/x.square().sum()),
                'median_error':float((requested-measured).norm(dim=1).median()*unit),
                'rms_requested':float(requested.square().mean().sqrt()*unit)}
    translation=stats(requested[:,:3],measured[:,:3],1000)
    rotation=stats(log(requested[:,3:]),log(measured[:,3:]),180/np.pi)
    result=dict(offline_replay=replay,first_step_translation_mm=translation,first_step_rotation_deg=rotation,
                first_step_max_board_force_N=float(force.max()),
                all_initial_IK_valid=all(manifest[k]['validity']['status']=='valid' for k in ids))
    (directory/'grasp_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    # Scaling below half or above 1.5 signifies a material loss/distortion of the specified perturbation.
    if not all(.5<s['slope']<1.5 for s in (translation,rotation)):
        raise RuntimeError('Measured grasp failed to reproduce applied perturbations. See '+str(directory/'grasp_verification.json'))
    return result
