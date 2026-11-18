"""Resolve the v2 defaults, protocol file and explicit run overrides."""
import hashlib
import math
from pathlib import Path

import yaml

DEFAULT=Path(__file__).resolve().parent/'configs/protocol_v2.yaml'

def load_protocol(path=DEFAULT,*,num_envs=None):
    cfg=yaml.safe_load(DEFAULT.read_text())
    supplied=yaml.safe_load(Path(path).read_text())
    for key,value in supplied.items():
        if isinstance(cfg.get(key),dict):
            cfg[key].update(value)
        else:
            cfg[key]=value
    if num_envs is not None:
        cfg['num_envs']=num_envs
    policy_dt=cfg['physics_dt']*cfg['decimation']
    if 'episode_length_s' in supplied:
        steps=round(supplied['episode_length_s']/policy_dt)
        if not math.isclose(steps*policy_dt,supplied['episode_length_s']):
            raise ValueError('episode_length_s must contain complete control steps')
        if 'episode_steps' in supplied and supplied['episode_steps']!=steps:
            raise ValueError('episode_length_s disagrees with episode_steps')
        cfg['episode_steps']=steps
    if not (policy_dt>0 and cfg['episode_steps']>0 and cfg['num_envs']>0 and
            0<cfg['hold_duration_s']<=cfg['episode_steps']*policy_dt):
        raise ValueError('invalid time scale, episode length or environment count')
    if cfg['control']['mode']!='measured_delta':
        raise ValueError('only measured_delta has been validated')
    if min(cfg['control'][key] for key in ['translation_limit','rotation_limit','effort_limit'])<=0:
        raise ValueError('control scales and arm effort limit must be positive')
    if cfg['ppo']['horizon_length']%cfg['ppo']['seq_length']:
        raise ValueError('rollout horizon must contain whole recurrent sequences')
    return cfg

def evidence_binding(protocol):
    here=DEFAULT.parent.parent
    root=here.parents[1]
    native=root/'multi-robot-assembly/multi_arm_assembly'
    assets=root/'multi-robot-assembly/taskboard/fmb_example'
    files=[here/name for name in ('run.py','protocol.py','min_env.py','geometry.py','scenarios.py','ppo_audit.py')]
    files += [native/'direct_isaac_lab_position.py',native/'assembly_insert_env_cfg.yaml',*sorted((native/'rl_components').glob('*.py'))]
    files += [assets/'Medium_Board_DarkBlue_Updated.usd',assets/'Medium_Short_Hexagon_Green_Updated.usd',DEFAULT]
    return {'protocol_sha256': hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),
            'files': {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
