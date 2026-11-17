"""Native INSERT reference and controller/contact diagnostic; no training changes."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import types
import numpy as np
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['reference', 'contact', 'paired'], default='reference')
p.add_argument('--num_envs', type=int, default=20)
p.add_argument('--seed', type=int, default=20000)
p.add_argument('--checkpoint', type=Path)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(a).app
import torch
import gymnasium as gym
import omni.isaac.lab_tasks
from omni.isaac.lab_tasks.utils import parse_env_cfg, load_cfg_from_registry
import multi_arm_assembly.direct_isaac_lab_position as native
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.utils import math as m

def serial(x):
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return str(x)

def dump(name, x):
    (a.output/name).write_text(json.dumps(x, default=serial, indent=2)+'\n')

env = None
try:
    cfg = parse_env_cfg('AssemblyInsert-v0', use_gpu=True, num_envs=a.num_envs)
    cfg.parts[0].spawn.activate_contact_sensors = True
    original_setup = native.MyEnv._setup_scene
    def setup(self):
        original_setup(self)
        self.scene.sensors['peg_contact'] = ContactSensor(ContactSensorCfg(
            prim_path='/World/envs/env_.*/'+self.cfg.initial_cfg.PEG_ATTACHMENT_PRIM,
            update_period=0.0, history_length=1))
    native.MyEnv._setup_scene = setup
    env = gym.make('AssemblyInsert-v0', cfg=cfg)
    e = env.unwrapped
    env.seed(a.seed)
    torch.manual_seed(a.seed)
    obs, _ = env.reset()
    if a.mode == 'paired':
        assert a.num_envs == 2
        # Reset-only placement: free at native height, blocked 0.3 mm above
        # the CAD board top, 8 mm laterally offset from the nominal hole.
        rp = e.scene['peg'].data.default_root_state.clone()
        rp[:, :3] = torch.tensor([[.0695,.2695,.115],[.0775,.2695,.09966008]],device=e.device)
        rp[:, 3:7] = torch.tensor([0.,0.,0.,1.],device=e.device)
        goal_root = native.IPose(rp[:,:3],rp[:,3:7])
        goal_ee = goal_root.multiply(native.IPose.from_pose(cfg.initial_cfg.PEG_T_TOOL).repeat(2))
        q = e.robots['robot1'].data.joint_pos.clone()
        for i in range(2):
            sol = native.solve_ik(cfg.robot1.init_state.pos, native.wxyz_to_xyzw(cfg.robot1.init_state.rot),
                                 goal_ee.pos[i].cpu(),native.wxyz_to_xyzw(goal_ee.quat[i].cpu()))
            q[i,e._joint_ids['robot1']] = torch.tensor(sol,device=e.device)
        e.robots['robot1'].write_joint_state_to_sim(q,torch.zeros_like(q))
        e.robots['robot1'].set_joint_position_target(q)
        rp[:,:3] += e.scene.env_origins
        rp[:,7:]=0
        e.scene['peg'].write_root_state_to_sim(rp)
        e.scene.write_data_to_sim()
        for _ in range(20):
            e.sim.step(render=False);e.scene.update(e.physics_dt)
        e.episode_length_buf[:]=0
        obs=e._get_observations()
    contact = e.scene.sensors['peg_contact']
    c = e._ik_controllers['robot1']
    peg_goal = native.IPose.from_pose(cfg.initial_cfg.WORLD_T_HOLE_START).multiply(native.IPose.from_pose(cfg.initial_cfg.PEG_GOAL)).repeat(a.num_envs)
    robot = e.robots['robot1']
    start = {'config': vars(cfg.initial_cfg), 'physics_dt': e.physics_dt, 'step_dt': e.step_dt,
             'max_episode_length': e.max_episode_length, 'num_actions': e.num_actions,
             'observation_shapes': {k:list(v.shape) for k,v in obs.items()},
             'robot_joint_pos': robot.data.joint_pos.clone(), 'robot_body_names': robot.body_names,
             'peg_root': e.scene['peg'].data.root_state_w.clone(),
             'ee_pose': torch.cat(e._compute_frame_pose('robot1',robot), dim=1),
             'goal': peg_goal.to_vec(), 'env_origins':e.scene.env_origins.clone(),
             'contact_paths': contact.body_physx_view.prim_paths,
             'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None}
    dump('initial_state.json', start)
    player = None
    if a.checkpoint:
        from rl_games.algos_torch import model_builder
        from multi_arm_assembly.rl_components.isaac_rlgames_wrapper import MyRlGamesVecEnvWrapper
        from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder
        from multi_arm_assembly.rl_components.my_a2c_model import MyA2CContinuousLogStd
        from multi_arm_assembly.rl_components.my_player import MyPpoPlayerContinuous
        model_builder.register_network('my_network', MyNetworkBuilder)
        model_builder.register_model('my_actor_model', lambda network, **kw: MyA2CContinuousLogStd(network))
        params = load_cfg_from_registry('AssemblyInsert-v0','rl_games_cfg_entry_point')['params']
        params['config']['env_info'] = MyRlGamesVecEnvWrapper(env,e.device,5.,1.).get_env_info()
        player = MyPpoPlayerContinuous(params)
        player.restore(str(a.checkpoint))
        player.has_batch_dimension, player.batch_size = True,a.num_envs
        player.init_rnn()
    records=[]
    original_done=e._get_dones
    def snapshot(self):
        pos,quat=self._compute_frame_pose('robot1',robot)
        peg=self.scene['peg'].data.root_state_w.clone()
        peg[:,:3]-=self.scene.env_origins
        command=torch.cat([c.ee_pos_des,c.ee_quat_des],-1).clone()
        ep=command[:,:3]-pos
        er=m.axis_angle_from_quat(m.quat_mul(command[:,3:],m.quat_conjugate(quat)))
        force=contact.data.net_forces_w[:, 0].clone()
        records.append({'measured_ee':torch.cat([pos,quat],-1),'commanded_ee':command,
          'tracking_residual':torch.cat([ep,er],-1),'diagnostic_peg':peg,
          'diagnostic_net_contact_force_world_N':force,'joint_pos':robot.data.joint_pos.clone(),
          'joint_vel':robot.data.joint_vel.clone(), 'joint_target':robot.data.joint_pos_target.clone()})
        return original_done()
    e._get_dones=types.MethodType(snapshot,e)
    actions_hist=[];obs_hist=[];reward_hist=[];done_hist=[]
    start_wall=time.monotonic()
    with torch.inference_mode():
        for step in range(e.max_episode_length-1):
            actions=torch.zeros((a.num_envs,e.num_actions),device=e.device)
            if player:
                actions=player.get_action(player.obs_to_torch(obs),is_deterministic=True).clamp(-1,1)
            elif a.mode in ['contact','paired']:
                actions[:,2]=-0.04  # same 2 mm world -Z command each control interval
            obs_hist.append(obs['policy'].clone())
            actions_hist.append(actions.clone())
            obs,reward,term,trunc,_=env.step(actions)
            reward_hist.append(reward.clone());done_hist.append((term|trunc).clone())
            if player and player.is_rnn:
                for s in player.states:s[:,term|trunc,:]=0
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
    arrays={k:torch.stack([r[k] for r in records]).cpu().numpy() for k in records[0]}
    arrays.update(actor_observation=torch.stack(obs_hist).cpu().numpy(),action=torch.stack(actions_hist).cpu().numpy(),
      reward=torch.stack(reward_hist).cpu().numpy(),done=torch.stack(done_hist).cpu().numpy(),
      time_s=(np.arange(len(records))+1)*e.step_dt,command_time_s=np.arange(len(records))*e.step_dt)
    np.savez_compressed(a.output/'trajectories.npz',**arrays)
    goal=peg_goal.to_vec().cpu().numpy()
    delta=arrays['diagnostic_peg'][:,:,:3]-goal[None,:,:3]
    rot=torch.stack([m.axis_angle_from_quat(m.quat_mul(peg_goal.quat,m.quat_conjugate(r['diagnostic_peg'][:,3:7]))) for r in records]).norm(dim=-1).cpu().numpy()
    # Provisional diagnostics, NOT a validated physical success detector.
    candidate=(np.linalg.norm(delta[:,:,:2],axis=-1)<.001)&(np.abs(delta[:,:,2])<.001)&(rot<np.deg2rad(2))
    held=np.zeros(a.num_envs,dtype=int);success=np.zeros(a.num_envs,dtype=bool)
    for row in candidate:
        held=np.where(row,held+1,0);success|=held>=5
    manifest=[];episodes=[]
    for i in range(a.num_envs):
        case=f'R0-native-{i:03d}'
        manifest.append({'case_id':case,'split':'reference','suite':'native','seed':a.seed,'env_index':i,
          'robot_joint_pos':start['robot_joint_pos'][i], 'peg_root_world':start['peg_root'][i],
          'hole_pose_config_xyzw':cfg.initial_cfg.WORLD_T_HOLE_START,'env_origin':start['env_origins'][i],
          'validity':'native reset; physical seating detector not validated'})
        episodes.append({'group':'R0','training_seed':42 if player else None,'checkpoint_id':start['checkpoint_sha256'],
          'case_id':case,'suite':'native','num_steps':len(records),'success':None,
          'provisional_pose_success':bool(success[i]),'termination_reason':'timeout' if done_hist[-1][i] else 'bounded_probe',
          'final_depth_offset_m':float(-delta[-1,i,2]),'final_lateral_error_m':float(np.linalg.norm(delta[-1,i,:2])),
          'final_rotation_error_rad':float(rot[-1,i]),'peak_contact_measure':float(np.linalg.norm(arrays['diagnostic_net_contact_force_world_N'][:,i],axis=-1).max()),
          'contact_measure_type':'3D net contact force on peg, world frame, N; moment unavailable',
          'tracking_residual_position_rms_m':float(np.sqrt(np.mean(np.sum(arrays['tracking_residual'][:,i,:3]**2,axis=-1)))),
          'wall_time_s':time.monotonic()-start_wall})
    for name,rows in [('manifest.jsonl',manifest),('episodes.jsonl',episodes)]:
        (a.output/name).write_text(''.join(json.dumps(r,default=serial)+'\n' for r in rows))
    dump('metrics.json',{'mode':a.mode,'num_episodes':a.num_envs,'transitions':len(records)*a.num_envs,
      'wall_time_s':time.monotonic()-start_wall,'provisional_pose_successes':int(success.sum()),
      'physical_successes':None,'finite':True})
    print('MINEXP_PROBE_COMPLETE',a.output,flush=True)
finally:
    if env:env.close()
    app.close()
