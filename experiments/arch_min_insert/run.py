"""Actual rl-games/MyA2CAgent training and deterministic fixed-case evaluation."""
import argparse,json,hashlib,time,sys,math,yaml
from pathlib import Path
from omni.isaac.lab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--protocol',type=Path,default=Path(__file__).with_name('configs')/'protocol_v2.yaml')
p.add_argument('--ppo_audit',action='store_true')
p.add_argument('--response_probe',action='store_true')
p.add_argument('--mode',choices=['smoke','train','eval','fixtures','manifest'],required=True)
p.add_argument('--group',choices=['B0','B1','B2'],default='B0')
p.add_argument('--seed',type=int,default=0);p.add_argument('--num_envs',type=int)
p.add_argument('--transitions',type=int,default=1996800);p.add_argument('--steps',type=int,default=192)
p.add_argument('--contact_pair',action='store_true');p.add_argument('--scripted',action='store_true');p.add_argument('--with_camera',action='store_true');p.add_argument('--layout_fixture',action='store_true')
p.add_argument('--checkpoint',type=Path);p.add_argument('--manifest',type=Path)
p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
protocol=yaml.safe_load(a.protocol.read_text())
a.num_envs=a.num_envs if a.num_envs is not None else protocol['num_envs']
protocol['num_envs']=a.num_envs
assert protocol['control']['mode']=='measured_delta'
assert not (a.scripted and a.checkpoint)
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'effective_config.yaml').write_text(yaml.safe_dump(protocol))
app=AppLauncher(a).app
import torch,numpy as np
import gymnasium as gym
import omni.isaac.lab_tasks
import multi_arm_assembly.direct_isaac_lab_position as native
from omni.isaac.lab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from min_env import MinInsertEnv,configure_board
from scenarios import make_case,IK
from geometry import compose,inverse,true_metrics
from rl_games.algos_torch import model_builder
from rl_games.common import env_configurations,vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv
from multi_arm_assembly.rl_components.isaac_rlgames_wrapper import MyRlGamesVecEnvWrapper
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder
from multi_arm_assembly.rl_components.my_a2c_model import MyA2CContinuousLogStd
from multi_arm_assembly.rl_components.my_agent import MyA2CAgent
from multi_arm_assembly.rl_components.my_player import MyPpoPlayerContinuous

def save(name,obj):
    (a.output/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
def tolist(t):return t.cpu().tolist()
env=None
try:
    cfg=parse_env_cfg('AssemblyInsert-v0',use_gpu=True,num_envs=a.num_envs)
    if a.with_camera:
        from dataclasses import replace
        cfg=cfg.replace(initial_cfg=replace(cfg.initial_cfg,USE_CAMERA=True))
    cfg.initial_cfg.USE_FT_SENSOR=False;cfg.num_observations=27;cfg.num_actions=6
    cfg.parts[0].spawn.activate_contact_sensors=True
    configure_board(cfg)
    cfg.sim.dt=protocol['physics_dt']
    cfg.decimation=protocol['decimation']
    cfg.episode_length_s=protocol['episode_steps']*(cfg.sim.dt*cfg.decimation)
    cfg.is_finite_horizon=True
    for field in ('stiffness','damping','effort_limit'):
        setattr(cfg.robot1.actuators['arm'],field,protocol['control'][field])
    # Native rendering assets/controller gains and physics dt remain unchanged.
    cases=None
    if a.manifest:
        cases=[json.loads(l) for l in a.manifest.read_text().splitlines()]
    if a.mode=='train' and cases is not None:
        assert cases and all(c['split']=='train' for c in cases), 'expected split=train'
    if a.mode=='manifest':
        ik=IK('ur_description/urdf/ur10e.urdf',cfg.robot1.init_state.pos,cfg.robot1.init_state.rot,cfg.joint_angles_list['robot1'])
        for split,base,count in [('validation',10000,50),('test',20000,100)]:
            rows=[]
            for si,suite in enumerate(['E0','E1','E2','E3']):
                for i in range(count):
                    c=make_case(base+si*1000+i,suite,split,protocol['scene'])
                    q,fk,check=ik.solve(c['ee_start']);c.update(robot_arm_joint_pos=q,ee_start_fk=fk,validity=check)
                    assert check['status']=='valid',c
                    rows.append(c)
            path=a.output/(split+'_manifest.jsonl');path.write_text(''.join(json.dumps(c)+'\n' for c in rows))
        save('summary.json',{'validation_cases':200,'test_cases':400,'rejected':0,'status':'IK valid; physical contact fixtures pending'})
    else:
        if a.mode=='fixtures':
            assert a.num_envs==4
            cases=[]
            for i,kind in enumerate(['seated','edge','penetrating','free']):
                c=make_case(20000+i,'E2' if a.layout_fixture else 'E0','fixture')
                if a.layout_fixture:
                    c['socket_estimated']=c['socket_true'];c['X_true']=c['X_hat'];c['socket_bias']=[0]*6;c['grasp_bias']=[0]*6
                star=torch.tensor(c['T_SP_star']);pre=star.clone()
                if kind=='edge':pre[0]=.008;pre[2]=-.0003
                if kind=='penetrating':pre[2]+=.02
                if kind=='free':pre[2]-=.06
                if a.contact_pair or a.response_probe:
                    kind='edge' if i%2 else 'free'
                    pre=star.clone();pre[2]=-.0003 if i%2 else star[2]-.06
                    if i%2:pre[0]=.008
                c['ee_start']=compose(compose(torch.tensor(c['socket_estimated']),pre),inverse(torch.tensor(c['X_hat']))).tolist()
                c['case_id']='fixture-'+kind+'-'+str(i);cases.append(c)
        env=MinInsertEnv(cfg,protocol,a.mode,group=a.group,seed=a.seed,output=a.output,cases=cases)
        from pxr import UsdPhysics
        stage=native.stage_utils.get_current_stage()
        save('fixed_joints.json',[{'path':str(prim.GetPath()),'body0':[str(x) for x in UsdPhysics.FixedJoint(prim).GetBody0Rel().GetTargets()], 'body1':[str(x) for x in UsdPhysics.FixedJoint(prim).GetBody1Rel().GetTargets()]} for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint) and 'Assembler' in str(prim.GetPath())])
        env.report_group='scripted-'+a.group if a.scripted else a.group
        env.policy_kind='scripted feedback' if a.scripted else 'trained PPO' if a.checkpoint else 'PPO training' if a.mode=='train' else 'zero-action diagnostic'
        env.checkpoint_id=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None
        torch.manual_seed(a.seed);env.seed(a.seed)
        model_builder.register_network('my_network',MyNetworkBuilder)
        model_builder.register_model('my_actor_model',lambda network,**kw:MyA2CContinuousLogStd(network))
        params=load_cfg_from_registry('AssemblyInsert-v0','rl_games_cfg_entry_point')
        params['params']['seed']=a.seed
        net=params['params']['network']
        net['rnn'].update(units=protocol['network']['rnn_units'],layers=protocol['network']['layers'])
        net['mlp']['units']=protocol['network']['mlp']
        net['space']['continuous']['sigma_init']['val']=protocol['network']['initial_log_sigma']
        pc=params['params']['config']
        pc.update(protocol['ppo'])
        rollout=a.num_envs*pc['horizon_length']
        pc.update(num_actors=a.num_envs,minibatch_size=rollout//4,
            train_dir=str(a.output/'runner'),full_experiment_name='run',max_epochs=a.transitions//(rollout),
            save_frequency=max(1,round(protocol['evaluation']['interval']/rollout)),zero_rnn_on_done=True)
        save('agent_config.json',params)
        identity={'group':a.group,'seed':a.seed,'mode':a.mode,'command':sys.argv,'started_unix':time.time(),
          'num_envs':a.num_envs,'physics_dt':env.physics_dt,'policy_dt':env.step_dt,'episode_steps':env.max_episode_length,
          'checkpoint':str(a.checkpoint) if a.checkpoint else None,
          'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,
          'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
          'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['run.py','min_env.py','geometry.py','scenarios.py']},
          'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest() if a.manifest else None}
        save('identity.json',identity)
        save('physics_snapshot.json',{'peg_mass_kg':tolist(env.scene['peg'].root_physx_view.get_masses()[0]),
            'peg_materials_static_dynamic_friction_restitution':tolist(env.scene['peg'].root_physx_view.get_material_properties()[0]),
            'board_materials_static_dynamic_friction_restitution':tolist(env.scene['hole'].root_physx_view.get_material_properties()[0]),
            'robot_joint_stiffness':tolist(env.robots['robot1'].data.joint_stiffness[0]),
            'robot_joint_damping':tolist(env.robots['robot1'].data.joint_damping[0]),
            'board_type':'kinematic rigid body with original CAD SDF collision','physics_dt':env.physics_dt})
        robot=env.robots['robot1']
        joint_ids=env._joint_ids['robot1']
        runtime={'physics_dt':env.physics_dt,'step_dt':env.step_dt,'episode_steps':env.max_episode_length,'num_envs':env.num_envs,'hold_steps':env.hold_steps,'env_origins':tolist(env.scene.env_origins)}
        view=robot.root_physx_view
        for key,values in [('stiffness',view.get_dof_stiffnesses()),('damping',view.get_dof_dampings()),('effort_limit',view.get_dof_max_forces())]:
            arm=values[:,joint_ids]
            runtime[key]=tolist(arm[0])
            assert torch.allclose(arm,torch.full_like(arm,protocol['control'][key])),(key,arm)
        save('runtime_config_check.json',runtime)
        assert env.physics_dt==cfg.sim.dt and env.step_dt==cfg.sim.dt*cfg.decimation
        assert env.num_envs==a.num_envs and env.max_episode_length==protocol['episode_steps']
        # Diagnostic RGB is not an actor modality.
        env.cnn=False
        wrapped=MyRlGamesVecEnvWrapper(env,env.device,protocol['clip_obs'],1.)
        if a.mode=='train':
            assert a.transitions%(rollout)==0,'Use equal complete PPO rollout budgets'
            vecenv.register('IsaacRlgWrapper',lambda config_name,num_actors,**kw:RlGamesGpuEnv(config_name,num_actors,**kw))
            env_configurations.register('rlgpu',{'vecenv_type':'IsaacRlgWrapper','env_creator':lambda **kw:wrapped})
            runner=Runner(IsaacAlgoObserver())
            if a.ppo_audit:
                from ppo_audit import AuditAgent as MyA2CAgent
            runner.algo_factory.register_builder('my_agent',lambda **kw:MyA2CAgent(**kw))
            runner.player_factory.register_builder('my_agent',lambda **kw:MyPpoPlayerContinuous(**kw))
            runner.load(params);runner.reset()
            start=time.monotonic();runner.run({'train':True,'play':False,'sigma':None,'checkpoint':str(a.checkpoint) if a.checkpoint else None})
            save('completion.json',{'status':'completed','actual_env_transitions':env.total_transitions,'wall_time_s':time.monotonic()-start})
        else:
            obs,_=env.reset();player=None
            checks={}
            if a.mode=='smoke':
                # Truth and diagnostic contact do not enter actor construction.
                before=env._get_observations()['policy'].clone()
                st=env.socket_true.clone();xt=env.xtrue.clone()
                env.socket_true[:,:3]+=.2;env.xtrue[:,:3]+=.1
                assert torch.equal(before,env._get_observations()['policy'])
                env.socket_true[:]=st;env.xtrue[:]=xt
                checks['hidden_truth_does_not_change_actor_observations']=True
                # Reset one env without simulation advance or other env history edits.
                if a.num_envs>1:
                    other=env.socket_estimated[1:].clone();old=env.socket_estimated[0].clone()
                    simstep=env._sim_step_counter
                    env._reset_idx(torch.tensor([0],device=env.device))
                    assert env._sim_step_counter==simstep
                    assert torch.equal(other,env.socket_estimated[1:])
                    assert not torch.equal(old,env.socket_estimated[0])
                    assert not env.valid[0] and torch.equal(env.xhat[0],env.xhat[1])
                    checks['asynchronous_reset_isolated_and_bias_resampled']=True
                obs=env._get_observations()
                save('implementation_checks.json',checks)
            if a.checkpoint:
                params['params']['config']['env_info']=wrapped.get_env_info()
                player=MyPpoPlayerContinuous(params['params']);player.restore(str(a.checkpoint))
                player.has_batch_dimension=True;player.batch_size=a.num_envs;player.init_rnn()
                testobs=player.obs_to_torch({'policy':obs['policy']})
                norm_before={k:v.clone() for k,v in player.model.state_dict().items() if 'running_mean_std' in k}
                # Exercise the rollout actor and action rescaling with the player's restored tensors.
                from geometry import tip_command
                train_actor=object.__new__(MyA2CAgent)
                train_actor.model=player.model
                train_actor.rnn_states=tuple(t.clone() for t in player.states)
                train_actor.has_central_value=False
                train_actor.clip_actions=player.clip_actions
                train_actor.actions_low,train_actor.actions_high=player.actions_low,player.actions_high
                train_actor.is_tensor_obses=True
                rollout_obs=wrapped._process_obs(dict(obs))
                sampled=train_actor.get_action_values({'obs':rollout_obs})
                train_action=train_actor.preprocess_actions(sampled['mus'])
                act1=player.get_action(testobs,is_deterministic=True)
                compare=lambda action:tip_command(env.ee(),env.xhat,env.socket_estimated,action,a.group!='B0',protocol['control']['translation_limit'],protocol['control']['rotation_limit'])[0]
                command_error=float((compare(train_action)-compare(act1)).abs().max())
                action_error=float((train_action-act1).abs().max())
                assert command_error<1e-5 and action_error<1e-5
                player.restore(str(a.checkpoint));player.init_rnn()
                act2=player.get_action(testobs,is_deterministic=True)
                assert torch.equal(act1,act2)
                assert all(torch.equal(v,player.model.state_dict()[k]) for k,v in norm_before.items())
                player.init_rnn()
                save('checkpoint_checks.json',{'command_max_error':command_error,'action_max_error':action_error,'reload_mean_actions_identical':True,'normalizers_frozen':True,'policy_dimension':27,'action_dimension':6})
            rows=[];steps=a.steps if a.mode!='eval' else env.max_episode_length*((len(cases)+a.num_envs-1)//a.num_envs)
            start=time.monotonic();obs0=obs['policy'].clone();x0=env.xhat.clone();s0=env.socket_estimated.clone()
            camera_frames=[]
            with torch.inference_mode():
                for step in range(steps):
                    actions=torch.zeros(a.num_envs,6,device=env.device)
                    if a.scripted:
                        from geometry import log,qmul,qinv,rotate
                        tip=compose(env.ee(),env.xhat);goal=compose(env.socket_estimated,env.star)
                        dp=(goal[:,:3]-tip[:,:3])*20/protocol['control']['translation_limit']
                        dr=log(qmul(goal[:,3:],qinv(tip[:,3:])))*10/protocol['control']['rotation_limit']
                        if a.group!='B0':
                            dp=rotate(qinv(env.socket_estimated[:,3:]),dp);dr=rotate(qinv(env.socket_estimated[:,3:]),dr)
                        actions=torch.cat((dp,dr),-1)
                    if a.contact_pair:
                        actions[:]=0;actions[:,2]=(-.002 if a.group=='B0' else .002)/protocol['control']['translation_limit']
                    if a.response_probe:
                        from geometry import log,qmul,qinv
                        ee=env.ee()
                        if step==0:
                            hold_target=compose(ee,env.xhat).clone()
                            hold_target[2,0]+=.002
                        sign=1 if (step//6)%2==0 else -1
                        actions[:,2]=-.002/protocol['control']['translation_limit']
                        actions[0,:]=0
                        actions[0,(step//12)%6]=sign*(.002/protocol['control']['translation_limit'] if step<36 else .01/protocol['control']['rotation_limit'])
                        tip=compose(ee,env.xhat)
                        actions[2,:3]=(hold_target[2,:3]-tip[2,:3])/protocol['control']['translation_limit']
                        actions[2,3:]=log(qmul(hold_target[2,3:],qinv(tip[2,3:])))/protocol['control']['rotation_limit']
                    if player:actions=player.get_action(player.obs_to_torch({'policy':obs['policy']}),is_deterministic=True)
                    case_ids=[c['case_id'] for c in env.case_info]
                    obs,reward,term,trunc,_=env.step(actions)
                    if player and player.is_rnn:
                        for state in player.states:state[:,term|trunc,:]=0
                    assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                    # Save all smoke/fixture trajectories; fixed small selection for evaluation.
                    if a.with_camera:
                        rgb=env.scene.sensors['camera'].data.output['rgb'][0,:,:,:3]
                        if rgb.dtype.is_floating_point and rgb.max()<=1:rgb=rgb*255
                        camera_frames.append(rgb.clamp(0,255).to(torch.uint8).cpu().numpy())
                    keep=range(a.num_envs if a.mode=='eval' else min(a.num_envs,8))
                    rows.append({k:v[list(keep)].cpu().numpy() for k,v in env.last_step.items()})
                    rows[-1]['case_id']=np.array([case_ids[i] for i in keep])
            if camera_frames:
                import cv2
                h,w=camera_frames[0].shape[:2]
                writer=cv2.VideoWriter(str(a.output/'fixed_case.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),24,(w,h))
                assert writer.isOpened()
                for rgb in camera_frames:writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                writer.release();cv2.imwrite(str(a.output/'fixed_case.png'),cv2.cvtColor(camera_frames[-1],cv2.COLOR_RGB2BGR))
            np.savez_compressed(a.output/'trajectories.npz',**{k:np.stack([row[k] for row in rows]) for k in rows[0]})
            if a.mode=='smoke':
                # Every stored command is from the immediately preceding interval.
                assert np.allclose(np.stack([r['time_s']-r['command_time_s'] for r in rows]),env.step_dt,atol=1e-6)
                checks['residual_timestamp_alignment']=True
                checks['reset_residual_mask']=bool((obs['policy'][:,21:]==0).all()) if not env.valid.any() else 'checked at reset'
                save('implementation_checks.json',checks)
            save('completion.json',{'status':'completed','transitions':env.total_transitions,'num_complete_episodes':len(env.completed),
              'successes':sum(r['success'] for r in env.completed),'wall_time_s':time.monotonic()-start,'observation_dim':obs['policy'].shape[-1],
              'actor_force_fields':[],'controller_uses_ft':False,'finite':True})
            if a.mode=='fixtures':
                save('fixture_result.json',{'cases':cases,'episodes':env.completed,'expected':[True,False,False,False],
                   'warning':'physical validation requires inspecting traces for stable seating / collision response'})
    print('MINEXP_COMPLETE',a.mode,a.group,a.output,flush=True)
except Exception:
    import traceback
    (a.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env:env.close()
    app.close()
