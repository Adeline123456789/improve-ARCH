"""Isolated INSERT research environment; reuses native PhysX assets, IK and actuators.
Never calls native global-stepping reset or puts simulator truth in actor_features.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
import omni.isaac.lab.utils.math as m
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor,ContactSensorCfg
import multi_arm_assembly.direct_isaac_lab_position as native
from geometry import *
from scenarios import make_case,IK
from omni.isaac.lab.assets import RigidObjectCfg
import omni.isaac.lab.sim as sim_utils

def spawn_kinematic_board(prim_path,cfg,translation=None,orientation=None):
    prim=sim_utils.spawn_from_usd(prim_path,cfg,translation=translation,orientation=orientation)
    for body in sim_utils.find_matching_prims(prim_path):
        api=UsdPhysics.RigidBodyAPI.Apply(body)
        api.CreateRigidBodyEnabledAttr(True);api.CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateDisableGravityAttr(True)
    return prim

def configure_board(cfg):
    board=next(x for x in cfg.peripherals if x.prim_path.endswith('/hole'))
    cfg.peripherals=[x for x in cfg.peripherals if x is not board]
    spawn=board.spawn.copy();spawn.func=spawn_kinematic_board
    cfg.parts.append(RigidObjectCfg(prim_path=board.prim_path,spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=board.init_state.pos,rot=board.init_state.rot)))


class MinInsertEnv(native.InsertEnv):
    def __init__(self,cfg,group='B0',seed=0,output=None,cases=None,**kw):
        self.group=group;self.training_seed=seed;self.output=Path(output) if output else None
        self.cases=cases;self.cursor=0;self.scene_rng=np.random.default_rng(seed);self.total_transitions=0
        self.completed=[];self.last_step=None;self.episode_records=[];self.record_trace=False;self.sampled_trace=[];self.sampled_episodes=0
        super().__init__(cfg,**kw)
        n=self.num_envs;d=self.device
        self.ik=IK('ur_description/urdf/ur10e.urdf',cfg.robot1.init_state.pos,cfg.robot1.init_state.rot,cfg.joint_angles_list['robot1'])
        self.ident=torch.tensor([0,0,0,1,0,0,0.],device=d).repeat(n,1)
        for name in ['socket_true','socket_estimated','xtrue','xhat','star','command','reset_ee']:
            setattr(self,name,self.ident.clone())
        self.root_tip=self.ident.clone();self.root_tip[:,2]=-.05;self.root_tip[:,3:]=torch.tensor([0,1,0,0.],device=d)
        self.valid=torch.zeros(n,device=d,dtype=torch.bool);self.just_reset=~self.valid.clone()
        self.first_substep=self.just_reset.clone();self.hold=torch.zeros(n,device=d,dtype=torch.long)
        self.ever_success=self.valid.clone();self.bad=self.valid.clone();self.case_info=[None]*n
        self.board_peak=torch.zeros(n,device=d);self.depth_history=torch.zeros(n,24,device=d);self.peak=torch.zeros(n,device=d);self.residual_sq=torch.zeros(n,device=d);self.saturation=torch.zeros(n,device=d)
        self.first_success=torch.full((n,),float("nan"),device=d);self.returns=torch.zeros(n,device=d);self.started=np.zeros(n);self.episode_serial=0
        if self.output:self.output.mkdir(parents=True,exist_ok=True)

    def create_rigid_attachments(self,attachment_path,robot_path,extra_attachments=[]):
        # clone_environments(copy_from_source=False) inherits env_0's new joint.
        # Define/override that same prim in every env rather than adding _1 joints.
        assert not extra_attachments, 'This experiment is scoped to one Hexagon grasp'
        stage=native.stage_utils.get_current_stage();joints=[]
        xr=native.IPose.from_pose(self.cfg.initial_cfg.PEG_T_TOOL).invert().to_vec()[0]
        for i in range(self.num_envs):
            prefix=f'/World/envs/env_{i}'
            joint=UsdPhysics.FixedJoint.Define(stage,prefix+'/'+robot_path+'/AssemblerFixedJoint')
            joint.GetBody0Rel().SetTargets([prefix+'/'+robot_path+'/tool0'])
            joint.GetBody1Rel().SetTargets([prefix+'/'+attachment_path])
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*xr[:3].cpu().tolist()))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(*xr[3:].cpu().tolist()))
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(0,0,0));joint.GetLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))
            joints.append(joint)
        return joints

    def _setup_scene(self):
        super()._setup_scene()
        self.scene.sensors['peg_contact']=ContactSensor(ContactSensorCfg(
            prim_path='/World/envs/env_.*/'+self.cfg.initial_cfg.PEG_ATTACHMENT_PRIM,
            update_period=0.,history_length=1, filter_prim_paths_expr=['/World/envs/env_.*/hole']))

    def _reset_idx(self,env_ids):
        DirectRLEnv._reset_idx(self,env_ids)
        if not hasattr(self,'ik'):return
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        if len(ids)==0:return
        robot=self.robots['robot1'];qs=robot.data.default_joint_pos[ids].clone()
        roots=self.scene['peg'].data.default_root_state[ids].clone();roots[:,7:]=0
        manifests=[];boards=torch.zeros(len(ids),13,device=self.device)
        for row,idx in enumerate(ids.cpu().tolist()):
            if self.cases is None:
                case=make_case(int(self.scene_rng.integers(1000000,2**31)), 'E1')
            else:
                # Padding repeats are explicitly excluded by evaluation driver.
                case=dict(self.cases[self.cursor%len(self.cases)]);self.cursor+=1
            sol,actual,check=self.ik.solve(case['ee_start'])
            case['robot_arm_joint_pos']=sol;case['ee_start_fk']=actual;case['validity']=check
            if check['status']!='valid':raise RuntimeError('Invalid pre-rollout scene: '+json.dumps(case))
            # Conservative CAD tip clearance at reset; evaluated before policy.
            initial_tip=compose(torch.tensor(actual),torch.tensor(case['X_true']))
            in_socket=compose(inverse(torch.tensor(case['socket_true'])),initial_tip)
            tilt=float(log(in_socket[3:]).norm())
            clearance=-float(in_socket[2])-.021*np.sin(min(tilt,np.pi/2))
            check['peg_clearance_lower_bound_m']=clearance
            check['full_robot_collision_check']='not exhaustive; IK limits and local work volume only'
            if case['split']!='fixture' and clearance<.005:
                raise RuntimeError('Invalid initial peg clearance before rollout: '+json.dumps(case))
            self.case_info[idx]=case
            for name,key in [('socket_true','socket_true'),('socket_estimated','socket_estimated'),('xtrue','X_true'),('xhat','X_hat'),('star','T_SP_star')]:
                getattr(self,name)[idx]=torch.tensor(case[key],device=self.device)
            self.reset_ee[idx]=torch.tensor(actual,device=self.device)
            # True grasp never replaces xhat; physical fixed joint is E -> peg root.
            xr=compose(self.xtrue[idx],inverse(self.root_tip[idx]))
            fixed=self.robot1_fixed_joints[idx]
            fixed.GetLocalPos0Attr().Set(Gf.Vec3f(*xr[:3].cpu().tolist()))
            fixed.GetLocalRot0Attr().Set(Gf.Quatf(*xr[3:].cpu().tolist()))
            fixed.GetLocalPos1Attr().Set(Gf.Vec3f(0,0,0));fixed.GetLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))
            boards[row,:7]=torch.tensor(case['board_true'],device=self.device)
            boards[row,:3]+=self.scene.env_origins[idx]
            qs[row,self._joint_ids['robot1']]=torch.tensor(sol,device=self.device)
            case['robot_joint_pos']=qs[row].cpu().tolist();case['robot_joint_vel']=[0.]*len(case['robot_joint_pos'])
            roots[row,:7]=compose(self.reset_ee[idx],xr)
            roots[row,:3]+=self.scene.env_origins[idx]
            self.started[idx]=time.monotonic();manifests.append(case)
        robot.write_joint_state_to_sim(qs,torch.zeros_like(qs),env_ids=ids)
        robot.set_joint_position_target(qs,env_ids=ids)
        self.scene['peg'].write_root_state_to_sim(roots,env_ids=ids)
        self.scene['hole'].write_root_state_to_sim(boards,env_ids=ids)
        self.command[ids]=self.reset_ee[ids];self.valid[ids]=False;self.just_reset[ids]=True;self.first_substep[ids]=True
        self.hold[ids]=0;self.ever_success[ids]=False;self.first_success[ids]=float('nan')
        initroot=roots[:,:7].clone();initroot[:,:3]-=self.scene.env_origins[ids]
        self.bad[ids]=true_metrics(initroot,self.root_tip[ids],self.socket_true[ids],self.star[ids])['invalid']
        for x in [self.peak,self.board_peak,self.residual_sq,self.saturation,self.returns]:x[ids]=0
        self._ik_controllers['robot1'].reset(ids)
        if self.output and (self.cases is not None or self.sampled_episodes<3):
            with (self.output/('realized_manifest.jsonl' if self.cases is not None else 'reset_manifest_sample.jsonl')).open('a') as f:
                for case in manifests:f.write(json.dumps(case)+'\n')

    def ee(self):
        r=self.robots['robot1'];pose=torch.cat(self._compute_frame_pose('robot1',r),-1)
        pose[self.just_reset]=self.reset_ee[self.just_reset]
        return pose
    def velocity(self):
        v=self.robots['robot1'].data.body_state_w[:,self._body_idx['robot1'],7:13].clone()
        v[self.just_reset]=0;return v
    def _get_observations(self):
        if not hasattr(self,'xhat'):return super()._get_observations()
        obs=actor_features(self.ee(),self.velocity(),self.xhat,self.socket_estimated,self.star,self.command,self.valid,self.group!='B0',self.group=='B2')
        assert torch.isfinite(obs).all()
        return {'policy':obs}
    def _pre_physics_step(self,actions):
        self.actions=actions.clamp(-1,1).clone()
        self.actor_before=self._get_observations()['policy'].clone()
        ee=self.ee();self.command,self.tip_cmd=tip_command(ee,self.xhat,self.socket_estimated,self.actions,self.group!='B0')
        self.saturation+=(self.actions[:,:3].norm(dim=-1)>1)|(self.actions[:,3:].norm(dim=-1)>1)
        self._ik_controllers['robot1'].set_command(self.command,ee[:,:3],ee[:,3:])
        self.valid[:]=True
    def _apply_action(self):
        # Initial FK/Jacobian after tensor reset is stale until the first physics
        # substep. Hold the reset joint target for that substep, then use native IK.
        r=self.robots['robot1'];p,q=self._compute_frame_pose('robot1',r)
        joint=r.data.joint_pos[:,self._joint_ids['robot1']]
        des=self._ik_controllers['robot1'].compute(p,q,self._compute_frame_jacobian('robot1',r),joint)
        des[self.first_substep]=joint[self.first_substep]
        r.set_joint_position_target(des,joint_ids=self._joint_ids['robot1'])
        self.first_substep[:]=False
        self.just_reset[:]=False
    def peg_root(self):
        pose=self.scene['peg'].data.root_state_w[:,:7].clone();pose[:,:3]-=self.scene.env_origins;return pose
    def physical_socket(self):
        board=self.scene['hole'].data.root_state_w[:,:7].clone();board[:,:3]-=self.scene.env_origins
        from scenarios import CENTER,TOP
        offset=torch.tensor([*CENTER,TOP,0,1,0,0.],device=self.device).expand_as(board)
        return compose(board,offset)
    def _get_dones(self):
        self.socket_true[:]=self.physical_socket()
        metrics=true_metrics(self.peg_root(),self.root_tip,self.socket_true,self.star)
        self.hold=torch.where(metrics['candidate'],self.hold+1,0)
        newly=(self.hold>=5)&~self.ever_success
        self.first_success[newly]=self.episode_length_buf[newly]*self.step_dt
        self.ever_success|=self.hold>=5;self.bad|=metrics['invalid']
        # Fixed duration preserves synchronous transitions; success still requires hold.
        timeout=self.episode_length_buf>=self.max_episode_length
        return torch.zeros_like(timeout),timeout
    def _get_rewards(self):
        peg=self.peg_root();target=compose(compose(self.socket_true,self.star),inverse(self.root_tip))
        # Preserve native reward structure, changing only target to actual board.
        difference=compose(target,inverse(peg))
        reward=-torch.cat((difference[:,:3],log(difference[:,3:])),dim=-1).norm(dim=-1).clamp(max=2)
        self.returns+=reward;self.total_transitions+=self.num_envs
        measured=self.ee();tip=compose(measured,self.xhat);cmd=compose(self.command,self.xhat)
        residual=torch.cat((cmd[:,:3]-tip[:,:3],log(qmul(cmd[:,3:],qinv(tip[:,3:])))),-1)
        force=self.scene.sensors['peg_contact'].data.net_forces_w[:,0].clone()
        self.peak=torch.maximum(self.peak,force.norm(dim=-1));self.residual_sq+=(residual[:,:3]**2).sum(-1)
        met=true_metrics(peg,self.root_tip,self.socket_true,self.star)
        board_force=self.scene.sensors['peg_contact'].data.force_matrix_w[:,0].sum(dim=1).clone()
        self.board_peak=torch.maximum(self.board_peak,board_force.norm(dim=-1))
        self.depth_history[:,(self.common_step_counter-1)%24]=met['depth']
        self.last_step={'actor_observation':self.actor_before.clone(),'action':self.actions.clone(),'commanded_ee':self.command.clone(),
          'measured_ee':measured.clone(),'tracking_residual_world':residual,'diagnostic_peg_root':peg,'diagnostic_socket_true':self.socket_true.clone(),'diagnostic_X_true':self.xtrue.clone(),'diagnostic_X_measured':compose(inverse(measured),compose(peg,self.root_tip)),'diagnostic_contact_force_world_N':force,'diagnostic_board_contact_force_world_N':board_force,
          'reward':reward.clone(),'done':self.reset_buf.clone(),'episode_step':self.episode_length_buf.clone(),
          'time_s':self.episode_length_buf*self.step_dt,'command_time_s':(self.episode_length_buf-1)*self.step_dt}
        if self.cases is None and self.output and self.sampled_episodes<3:
            self.sampled_trace.append({k:v[0].detach().cpu().numpy() for k,v in self.last_step.items()})
            if self.reset_buf[0]:
                np.savez_compressed(self.output/f'training_trace_{self.sampled_episodes}.npz',
                    **{k:np.stack([r[k] for r in self.sampled_trace]) for k in self.sampled_trace[0]})
                self.sampled_trace=[];self.sampled_episodes+=1
        for i in self.reset_buf.nonzero(as_tuple=False).flatten().cpu().tolist():
            n=int(self.episode_length_buf[i]);case=self.case_info[i]
            success=bool(self.ever_success[i]) and not bool(self.bad[i])
            reason='success' if success else 'abnormal_physics' if self.bad[i] else 'timeout'
            if not success and not self.bad[i]:
                if met['lateral'][i]>.15 or met['depth'][i]<-.15:reason='out_of_workspace'
                elif n>=24 and self.board_peak[i]>.5 and self.depth_history[i].max()-self.depth_history[i].min()<.001:
                    reason='contact_no_progress'
            row={'group':getattr(self,'report_group',self.group),'policy_kind':getattr(self,'policy_kind','diagnostic'),'checkpoint_id':getattr(self,'checkpoint_id',None),'training_seed':self.training_seed,'case_id':case['case_id'],'suite':case['suite'],'num_steps':n,
             'success':success,'termination_reason':reason,'completion_time_s':float(self.first_success[i]) if success else None,'final_depth_m':float(met['depth'][i]),
             'final_lateral_error_m':float(met['lateral'][i]),'final_rotation_error_rad':float(met['rotation'][i]),
             'peak_contact_measure':float(self.peak[i]),'peak_board_contact_force_N':float(self.board_peak[i]),'contact_measure_type':'net force on peg, world, N; torque unavailable',
             'tracking_residual_rms':float((self.residual_sq[i]/n).sqrt()),'action_saturation_rate':float(self.saturation[i]/n),
             'wall_time_s':time.monotonic()-self.started[i],'return':float(self.returns[i]),'transitions':self.total_transitions}
            self.completed.append(row)
            if self.output:
                with (self.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return reward
