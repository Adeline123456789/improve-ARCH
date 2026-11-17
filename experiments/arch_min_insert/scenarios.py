"""Independent RNG domains; save actual transforms, not just seeds."""
import numpy as np
import torch
from geometry import compose,delta,inverse

# CAD center of the 48-vertex rounded hexagonal aperture, expressed in board frame.
CENTER=(-.0681832563,-.0885440744)
TOP=.02436008

def tensor(x):return torch.as_tensor(x,dtype=torch.float64)
def make_case(seed,suite,split='train'):
    r=np.random.default_rng(seed)
    layout=r.uniform(-.02,.02,2);yaw=r.uniform(-15,15)*np.pi/180
    if suite=='E0':layout[:]=0;yaw=0
    stratum='id'
    if suite=='E2':
        if seed%2:
            layout[0]=r.choice([-1,1])*r.uniform(.020001,.04);stratum='translation'
        else:yaw=r.choice([-1,1])*r.uniform(15.0001,30)*np.pi/180;stratum='yaw'
    scale=2 if suite=='E3' else 1
    bs=r.uniform(-1,1,6)*[.001*scale,.001*scale,0,*([np.deg2rad(.5)*scale]*3)]
    bg=r.uniform(-1,1,6)*([.0005*scale]*3+[np.deg2rad(.5)*scale]*3)
    if suite=='E0':bs[:]=0;bg[:]=0
    board=compose(tensor([layout[0],.18+layout[1],.025,0,0,0,1]),delta(tensor([0,0,0,0,0,yaw])))
    board_socket=tensor([*CENTER,TOP,0,1,0,0])
    socket=compose(board,board_socket)
    estimated=compose(socket,delta(tensor(bs)))
    # E=tool0, CAD bottom tip P: +0.32 m along E's Z, same orientation.
    xhat=tensor([0,0,.32,1,0,0,0]);xtrue=compose(xhat,delta(tensor(bg)))
    star=tensor([0,0,TOP+.02,1,0,0,0])
    pre=star.clone();pre[2]-=.06
    start=compose(compose(estimated,pre),inverse(xhat))
    out={'case_id':f'{split}-{suite}-{seed:08d}','split':split,'suite':suite,'seed':int(seed),'stratum':stratum,
         'board_true':board.tolist(),'socket_true':socket.tolist(),'socket_estimated':estimated.tolist(),
         'X_true':xtrue.tolist(),'X_hat':xhat.tolist(),'T_SP_star':star.tolist(),'T_SP_pre':pre.tolist(),
         'ee_start':start.tolist(),'socket_bias':bs.tolist(),'grasp_bias':bg.tolist(),
         'approach_retreat_m':.06,'physics':{'dt':1/120,'decimation':5,'stiffness':400,'damping':800,'effort_limit':100,'gravity_disabled_robot_peg':True},
         'success_version':'cad-hex-v1','error_outside_training':bool(np.any(np.abs(bs)>[.001,.001,0,*([np.deg2rad(.5)]*3)]) or np.any(np.abs(bg)>[.0005]*3+[np.deg2rad(.5)]*3)),
         'validity':{'status':'pending_ik_and_initial_clearance'}}
    return out

class IK:
    """Reusable existing UR10e URDF, avoids native solve_ik's connection leak/reseeding."""
    def __init__(self,urdf,base_pos,base_quat,seed_joints):
        import pybullet as p
        self.p=p;self.cid=p.connect(p.DIRECT)
        self.robot=p.loadURDF(urdf,base_pos,[base_quat[1],base_quat[2],base_quat[3],base_quat[0]],useFixedBase=True,physicsClientId=self.cid)
        self.joints=[i for i in range(p.getNumJoints(self.robot,physicsClientId=self.cid)) if p.getJointInfo(self.robot,i,physicsClientId=self.cid)[2]!=p.JOINT_FIXED]
        self.link=next(i for i in range(p.getNumJoints(self.robot,physicsClientId=self.cid)) if p.getJointInfo(self.robot,i,physicsClientId=self.cid)[12]==b'tool0')
        self.seed=seed_joints
    def solve(self,pose):
        p=self.p
        for j,q in zip(self.joints,self.seed):p.resetJointState(self.robot,j,float(q),physicsClientId=self.cid)
        q=p.calculateInverseKinematics(self.robot,self.link,pose[:3],[*pose[4:],pose[3]],maxNumIterations=1000,residualThreshold=1e-7,physicsClientId=self.cid)
        for j,v in zip(self.joints,q):p.resetJointState(self.robot,j,v,physicsClientId=self.cid)
        fk=p.getLinkState(self.robot,self.link,computeForwardKinematics=True,physicsClientId=self.cid)
        actual=[*fk[4],fk[5][3],*fk[5][:3]]
        error=float(np.linalg.norm(np.array(actual[:3])-pose[:3]))
        # Numerical IK + conservative preinsertion gap; no silent rejection after rollout.
        from geometry import log,qmul,qinv
        angular=float(log(qmul(tensor(actual[3:]),qinv(tensor(pose[3:])))).norm())
        limits=all(p.getJointInfo(self.robot,j,physicsClientId=self.cid)[8]-1e-5<=v<=p.getJointInfo(self.robot,j,physicsClientId=self.cid)[9]+1e-5 for j,v in zip(self.joints,q))
        return list(q),actual,{'status':'valid' if error<.0001 and angular<.001 and limits else 'invalid_ik','position_error_m':error,'rotation_error_rad':angular,'joint_limits':limits}
