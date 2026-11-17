"""Pure torch SE(3) interface. P is the CAD bottom tip, S has +Z into hole.
Quaternions are wxyz; translations in m, rotation vectors in rad.
Only explicit deployed arguments enter actor_features (no simulator object).
"""
import torch

def qmul(a,b):
    aw,av=a[...,:1],a[...,1:];bw,bv=b[...,:1],b[...,1:]
    return torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=-1)),-1)
def qinv(q): return torch.cat((q[...,:1],-q[...,1:]),-1)
def rotate(q,v):
    t=2*torch.cross(q[...,1:],v,dim=-1)
    return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)
def exp(v):
    angle=v.norm(dim=-1,keepdim=True)
    return torch.cat((torch.cos(angle/2),.5*torch.sinc(angle/(2*torch.pi))*v),-1)
def log(q):
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    q=torch.where(q[...,:1]<0,-q,q)
    n=q[...,1:].norm(dim=-1,keepdim=True)
    return q[...,1:]*(2*torch.atan2(n,q[...,:1])/n.clamp_min(1e-9))
def compose(a,b):return torch.cat((a[...,:3]+rotate(a[...,3:],b[...,:3]),qmul(a[...,3:],b[...,3:])),-1)
def inverse(a):
    qi=qinv(a[...,3:]);return torch.cat((rotate(qi,-a[...,:3]),qi),-1)
def delta(v):return torch.cat((v[...,:3],exp(v[...,3:])),-1)
def normclip(x,limit):return x*(limit/x.norm(dim=-1,keepdim=True).clamp_min(limit))
def tip_command(ee,xhat,socket,action,local,translation_limit=.05,rotation_limit=.1):
    tip=compose(ee,xhat);dp=normclip(action[...,:3],1)*translation_limit
    dr=normclip(action[...,3:6],1)*rotation_limit
    if local:dp=rotate(socket[...,3:],dp);dr=rotate(socket[...,3:],dr)
    cmd=torch.cat((tip[...,:3]+dp,qmul(exp(dr),tip[...,3:])),dim=-1)
    return compose(cmd,inverse(xhat)),cmd

def actor_features(ee,ee_velocity,xhat,socket,star,previous_ee_command,valid,local,residual):
    tip=compose(ee,xhat);goal=compose(socket,star)
    pos=goal[...,:3]-tip[...,:3]
    rot=log(qmul(goal[...,3:],qinv(tip[...,3:])))
    omega=ee_velocity[...,3:]
    vel=ee_velocity[...,:3]+torch.cross(omega,rotate(ee[...,3:],xhat[...,:3]),dim=-1)
    grav=torch.zeros_like(pos);grav[...,2]=-1
    cmd=compose(previous_ee_command,xhat)
    ep=cmd[...,:3]-tip[...,:3];er=log(qmul(cmd[...,3:],qinv(tip[...,3:])))
    if local:
        qi=qinv(socket[...,3:]);pos=rotate(qi,pos);rot=rotate(qi,rot);vel=rotate(qi,vel)
        omega=rotate(qi,omega);grav=rotate(qi,grav);ep=rotate(qi,ep);er=rotate(qi,er)
    # Global orientation kept identical in all groups. Rotation matrix columns.
    ex=torch.zeros_like(pos);ex[...,0]=1;ey=torch.zeros_like(pos);ey[...,1]=1
    orientation=torch.cat((rotate(socket[...,3:],ex),rotate(socket[...,3:],ey)),-1)
    res=torch.cat((ep,er),-1)*valid[...,None]
    if not residual:res=torch.zeros_like(res)
    return torch.cat((pos,rot,vel,omega,orientation,grav,res),-1)

def true_metrics(peg_root,root_tip,socket,star):
    relative=compose(inverse(socket),compose(peg_root,root_tip))
    error=relative[...,:3]-star[...,:3]
    # Hexagon permits 6 in-plane rotations, evaluated in the socket frame.
    angles=[]
    for k in range(6):
        v=torch.zeros_like(error);v[...,2]=k*torch.pi/3
        angles.append(log(qmul(relative[...,3:],qinv(qmul(star[...,3:],exp(v))))).norm(dim=-1))
    angle=torch.stack(angles,-1).min(-1).values
    lateral=error[...,:2].norm(dim=-1);depth=relative[...,2]
    finite=torch.isfinite(relative).all(-1)
    # Conservative tolerances: 0.75 mm < half the ~1.49 mm CAD side clearance.
    candidate=(lateral<=.00075)&(error[...,2].abs()<=.001)&(angle<=torch.pi/90)&finite
    invalid=(depth>star[...,2]+.003)|(~finite)
    candidate &= ~invalid
    return {'depth':depth,'lateral':lateral,'rotation':angle,'candidate':candidate,'invalid':invalid}
