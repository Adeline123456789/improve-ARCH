"""Risk-focused checks independent of Isaac; physical detector validation is separate."""
import json,torch
from geometry import *
torch.manual_seed(7)
n=32;dtype=torch.float64
pose=lambda:delta(torch.randn(n,6,dtype=dtype)*.1)
ee,x,s,g,cmd=pose(),pose(),pose(),pose(),pose();v=torch.randn(n,6,dtype=dtype)
valid=torch.ones(n,dtype=torch.bool);a=torch.randn(n,6,dtype=dtype);z=torch.zeros(n,6,dtype=dtype)
obs=actor_features(ee,v,x,s,g,cmd,valid,True,True)
y=pose();ee2=compose(y,ee);s2=compose(y,s);cmd2=compose(y,cmd)
v2=torch.cat((rotate(y[:,3:],v[:,:3]),rotate(y[:,3:],v[:,3:])),dim=-1)
obs2=actor_features(ee2,v2,x,s2,g,cmd2,valid,True,True)
# Gravity is intentionally tied to world gravity; compare geometry/residual only.
assert torch.allclose(obs[:,:12],obs2[:,:12],atol=1e-10)
assert torch.allclose(obs[:,21:],obs2[:,21:],atol=1e-10)
ec,pc=tip_command(ee,x,s,a,True);ec2,pc2=tip_command(ee2,x,s2,a,True)
assert torch.allclose(compose(y,ec),ec2,atol=1e-10)
a[:,:3]=0;ec,pc=tip_command(ee,x,s,a,True)
assert torch.allclose(compose(ec,x)[:,:3],compose(ee,x)[:,:3],atol=1e-10)
assert actor_features(ee,v,x,s,g,cmd,valid,True,False)[:,21:].count_nonzero()==0
assert actor_features(ee,v,x,s,g,cmd,valid&False,True,True)[:,21:].count_nonzero()==0
# Per-environment translation cancellation, no double subtraction.
origin=torch.randn(n,3,dtype=dtype)*100
te=ee.clone();ts=s.clone();tc=cmd.clone()
for t in [te,ts,tc]:t[:,:3]+=origin;t[:,:3]-=origin
assert torch.allclose(obs,actor_features(te,v,x,ts,g,tc,valid,True,True),atol=1e-10)
# Geometry examples are algebraic fixtures, not proof of nonpenetrating dynamics.
ident=torch.tensor([[0,0,0,1,0,0,0]],dtype=dtype)
star=ident.clone();star[:,2]=.04436008
root_tip=ident.clone();root_tip[:,2]=-.05
root=compose(star,inverse(root_tip));met=true_metrics(root,root_tip,ident,star)
assert met['candidate'].item()
edge=root.clone();edge[:,0]+=.008;assert not true_metrics(edge,root_tip,ident,star)['candidate'].item()
through=root.clone();through[:,2]+=.02;assert true_metrics(through,root_tip,ident,star)['invalid'].item()
print(json.dumps({'algebra_tests':'passed','n':n,'dtype':str(dtype),'physical_detector':'requires simulator fixtures'}))
