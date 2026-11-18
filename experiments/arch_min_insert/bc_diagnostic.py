"""Fit deployed observations and effective actions from six nominal demonstrations."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from geometry import normclip

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'multi-robot-assembly'))
from multi_arm_assembly.rl_components.my_network import MyNetworkBuilder
from multi_arm_assembly.rl_components.my_a2c_model import MyA2CContinuousLogStd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
a=p.parse_args()
out=a.run/'diagnostics/bc'
out.mkdir(exist_ok=False)
torch.manual_seed(101)
torch.set_num_threads(2)
params=json.loads((a.run/'g1/agent_config.json').read_text())['params']
net=MyNetworkBuilder()
net.load(params['network'])
model=MyA2CContinuousLogStd(net).build(dict(actions_num=6,input_shape={'policy':(27,)},num_seqs=6,
                                           normalize_value=True,normalize_input=True))
path=a.run/'g0_scripted/trajectories.npz'
with np.load(path) as data:
    obs=torch.from_numpy(data['actor_observation'][:,:8].transpose(1,0,2).copy())
    raw=torch.from_numpy(data['action'][:,:8].transpose(1,0,2).copy())
    labels=list(data['case_id'][0,:8])
action=torch.cat([normclip(raw[...,:3],1.),normclip(raw[...,3:],1.)],-1)
with torch.no_grad():
    model.norm_obs({'policy':obs[:6].reshape(-1,27)})
model.eval()
def predict(x):
    obs_norm=model.norm_obs({'policy':x.reshape(-1,27)})
    states=tuple(torch.zeros(s.shape[0],len(x),s.shape[2]) for s in model.get_default_rnn_state())
    mu,*_=model.a2c_network(dict(obs=obs_norm,rnn_states=states,seq_length=x.shape[1]))
    return mu.reshape(len(x),x.shape[1],6)
opt=torch.optim.Adam(model.parameters(),lr=.0003)
losses=[]
for iteration in range(180):
    mu=predict(obs[:6])
    loss=(mu-action[:6]).square().mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step()
    losses.append(float(loss.detach()))
with torch.no_grad():
    train_rmse=float((predict(obs[:6])-action[:6]).square().mean().sqrt())
    check_rmse=float((predict(obs[6:])-action[6:]).square().mean().sqrt())
assert np.isfinite(train_rmse) and losses[-1]<losses[0]
kind='BC diagnostic from six scripted nominal demonstrations'
torch.save(dict(model=model.state_dict(),policy_kind=kind),out/'bc_diagnostic.pth')
(out/'fit.json').write_text(json.dumps(dict(policy_kind=kind,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    fit_cases=labels[:6],offline_check_cases=labels[6:],physical_nominal_cases=1,updates=len(losses),
    train_action_rmse=train_rmse,check_action_rmse=check_rmse,loss_history=losses,online_training_transitions=0),indent=2)+'\n')
print('Action RMSE',train_rmse,check_rmse)
