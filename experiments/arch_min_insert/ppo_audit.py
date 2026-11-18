"""Instrument the existing PPO for two diagnostic rollouts only."""
import json
from pathlib import Path
import torch
from multi_arm_assembly.rl_components.my_agent import MyA2CAgent

class AuditAgent(MyA2CAgent):
    @torch.no_grad()
    def play_steps_rnn(self):
        self.audit_path=Path(self.config['train_dir']).parent
        self.reset_samples=0
        b=super().play_steps_rnn()
        self.model.eval()
        normalized=self.model.norm_obs(self._preproc_obs(b['obses']))
        replay={'obs':normalized,'rnn_states':b['rnn_states'],'seq_length':self.seq_length,'dones':b['dones']}
        mu,logstd,_,_=self.model.a2c_network(replay)
        nlp=self.model.neglogp(b['actions'],mu,logstd.exp(),logstd)
        err=float((nlp-b['neglogpacs']).abs().max())
        assert err<1e-3 and self.reset_samples>0,(err,self.reset_samples)
        torch.save(dict(normalized=normalized,old_neglogp=b['neglogpacs'],raw=b['actions'],mu=b['mus'],
                        log_std=b['sigmas'].log(),dones=b['dones'],hc=b['rnn_states']),self.audit_path/f'rollout_{self.epoch_num}.pt')
        td=self.experience_buffer.tensor_dict
        rew=td['rewards'];v=td['values'];last=self.get_values(self.obs)
        targets=self.discount_values(self.dones.float(),last,td['dones'].float(),v,rew)+v
        end=td['dones'][1:].bool()
        assert end.any()
        torch.testing.assert_close(targets[:-1][end],rew[:-1][end])
        open_mask=~self.dones.bool()
        assert open_mask.any()
        torch.testing.assert_close(targets[-1][open_mask],(rew[-1]+self.gamma*last)[open_mask])
        raw=b['actions']
        self.stats={'epoch':self.epoch_num,'reset_samples':self.reset_samples,'neglogp_max_error':err,
            'raw_action_mean':raw.mean(0).tolist(),'raw_action_std':raw.std(0).tolist(),
            'executed_action_component_std':raw.clamp(-1,1).std(0).tolist(),
            'translation_saturation':float((raw[:,:3].clamp(-1,1).norm(dim=1)>=1).float().mean()),
            'rotation_saturation':float((raw[:,3:].clamp(-1,1).norm(dim=1)>=1).float().mean()),
            'terminal_target_error':float((targets[:-1][end]-rew[:-1][end]).abs().max()),
            'optimizer_steps':0,'max_clip_fraction':0.}
        for name,t in {'reward':rew,'value':v,'return':targets,'advantage':targets-v}.items():
            assert torch.isfinite(t).all() and t.std()>0,name
            self.stats[name+'_std']=float(t.std())
        return b

    def get_action_values(self,obs):
        ended=self.dones.bool()
        if ended.any():
            assert all(s[:,ended].count_nonzero()==0 for s in self.rnn_states)
            self.reset_samples+=int(ended.sum())
        return super().get_action_values(obs)

    def train_epoch(self):
        head={n:p for n,p in self.model.named_parameters() if n.split('.')[-2]=='mu' or n.endswith('.sigma')}
        before={n:p.detach().clone() for n,p in head.items()}
        result=super().train_epoch()
        self.stats['actor_update_norm']={n:float((p.detach()-before[n]).norm()) for n,p in head.items()}
        assert len(head)>=2 and all(self.stats['actor_update_norm'].values())
        for losses in [result[4],result[5],result[8]]:
            assert torch.isfinite(torch.stack(losses)).all()
        self.stats['mean_kl']=float(torch.stack(result[8]).mean())
        with (self.audit_path/'ppo_updates.jsonl').open('a') as f:f.write(json.dumps(self.stats)+'\n')
        return result

    def train_actor_critic(self,b):
        result=super().train_actor_critic(b)
        mu,sigma=result[6:8]
        logp=self.model.neglogp(b['actions'],mu,sigma,sigma.log())
        cf=float((((b['old_logp_actions']-logp).exp()-1).abs()>self.e_clip).float().mean())
        grads=[p.grad.detach().norm() for p in self.model.parameters() if p.grad is not None]
        norm=float(torch.stack(grads).norm())
        assert 0<norm<float('inf')
        self.stats.update(optimizer_steps=self.stats['optimizer_steps']+1,gradient_norm=norm,
                          max_clip_fraction=max(cf,self.stats['max_clip_fraction']))
        return result
