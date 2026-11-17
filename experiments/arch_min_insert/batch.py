"""Sequential bounded runs; persists status, exact commands and verified checkpoints.
This driver never chooses a checkpoint using validation/test success.
"""
import argparse,hashlib,json,os,shlex,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--phase',default='screen')
p.add_argument('--transitions',type=int,default=1996800);p.add_argument('--num_envs',type=int,default=512)
p.add_argument('--wall_budget_s',type=float,default=19800);p.add_argument('--evaluate_test',action='store_true');a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/arch_minexp'
status_path=OUT/f'{a.phase}_job.json';start=time.monotonic()
source_hash={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['run.py','min_env.py','geometry.py','scenarios.py']}
status={'status':'starting','pid':os.getpid(),'phase':a.phase,'seeds':a.seeds,'budget_s':a.wall_budget_s,'source_sha256':source_hash,'runs':[]}
def save():
    status['elapsed_s']=time.monotonic()-start;tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(status_path)
def run(cmd,log):
    for name,expected in source_hash.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==expected,'Source changed mid-comparison'
    with (OUT/'commands.txt').open('a') as f:f.write(shlex.join(cmd)+' > '+str(log)+' 2>&1\n')
    with log.open('w') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        status.update(status='running',child_pid=child.pid,command=cmd,log=str(log));save()
        remaining=a.wall_budget_s-(time.monotonic()-start)
        try:code=child.wait(timeout=max(1,remaining))
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise RuntimeError('Wall budget reached; preserve latest periodic checkpoint, do not label complete')
        if code:raise RuntimeError(f'Child exited {code}; see {log}')
try:
    import torch
    for seed in a.seeds:
        for group in ['B0','B1','B2']:
            name=f'{a.phase}_{group}_s{seed}';directory=OUT/name
            record={'group':group,'seed':seed,'requested_transitions':a.transitions,'train_dir':str(directory),'status':'starting'}
            status['runs'].append(record);save()
            cmd=['bash',str(ROOT/'experiments/arch_min_insert/run_train.sh'),'--group',group,'--seed',str(seed),'--num_envs',str(a.num_envs),
                 '--transitions',str(a.transitions),'--headless','--output',str(directory)]
            run(cmd,OUT/'logs'/f'{name}.log')
            candidates=list((directory/'runner/run/nn').glob(f'*_ep_{a.transitions//(a.num_envs*100)}_*.pth'))
            assert len(candidates)==1,candidates
            checkpoint=candidates[0];state=torch.load(checkpoint,map_location='cpu')
            assert state['frame']==a.transitions,(state['frame'],a.transitions)
            assert all(torch.isfinite(v).all() for v in state['model'].values() if torch.is_tensor(v))
            copied=OUT/'checkpoints'/f'{group}_seed{seed}_{a.transitions}.pth';assert not copied.exists(),copied
            shutil.copy2(checkpoint,copied)
            record.update(status='trained',checkpoint=str(copied),checkpoint_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
                          actual_transitions=state['frame'],epoch=state['epoch'],has_optimizer='optimizer' in state,
                          normalization_keys=[k for k in state['model'] if 'running_mean_std' in k])
            save()
            for split in (['validation','test'] if a.evaluate_test else ['validation']):
                manifest=OUT/('evaluation_manifest.jsonl' if split=='test' else 'validation_manifest.jsonl')
                evalname=f'{split}_{group}_s{seed}_{a.transitions}'
                cmd=['bash',str(ROOT/'experiments/arch_min_insert/run_eval.sh'),'--group',group,'--seed',str(seed),'--num_envs','100',
                     '--checkpoint',str(copied),'--manifest',str(manifest),'--headless','--output',str(OUT/evalname)]
                run(cmd,OUT/'logs'/f'{evalname}.log')
                record[split]=str(OUT/evalname);save()
            record['status']='completed';save()
    status.update(status='completed');save()
except BaseException as exc:
    status.update(status='failed',error=repr(exc));save();raise
