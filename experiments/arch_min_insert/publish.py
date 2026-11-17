"""Publish an explicit snapshot using a temporary Git index; leave user staging alone."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser();p.add_argument('--include_code',action='store_true');p.add_argument('--message',default='Update ARCH INSERT experiment evidence');a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/arch_minexp';STATE=OUT/'publication.json'
REMOTE='https://github.com/Adeline123456789/improve-ARCH.git'
def git(args,env=None):return subprocess.check_output(['git',*args],cwd=ROOT,env=env,text=True).strip()
old=json.loads(STATE.read_text()) if STATE.exists() else None
parent=old['commit'] if old else 'a6a224b79f57e0b6c149e5b0f9d51a9b56ca131e'
remote=git(['ls-remote',REMOTE,'refs/heads/main']);remote_commit=remote.split()[0] if remote else None
assert remote_commit==(old['commit'] if old else None),'Remote main changed; refusing to overwrite external work'
paths=[]
for path in OUT.rglob('*'):
    if not path.is_file() or 'runner' in path.parts or '__pycache__' in path.parts:continue
    rel=path.relative_to(ROOT)
    if path.name in ['publication.json'] or path.suffix in ['.tmp']:continue
    if path.suffix=='.pth' and path.parent.name!='checkpoints':continue
    if path.name=='trajectories.npz' and path.parent.name.startswith(('validation_','test_','screen_','confirm_','extended_','smoke')):continue
    if path.name=='reset_manifest_sample.jsonl':continue
    if path.suffix not in ['.md','.patch','.json','.jsonl','.yaml','.txt','.csv','.log','.npz','.pth','.png','.pdf','.mp4']:continue
    assert path.stat().st_size<90_000_000,(path,'Unexpected large artifact')
    paths.append(str(rel))
if a.include_code:
    paths.extend(str(x.relative_to(ROOT)) for x in (ROOT/'experiments/arch_min_insert').rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix in ['.py','.sh','.md','.yaml'])
    for name in ['multi-robot-assembly/multi_arm_assembly/rl_components/my_agent.py','multi-robot-assembly/multi_arm_assembly/train_rl.py','environment/env-spec.json','environment/installed-versions.json','environment/isaaclab-replicator.patch']:
        if (ROOT/name).exists():paths.append(name)
with tempfile.TemporaryDirectory(prefix='arch-minexp-index-') as tmp:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'))
    git(['read-tree',parent],env)
    for start in range(0,len(paths),100):git(['add','-f','--',*paths[start:start+100]],env)
    if a.include_code:
        diff=git(['diff','--cached',parent,'--','experiments/arch_min_insert','multi-robot-assembly/multi_arm_assembly/rl_components/my_agent.py','multi-robot-assembly/multi_arm_assembly/train_rl.py'],env)
        (OUT/'code_diff.patch').write_text(diff+'\n');git(['add','-f','artifacts/arch_minexp/code_diff.patch'],env)
    tree=git(['write-tree'],env)
    if tree==git(['rev-parse',parent+'^{tree}']):print('No changes to publish');raise SystemExit(0)
    commit=git(['commit-tree',tree,'-p',parent,'-m',a.message],env)
    subprocess.run(['git','push',REMOTE,commit+':refs/heads/main'],cwd=ROOT,check=True)
STATE.write_text(json.dumps({'remote':REMOTE,'branch':'main','commit':commit,'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'user_index_untouched':True},indent=2)+'\n')
print('PUBLISHED',commit,REMOTE)
