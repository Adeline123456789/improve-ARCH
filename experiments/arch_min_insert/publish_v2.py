"""Publish this experiment's code and measured evidence using an isolated Git index."""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TARGET='https://github.com/Adeline123456789/improve-ARCH.git'
BRANCH='refs/heads/codex/arch-v2-evidence'
def run(*args,env=None):
    return subprocess.check_output(args,cwd=ROOT,env=env,text=True).strip()
def publish(out,message):
    parent=out.parent
    selected=set(p for p in (ROOT/'experiments/arch_min_insert').glob('*.py'))
    selected.update(p for p in parent.iterdir() if p.is_file() and p.suffix in {'.json','.md'})
    for p in out.iterdir():
        if p.is_file() and p.suffix in {'.json','.yaml','.md','.csv','.jsonl','.log'}:
            selected.add(p)
    for name in ['diagnostics','manifests']:
        selected.update(p for p in (out/name).glob('*') if p.is_file() and not p.name.endswith('.tmp'))
    selected.update(p for p in out.glob('*/completion.json'))
    for comp in out.glob('*/completion.json'):
        folder=comp.parent
        identity=folder/'identity.json'
        if not identity.exists():continue
        train=json.loads(identity.read_text())['mode']=='train'
        # The generator plus manifest and seed preserve training input. Per-episode
        # data and representative dynamics are published for every training process.
        for p in folder.iterdir():
            if p.is_file() and p.suffix in {'.json','.yaml','.jsonl','.npz','.pt'}:
                if p.name=='realized_manifest.jsonl' and train:continue
                selected.add(p)
    report=json.loads((out/'gate_report.json').read_text())
    curve=report['G2'].get('curve',[])
    # Gate resumption checks the hashes and frames of all scheduled models.
    selected.update(out/row['checkpoint'] for row in curve)
    for row in report['G3'].get('curve',[])+report['G4'].get('runs',[]):
        selected.add(out/row['checkpoint'])
    # A single JSONL is convenient locally. Very long formal runs remain
    # downloaded as complete per-process JSONL files under the same names.
    selected={p for p in selected if p.name!='episodes.jsonl' or p.parent!=out or p.stat().st_size<90*1024**2}
    with tempfile.TemporaryDirectory() as temp:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(temp)/'index')}
        prev=run('git','rev-parse',BRANCH)
        run('git','read-tree',prev,env=env)
        paths=[str(p.relative_to(ROOT)) for p in sorted(selected)]
        for i in range(0,len(paths),100):
            run('git','add','-f','--',*paths[i:i+100],env=env)
        if out/'episodes.jsonl' not in selected:
            run('git','rm','--cached','--ignore-unmatch','--',str((out/'episodes.jsonl').relative_to(ROOT)),env=env)
        tree=run('git','write-tree',env=env)
        commit=prev
        if tree!=run('git','rev-parse',prev+'^{tree}'):
            commit=run('git','commit-tree',tree,'-p',prev,'-m',message)
            run('git','update-ref',BRANCH,commit,prev)
        run('git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential','push',TARGET,BRANCH)
        remote=run('git','ls-remote',TARGET,BRANCH).split()[0]
        assert remote==commit
        print(commit,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--message',required=True)
    args=p.parse_args()
    publish(args.output.resolve(),args.message)
