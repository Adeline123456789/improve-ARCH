"""Account completed simulator invocations from saved run outputs and Kit logs."""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

def budget(roots):
    jobs=[]
    for root in roots:
        kit={}
        for log in root.rglob('*.log'):
            last=None
            with log.open(errors='replace') as stream:
                for line in stream:
                    if 'MINEXP_COMPLETE ' in line:
                        last=Path(line.split()[-1]).name
                    elif last and 'Simulation App Shutting Down' in line:
                        match=re.search(r'\[([\d.]+)s\]',line)
                        if match:kit[last]=float(match[1])
                        last=None
        for directory in sorted(root.iterdir()):
            identity=directory/'identity.json'
            if not identity.is_file():
                continue
            ident=json.loads(identity.read_text())
            completion=directory/'completion.json'
            info=json.loads(completion.read_text()) if completion.is_file() else {}
            start=int(subprocess.check_output(['stat','-c','%W',directory]))
            end=(completion.stat().st_mtime if info else time.time())
            elapsed=kit.get(directory.name,max(0,end-start))
            observed=info.get('actual_env_transitions')
            episodes=directory/'episodes.jsonl'
            if observed is None and ident['mode']=='train' and episodes.is_file():
                with episodes.open() as stream:
                    observed=max(json.loads(line)['transitions'] for line in stream if line.endswith('\n'))
            jobs.append(dict(last_recorded_training_step=observed,directory=str(directory),mode=ident['mode'],group=ident['group'],seed=ident['seed'],
                run_complete=bool(info),start_unix=start,simulator_wall_seconds=elapsed,
                source='Kit shutdown log' if directory.name in kit else 'output-file interval estimate',
                new_training_steps=info.get('actual_env_transitions') if ident['mode']=='train' else None,
                evaluation_steps=info.get('transitions') if ident['mode']!='train' else None))
        failed=root/'g2_init_abort_512'
        if failed.exists():
            log=(root/'training.log').read_text(errors='replace')
            assert 'malloc(): invalid size' in log
            jobs.append(dict(directory=str(failed),mode='train',group='B0',seed=101,run_complete=False,
                start_unix=int(subprocess.check_output(['stat','-c','%W',failed])),simulator_wall_seconds=8.,
                source='last PhysX log at 7.508 s, rounded estimate',new_training_steps=0,
                reason='SIGABRT during scene replication before first rollout'))
    spans=sorted((j['start_unix'],j['start_unix']+j['simulator_wall_seconds']) for j in jobs)
    union=0;end=0
    for left,right in spans:
        union+=max(0,right-max(left,end))
        end=max(end,right)
    return dict(jobs=jobs,completed_training_steps=sum(j['new_training_steps'] for j in jobs if j['new_training_steps'] is not None),
                recorded_training_steps_lower_bound=sum(j['last_recorded_training_step'] for j in jobs if j.get('last_recorded_training_step') is not None),
                measured_evaluation_and_probe_steps=sum(j['evaluation_steps'] for j in jobs if j.get('evaluation_steps') is not None),
                simulator_process_seconds=sum(j['simulator_wall_seconds'] for j in jobs),
                gpu_wall_seconds_union_estimate=union,elapsed_first_to_last_simulator_seconds=end-spans[0][0],
                diagnostic_step_limit=5400000,total_step_limit=30000000,gpu_wall_limit_seconds=21600)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path)
    paths=p.parse_args().roots
    measured=budget(paths)
    (Path(os.path.commonpath(paths))/'budget_ledger.json').write_text(json.dumps(measured,indent=2)+'\n')
    print(json.dumps({k:v for k,v in measured.items() if k!='jobs'}))
