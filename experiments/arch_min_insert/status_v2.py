"""Print gate counts and the latest physical rollout without reading a full training log."""
import argparse
import json
import time
from pathlib import Path

def show(out):
    report=json.loads((out/'gate_report.json').read_text())
    print(' '.join(f"{k}={report[k]['status']}" for k in ('G0','G1','G2','G3','G4')),flush=True)
    checkpoints={r['frame']:r for r in report['G2'].get('curve',[])}
    for path in out.glob('g2_eval_*/completion.json'):
        frame=int(path.parent.name.rsplit('_',1)[1])
        data=json.loads(path.read_text())
        checkpoints[frame]=dict(frame=frame,successes=data['successes'],episodes=data['num_complete_episodes'])
    for row in sorted(checkpoints.values(),key=lambda r:r['frame'])[-2:]:
        print(f"G2  {row['frame']:,}  nominal {row['successes']}/{row['episodes']}")
    for row in report['G3'].get('curve',[])[-2:]:
        print(f"G3  {row['frame']:,}  scale {row['scale']}  nominal {row['nominal_successes']}/20  mild {row['mild_successes']}/100")
    for stage in sorted(out.glob('*/episodes.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)[:1]:
        with stage.open('rb') as f:
            length=f.seek(0,2)
            f.seek(max(0,length-16384))
            lines=f.read().splitlines(keepends=True)
        recent=[line for line in lines[-2:] if line.endswith(b'\n')]
        if recent:
            row=json.loads(recent[-1])
            print(f"Latest episode  {stage.parent.name}  {row['group']}  process steps {row.get('transitions')}  terminal success {row['success']}",flush=True)
    print('Simulator log  '+str(out/'training.log'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--follow',action='store_true')
    args=p.parse_args()
    while True:
        show(args.output.resolve())
        if not args.follow:break
        time.sleep(10)
