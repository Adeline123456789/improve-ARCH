"""Run the formal comparison when eligible, and refresh the evidence on completion."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--protocol',required=True,type=Path)
p.add_argument('--publish',action='store_true',help='Push to the authorized experiment branch')
a=p.parse_args()
out=a.output.resolve()
def state():
    return json.loads((out/'gate_report.json').read_text())
def execute(name,*args):
    subprocess.run([sys.executable,'-u',str(HERE/name),*map(str,args)],check=True)
if state()['G3']['status']=='PASS' and state()['G4']['status']!='PASS':
    execute('formal_v2.py','--output',out,'--protocol',a.protocol.resolve())
execute('summarize_v2.py',out)
execute('report_v2.py',out)
if a.publish:
    r=state()
    message='Record ARCH v2 evidence for G2 '+r['G2']['status']+', G3 '+r['G3']['status']+' and G4 '+r['G4']['status']
    execute('publish_v2.py',out,'--message',message)
