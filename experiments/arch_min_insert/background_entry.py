"""Detached bootstrap: initial publication must not prevent experiment supervision."""
import os,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/arch_minexp';PY=str(ROOT/'.venv-arch/bin/python')
try:
    subprocess.run([PY,str(Path(__file__).with_name('summarize_results.py'))],cwd=ROOT,check=True)
    subprocess.run([PY,str(Path(__file__).with_name('publish.py')),'--include_code','--message','Add bounded ARCH INSERT experiment and pretraining evidence'],cwd=ROOT,check=True)
except BaseException:
    (OUT/'logs/initial_publication_error.txt').write_text(traceback.format_exc())
os.execv(PY,[PY,str(Path(__file__).with_name('finish_background.py'))])
