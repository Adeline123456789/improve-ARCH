"""Finish the authorized experiment after screening, under a single wall budget."""
import hashlib,json,os,shlex,subprocess,time,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/arch_minexp';PYTHON=str(ROOT/'.venv-arch/bin/python')
state_path=OUT/'background_status.json'
# Conservative wall deadline includes preflight: six hours from first native probe.
started=(OUT/'logs/r0_native.log').stat().st_mtime
DEADLINE=started+6*3600
state={'status':'waiting_for_screen','pid':os.getpid(),'deadline_unix':DEADLINE,'deadline_local':datetime.fromtimestamp(DEADLINE,ZoneInfo('Asia/Shanghai')).isoformat(),'decision':'单种子筛查正在后台运行；最终测试尚未开始','completed_commands':[]}
def save():
    state['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n');tmp.replace(state_path)
def run(command,name,reserve=0):
    remaining=DEADLINE-time.time()-reserve
    if remaining<=0:raise TimeoutError('Six-hour wall budget exhausted')
    log=OUT/'logs'/name
    with (OUT/'commands.txt').open('a') as f:f.write(shlex.join(command)+' > '+str(log)+' 2>&1\n')
    with log.open('w') as f:
        child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        state.update(status='running',child_pid=child.pid,command=command,log=str(log));save()
        try:code=child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise TimeoutError('Wall budget reached; periodic checkpoints preserved')
        if code:raise RuntimeError(f'{name} exited {code}; inspect log')
    state['completed_commands'].append(command);save()
def records(job):return json.loads((OUT/(job+'_job.json')).read_text())['runs']
def val_successes(rec):
    return sum(json.loads(l)['success'] for l in (Path(rec['validation'])/'episodes.jsonl').read_text().splitlines() if json.loads(l)['suite']=='E0')
def run_batch(phase,seeds,frames):
    command=[PYTHON,str(ROOT/'experiments/arch_min_insert/batch.py'),'--phase',phase,'--seeds',*[str(s) for s in seeds],
             '--transitions',str(frames),'--num_envs','512','--wall_budget_s',str(max(1,DEADLINE-time.time()-1200))]
    run(command,phase+'_driver.log',reserve=1000)
try:
    save()
    while True:
        screen=json.loads((OUT/'screen_job.json').read_text())
        if screen['status']=='completed':break
        if screen['status']=='failed':raise RuntimeError('Screening failed: '+screen.get('error',''))
        if time.time()>DEADLINE:raise TimeoutError('Screening exceeded total wall budget')
        time.sleep(20)
    selected=records('screen')
    speed=min(r['actual_transitions']/json.loads((Path(r['train_dir'])/'completion.json').read_text())['wall_time_s'] for r in selected)
    remaining=DEADLINE-time.time();state['measured_training_transitions_per_s']=speed
    if any(val_successes(r)>0 for r in selected):
        # Full 45M confirmation does not fit measured budget. Predeclared common snapshot.
        estimate=6*1996800/speed*1.15+1200
        if remaining>estimate:
            state['decision']='基础学习信号存在；在预算内新增 seeds 1、2，各组仍为 1,996,800 transitions。未达到建议每种子 500 万步，按缩短预算确认报告。';save()
            run_batch('confirm',[1,2],1996800);selected+=records('confirm')
        else:state['decision']='单种子已完成；剩余预算不足以完成六个等预算新增种子，保留共同快照并做最终测试。';save()
    else:
        # No learner has basic ability. Avoid claiming a negative algorithm result;
        # retry equal longer budgets from scratch, because native checkpoints omit live simulator/RNN state.
        estimate=3*4966400/speed*1.15+600
        if remaining>estimate:
            state['decision']='三组名义验证均未成功；各组从头执行共同 4,966,400 transitions 的有限延长筛查。保留旧 run，未冒称恢复完整仿真状态。';save()
            run_batch('extended',[0],4966400);selected=records('extended')
        else:state['decision']='三组名义验证未学会，且剩余预算不足以公平延长；结论标记未完成有效检验。';save()
    final=[]
    for rec in selected:
        target=OUT/f"test_{rec['group']}_s{rec['seed']}_{rec['actual_transitions']}"
        item={'group':rec['group'],'seed':rec['seed'],'transitions':rec['actual_transitions'],'checkpoint':rec['checkpoint'],
              'checkpoint_sha256':rec['checkpoint_sha256'],'test_dir':str(target),'train_dir':rec['train_dir']}
        final.append(item);(OUT/'final_selection.json').write_text(json.dumps(final,indent=2)+'\n')
        command=['bash',str(ROOT/'experiments/arch_min_insert/run_eval.sh'),'--group',rec['group'],'--seed',str(rec['seed']),
           '--num_envs','100','--checkpoint',rec['checkpoint'],'--manifest',str(OUT/'evaluation_manifest.jsonl'),'--headless','--output',str(target)]
        run(command,target.name+'.log')
    state['status']='experiments_completed';save()
except BaseException:
    state['status']='incomplete';state['error']=traceback.format_exc();save()
finally:
    # Report partial truth even after a failure; never synthesize missing metrics.
    try:
        with (OUT/'logs/summarize.log').open('w') as f:
            subprocess.run([PYTHON,str(ROOT/'experiments/arch_min_insert/summarize_results.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        with (OUT/'logs/publish_final.log').open('w') as f:
            subprocess.run([PYTHON,str(ROOT/'experiments/arch_min_insert/publish.py'),*(['--include_code'] if not (OUT/'publication.json').exists() else []),'--message','Record ARCH INSERT experiment outcomes and evidence'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        state['publication']=json.loads((OUT/'publication.json').read_text());state['publication_status']='uploaded'
        if state['status']=='experiments_completed':state['status']='completed'
    except BaseException:state['publication_status']='failed';state['finalization_error']=traceback.format_exc()
    save()
