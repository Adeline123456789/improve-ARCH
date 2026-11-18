"""Collect recorded episodes and geometry by run, keeping incomplete runs identifiable."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

def collect(out):
    summary=[]
    merged=out/'episodes.jsonl.tmp'
    with merged.open('w') as dest:
        for path in sorted(out.glob('*/episodes.jsonl')):
            directory=path.parent
            identity=json.loads((directory/'identity.json').read_text())
            completed=(directory/'completion.json').is_file()
            groups={}
            with path.open() as stream:
                for line in stream:
                    if not line.endswith('\n'):
                        assert not completed
                        break
                    episode=json.loads(line)
                    if '--response_probe' in identity['command'] or '--contact_pair' in identity['command']:
                        episode['policy_kind']='physical control response probe'
                    episode.update(run=directory.name,run_mode=identity['mode'],protocol_sha256=identity['protocol_sha256'])
                    dest.write(json.dumps(episode,separators=(',',':'))+'\n')
                    groups.setdefault(episode['suite'],[]).append(episode)
            for suite,episodes in groups.items():
                count=len(episodes)
                row=dict(run=directory.name,mode=identity['mode'],group=episodes[0]['group'],
                         policy_kind=episodes[0]['policy_kind'],seed=identity['seed'],suite=suite,checkpoint=identity['checkpoint_sha256'],
                         run_complete=completed,recorded=count,successes=sum(e['success'] for e in episodes),
                         valid=sum(e['valid'] for e in episodes) if 'valid' in episodes[0] else None)
                row['success_rate']=row['successes']/count
                for key in ['final_lateral_error_m','final_depth_error_m','final_rotation_error_rad','tracking_residual_rms','action_saturation_rate']:
                    row['median_'+key]=statistics.median(e[key] for e in episodes) if key in episodes[0] else None
                summary.append(row)
    merged.replace(out/'episodes.jsonl')
    with (out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    curve=[]
    from trajectory_diagnosis import diagnose
    for directory in out.glob('g2_eval_*'):
        if not (directory/'completion.json').is_file():
            continue
        measured=diagnose(directory)
        (directory/'trajectory_diagnosis.json').write_text(json.dumps(measured,indent=2)+'\n')
        curve.append(dict(frame=int(directory.name.split('_')[-1]),**measured))
    if curve:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        curve.sort(key=lambda row:row['frame'])
        diagrams=out/'diagnostics'
        diagrams.mkdir(exist_ok=True)
        (diagrams/'g2_curve.json').write_text(json.dumps(curve,indent=2)+'\n')
        fig,axes=plt.subplots(1,3,figsize=(11,3.4))
        xs=[r['frame']/1e6 for r in curve]
        for ax,key,label in zip(axes,['successes','median_final_lateral_mm','median_final_depth_error_mm'],
                                ['Terminal successes / 20','Median lateral error (mm)','Median signed depth error (mm)']):
            ax.plot(xs,[r[key] for r in curve],marker='o',color='#24617b')
            ax.set(xlabel='Training steps (millions)',ylabel=label)
            ax.grid(axis='y',alpha=.25)
        axes[0].set_ylim(-1,21)
        axes[0].axhline(19,ls=':',c='#635667')
        axes[1].axhline(.75,ls=':',c='#635667')
        axes[2].axhspan(-1,1,color='#635667',alpha=.15)
        fig.tight_layout()
        fig.savefig(diagrams/'g2_curve.png',dpi=150)
        plt.close(fig)
    manifest_info={}
    for path in sorted((out/'manifests').glob('*.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        physical={json.dumps([r[k] for k in ['board_true','socket_estimated','X_true','X_hat','ee_start']]) for r in rows}
        manifest_info[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':len(rows),
                                  'unique_ids':len({r['case_id'] for r in rows}),'unique_physical_cases':len(physical)}
    (out/'manifests'/'inventory.json').write_text(json.dumps(manifest_info,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    collect(parser.parse_args().output)
