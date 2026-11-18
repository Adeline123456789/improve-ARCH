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
                    episode.update(run=directory.name,run_mode=identity['mode'],protocol_sha256=identity['protocol_sha256'])
                    dest.write(json.dumps(episode,separators=(',',':'))+'\n')
                    groups.setdefault(episode['suite'],[]).append(episode)
            for suite,episodes in groups.items():
                count=len(episodes)
                row=dict(run=directory.name,mode=identity['mode'],group=episodes[0]['group'],
                         seed=identity['seed'],suite=suite,checkpoint=identity['checkpoint_sha256'],
                         run_complete=completed,recorded=count,successes=sum(e['success'] for e in episodes),
                         valid=sum(e['valid'] for e in episodes) if 'valid' in episodes[0] else None)
                row['success_rate']=row['successes']/count
                for key in ['final_lateral_error_m','final_depth_error_m','final_rotation_error_rad','tracking_residual_rms','action_saturation_rate']:
                    row['median_'+key]=statistics.median(e[key] for e in episodes) if key in episodes[0] else None
                summary.append(row)
    merged.replace(out/'episodes.jsonl')
    with (out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
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
