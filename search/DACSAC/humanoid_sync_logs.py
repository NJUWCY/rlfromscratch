"""Print one stable-ID sync command per completed local run; --upload executes.

Use the existing SwanLab login/API-key environment. No credentials in arguments.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sweep',type=Path)
    p.add_argument('--upload',action='store_true')
    p.add_argument('--include-incomplete',action='store_true',
                   help='Include stopped/crashed runs without finish records; use only after they stop.')
    args=p.parse_args()
    if not args.sweep.is_dir():
        p.error('sweep directory does not exist')
    from swanlab.sdk.internal.core_python.store import DataStoreReader
    from swanlab.proto.swanlab.record.v1.record_pb2 import Record
    cli=Path(sys.executable).parent/'swanlab'
    failures=[]
    for path in sorted(args.sweep.rglob('*.swanlab')):
        reader=DataStoreReader(); reader.open(path)
        start=Record.FromString(reader.scan())
        finished=False
        for raw in reader:
            if Record.FromString(raw).HasField('finish'):
                finished=True
        reader.close()
        if not finished and not args.include_incomplete:
            print(f'Skipping run without finish event (may still be running): {path}',file=sys.stderr)
            continue
        run_id=start.start.id
        if not run_id:
            print(f'Missing recorded run ID; inspect manually: {path}',file=sys.stderr)
            failures.append(str(path))
            continue
        cmd=[str(cli),'sync',str(path.parent),'--id',run_id]
        print(shlex.join(cmd),flush=True)
        if args.upload:
            if subprocess.call(cmd)!=0:
                failures.append(str(path))
    if args.upload:
        (args.sweep/'sync_status.json').write_text(json.dumps({'failed':failures},indent=2)+'\n')
    if failures:
        raise SystemExit(1)

if __name__=='__main__':
    main()
