"""Run four DAC-derived parameter sets at two fixed temperatures with HyPE's rollout schedule."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from parallel_search import SearchSpec, build_command, run_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_suffix('.json').read_text())
    if not 1 <= args.limit <= len(data['configs']):
        parser.error(f'--limit must be between 1 and {len(data["configs"])}')
    configs = data['configs'][:args.limit]
    spec = SearchSpec(data['run_script'], {}, data['fixed_overrides'], data['sweep_name'])
    expert_text = next(x.split('=', 1)[1] for x in spec.fixed_overrides
                       if x.lstrip('+').startswith('expert_dataset.data_path='))
    expert = Path(json.loads(expert_text))
    if not expert.is_file():
        parser.error(f'Expert data missing: {expert}')
    destination = ROOT / 'outputs' / spec.sweep_name / datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    tasks = []
    for index, cfg in enumerate(configs, 1):
        run_dir = str(destination / f'run_{index:02d}_{data["candidate_names"][index-1]}')
        tasks.append((index, build_command(spec, cfg, run_dir, []), run_dir))
    gpu_slots = [0, 1] * 4
    workers = min(len(tasks), len(gpu_slots))
    print(f'Will run: {len(tasks)}; workers: {workers}; GPU slots: {gpu_slots}', flush=True)
    print(f'Sweep dir: {destination}', flush=True)
    print('Preserves HyPE fresh-rollout schedule; this is shared-parameter alignment, not complete update-frequency alignment.', flush=True)
    if args.dry_run:
        for index, cmd, _ in tasks:
            print(f'[{index:02d}] ' + ' '.join(cmd))
        return

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        parser.error('Two visible GPUs required; training was not started.')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'manifest.json').write_text(json.dumps({
        'created_at': datetime.now().astimezone().isoformat(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'spec': data, 'selected_configs': configs, 'alignment_notes': data['alignment_notes'],
        'gpu_slots': gpu_slots, 'swanlab_mode': 'offline',
    }, indent=2) + '\n')
    (destination / 'hype_source.py').write_bytes((ROOT / 'algorithm/hype.py').read_bytes())
    gpu_queue = Queue()
    for gpu in gpu_slots:
        gpu_queue.put(str(gpu))
    counter = {'done': 0, 'lock': threading.Lock()}
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, index, len(tasks), cmd, run_dir,
                               gpu_queue, started, counter, workers, 48 * 3600)
                   for index, cmd, run_dir in tasks]
        for future in as_completed(futures):
            index, rc, elapsed = future.result()
            results.append({'run': index, 'exit_code': rc, 'seconds': elapsed})
            (destination / 'results.json').write_text(json.dumps(
                sorted(results, key=lambda r: r['run']), indent=2) + '\n')
    failed = sum(result['exit_code'] != 0 for result in results)
    print(f'Finished {len(results)} runs; failures: {failed}; output: {destination}', flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
