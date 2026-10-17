"""Three DAC bases x discriminator LR/optimizer; reuse three completed controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
from queue import Queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from parallel_search import SearchSpec, build_command, iter_configs, run_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_suffix('.json').read_text())
    spec = SearchSpec(data['run_script'], data['search_space'],
                      data['fixed_overrides'], data['sweep_name'])
    trials = []
    for base_name, base in data['base_configs'].items():
        for choice in iter_configs(spec):
            # All three Adam / 1e-4 controls already completed at seed=0.
            if (choice['algorithm.discriminator.discriminator_optimizer'] == 'Adam'
                    and choice['algorithm.discriminator.disc_lr'] == 1e-4):
                continue
            trials.append((base_name, {**base, **choice}))

    destination = ROOT / 'outputs' / spec.sweep_name / datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    tasks = []
    for index, (base_name, cfg) in enumerate(trials, 1):
        optimizer = cfg['algorithm.discriminator.discriminator_optimizer']
        lr = cfg['algorithm.discriminator.disc_lr']
        run_dir = str(destination / f'run_{index:02d}_{base_name}_{optimizer}_lr-{lr:g}')
        tasks.append((index, build_command(spec, cfg, run_dir, []), run_dir))
    print(f'Will run: {len(tasks)}; workers: 8; GPU slots: 0,1,0,1,0,1,0,1', flush=True)
    print(f'Sweep dir: {destination}', flush=True)
    if args.dry_run:
        for index, cmd, _ in tasks:
            print(f'[{index:02d}] ' + ' '.join(cmd))
        return

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        parser.error('Two visible GPUs required; training was not started.')
    expert_path = next(x.split('=', 1)[1] for x in spec.fixed_overrides
                       if x.startswith('expert_dataset.data_path='))
    if not Path(expert_path).is_file():
        parser.error(f'Expert data missing: {expert_path}')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'manifest.json').write_text(json.dumps({
        'spec': data, 'trials': [{'base': name, **cfg} for name, cfg in trials],
        'gpu_slots': [0, 1, 0, 1, 0, 1, 0, 1],
    }, indent=2) + '\n')
    gpu_queue = Queue()
    for gpu in [0, 1, 0, 1, 0, 1, 0, 1]:
        gpu_queue.put(str(gpu))
    counter = {'done': 0, 'lock': threading.Lock()}
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_one, index, len(tasks), cmd, run_dir,
                               gpu_queue, started, counter, 8, 48 * 3600)
                   for index, cmd, run_dir in tasks]
        for future in as_completed(futures):
            index, rc, elapsed = future.result()
            results.append({'run': index, 'exit_code': rc, 'seconds': elapsed})
    (destination / 'results.json').write_text(json.dumps(sorted(results, key=lambda r: r['run']), indent=2) + '\n')
    failed = sum(result['exit_code'] != 0 for result in results)
    print(f'Finished {len(results)} runs; failures: {failed}; output: {destination}', flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
