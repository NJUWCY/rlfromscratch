"""Freeze exactly five source expert episodes; never alter the source dataset."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/collected/SAC-Mujoco/sac_Humanoid-v5_newest_100eps_min_reward_0.hdf5'
TARGET=SOURCE.parent/'humanoid_wave1_fixed5.hdf5'
META=TARGET.with_suffix('.json')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    if TARGET.exists() or META.exists():
        if not TARGET.exists() or not META.exists():
            raise RuntimeError('Incomplete frozen dataset; inspect the existing files instead of overwriting.')
        m=json.loads(META.read_text())
        if m['source_sha256']!=digest(SOURCE) or m['frozen_sha256']!=digest(TARGET):
            raise RuntimeError('Dataset hash changed; inspect before reuse.')
        print(json.dumps(m,indent=2))
        return
    with h5py.File(SOURCE,'r') as src:
        ends=np.flatnonzero(src['episode_ends'][:])
        if len(ends)<5 or ends[-1]!=len(src['actions'])-1:
            raise ValueError('Source must contain at least five complete episodes.')
        starts=np.r_[0,ends[:-1]+1]
        selected=sorted(np.random.default_rng(0).choice(len(ends),5,replace=False).tolist())
        lengths=[int(ends[i]-starts[i]+1) for i in selected]
        returns=[float(src['rewards'][starts[i]:ends[i]+1].sum(dtype=np.float64)) for i in selected]
        with h5py.File(TARGET,'x') as dst:
            for key in ['observations','next_observations','actions','rewards','terminals','timeouts','episode_ends']:
                dst.create_dataset(key,data=np.concatenate([src[key][starts[i]:ends[i]+1] for i in selected]))
            if 'infos/action_log_probs' in src:
                dst.create_dataset('infos/action_log_probs',data=np.concatenate([
                    src['infos/action_log_probs'][starts[i]:ends[i]+1] for i in selected]))
    m={'source':str(SOURCE),'source_sha256':digest(SOURCE),'selection_seed':0,
       'episode_indices':selected,'trajectory_num':5,'subsample_frequency':1,
       'episode_lengths':lengths,'episode_returns':returns,
       'frozen_path':str(TARGET),'frozen_sha256':digest(TARGET)}
    META.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(m,indent=2))

if __name__=='__main__':
    main()
