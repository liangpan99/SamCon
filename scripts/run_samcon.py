import os
import sys
sys.path.append(os.getcwd())

import time
import shutil
import argparse
import numpy as np

from samcon.utils.config import Config
from samcon.core.samcon import SamCon
from data_loaders.amass_motion_data import AmassMotionData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='walk')
    parser.add_argument('--num_processes', type=int, default=12)
    args = parser.parse_args()

    cfg = Config(args.cfg, base_dir='', create_dirs=True)

    """ load reference motion (only support 1 sequence)
    cfg.data_path: path to amass_bullet.pkl
    cfg.motion_seq: sequence name (you can pick one in the all_seq_names.txt)
    """
    data = AmassMotionData(
        cfg.data_path, 
        cfg.motion_seq
    )

    if cfg.auto_nIter:
        cycleTime = data.getCycleTime()
        nIter = int(cycleTime * cfg.fps_act)
        cfg.nIter = nIter

    algo = SamCon(cfg, data, args.num_processes, cfg.nIter, cfg.nSample, cfg.nSave)
    algo.sample()

    shutil.rmtree(cfg.state_dir)
