import yaml
import os
import os.path as osp
import numpy as np

from utils.tools import recreate_dirs

class Config:

    def __init__(self, cfg_id, base_dir="", create_dirs=False, cfg_dict=None):
        self.id = cfg_id
        base_dir = base_dir if base_dir else ''
        self.base_dir = os.path.expanduser(base_dir)

        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg_path =  osp.join(self.base_dir, "samcon", "cfg", f"{cfg_id}.yml")
            cfg = yaml.safe_load(open(cfg_path, 'rb'))
        self.cfg_dict = cfg

        # create dirs
        self.result_dir = osp.join(self.base_dir, "results")
        self.cfg_dir = '%s/samcon/%s' % (self.result_dir, cfg_id)
        self.output_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.info_dir = '%s/info' % self.cfg_dir
        self.state_dir =  '%s/states' % cfg['temp_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.info_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.log_dir, self.state_dir)
        
        # read parameters
        self.fps_sim = cfg['fps_sim']
        self.fps_act = cfg['fps_act']
        self.seed = cfg['seed']
        self.auto_nIter = cfg['auto_nIter']
        self.nIter = cfg['nIter']
        self.nSample = cfg['nSample']
        self.nSave = cfg['nSave']

        self.data_path = cfg['data_path']
        self.motion_seq = cfg['motion_seq']

        self.noise_type = cfg['noise_type']
        if 'joint_noises' in cfg:
            jparam = zip(*cfg['joint_noises'])
            jparam = [np.array(p) for p in jparam]
            self.values, = jparam[1:]

        self.model_file = cfg['model_file']
        self.scale = cfg['scale']
        self.height = cfg['height']
        self.self_collision = cfg['self_collision']
        if 'joint_params' in cfg:
            jparam = zip(*cfg['joint_params'])
            jparam = [np.array(p) for p in jparam]
            self.kps, self.kds, self.max_forces = jparam[1:4]
            self.char_info = {
                'kps': self.kps,
                'kds': self.kds,
                'max_forces': self.max_forces
            }
        self.joint_weights = cfg['joint_weights']
        self.end_effectors = cfg['end_effectors']
        self.cost_weights = cfg['cost_weights']

    def get(self, key, default = None):
        return self.cfg_dict.get(key, default)
