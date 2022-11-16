import os.path as osp
import numpy as np
import treelib
import joblib
import datetime

class Trajectory:
    """ 
    1. store sample for each iter
    2. select elite sample for each iter
    3. find path and save
    """

    def __init__(self, nIter, nSample, nSave):

        self._tree = treelib.Tree()
        self._tree.create_node(f'uid_0', 0) # root node

        self._nIter = nIter
        self._nSample = nSample
        self._nSave = nSave

        self.prev_uid = []
        self.curr_uid = []
        self.target_state = []
        self.simulated_state = []
        self.action = []
        self.cost = []
        self.info = []

        self.elite_inds = [] # store indices that indicate which samples are elite

        for _ in range(nIter + 1):
            self.prev_uid.append([])
            self.curr_uid.append([])
            self.target_state.append([])
            self.simulated_state.append([])
            self.action.append([])
            self.cost.append([])
            self.info.append([])

            self.elite_inds.append([])

    def push(self, i, data):
        d1, d2, d3, d4, d5, d6, d7 = data
        self.prev_uid[i].append(d1)
        self.curr_uid[i].append(d2)
        self.target_state[i].append(d3)
        self.simulated_state[i].append(d4)
        self.action[i].append(d5)
        self.cost[i].append(d6)
        self.info[i].append(d7)

        if i > 0:
            self._tree.create_node(f'uid_{d2}', d2, parent=d1)

    def get_elite_inds(self, i):
        return np.array(self.elite_inds[i], dtype=np.int)

    # ===========================================================
    # get all samples (num = nSample)
    def get_prev_uid(self, i):
        return np.array(self.prev_uid[i], dtype=np.int)
    
    def get_curr_uid(self, i):
        return np.array(self.curr_uid[i], dtype=np.int)
    
    def get_target_state(self, i):
        return np.array(self.target_state[i], dtype=np.float64)
    
    def get_simulated_state(self, i):
        return np.array(self.simulated_state[i], dtype=np.float64)
    
    def get_action(self, i):
        return np.array(self.action[i], dtype=np.float64)

    def get_cost(self, i):
        return np.array(self.cost[i], dtype=np.float64)
    
    def get_info(self, i):
        return np.array(self.info[i], dtype=np.float64)
    # ===========================================================
   
    # ===========================================================
    # get elite samples (num = nSave)
    def get_prev_uid_elite(self, i):
        return self.get_prev_uid(i)[self.get_elite_inds(i)]
    
    def get_curr_uid_elite(self, i):
        return self.get_curr_uid(i)[self.get_elite_inds(i)]
    
    def get_target_state_elite(self, i):
        return self.get_target_state(i)[self.get_elite_inds(i)]
    
    def get_simulated_state_elite(self, i):
        return self.get_simulated_state(i)[self.get_elite_inds(i)]
    
    def get_action_elite(self, i):
        return self.get_action(i)[self.get_elite_inds(i)]

    def get_cost_elite(self, i):
        return self.get_cost(i)[self.get_elite_inds(i)]

    def get_info_elite(self, i):
        return self.get_info(i)[self.get_elite_inds(i)]
    # ===========================================================

    def select(self, i, mode='greedy'):
        assert len(self.get_cost(i)) == self._nSample
        if mode == 'greedy':
            order = self.get_cost(i).argsort() # sort from small to large
            self.elite_inds[i] = order[:self._nSave]
        elif mode == 'diversity':
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def find_path_and_save(self, cfg):
        # 1. search nSave paths
        # data struct
        # [ ...... path0 ......]
        # [ ...... path1 ......]
        # [ ...... path2 ......]
        #           ...
        paths = []
        end_points = self.get_curr_uid_elite(-1)
        for ep in end_points:
            p = [uid for uid in self._tree.rsearch(ep)][::-1]
            paths.append(p)
        paths = np.array(paths, dtype=np.int)
        num_path = paths.shape[0]

        paths_my_sys = paths.copy()
        for i in range(1, self._nIter + 1):
            candidates = self.get_curr_uid_elite(i)
            idx_my_sys = [np.where(candidates == paths[j, i])[0] for j in range(num_path)]
            paths_my_sys[:, i] = idx_my_sys

        # 2. calculate the total cost of each path
        # data struct
        # [total cost of path0, total cost of path1, total cost of path2, ...]
        total_cost = np.zeros(num_path)
        for i in range(1, self._nIter + 1):
            candidates = self.get_cost_elite(i)
            total_cost += candidates[paths_my_sys[:, i]]
        
        # 3. get order from small to large
        order = total_cost.argsort()

        # 4. save
        out = {
            cfg.motion_seq: {}
        }
        
        for i in range(num_path):
            single_path = paths_my_sys[order[i]]
            data = {
                'target_state': [],
                'simulated_state': [],
                'action': [],
                'cost': [],
            }
            for t in range(1, self._nIter + 1):
                dummy = single_path[t]
                data['target_state'].append(self.get_target_state_elite(t)[dummy])
                data['simulated_state'].append(self.get_simulated_state_elite(t)[dummy])
                data['action'].append(self.get_action_elite(t)[dummy])
                data['cost'].append(self.get_cost_elite(t)[dummy])
            
            data = {k: np.vstack(v) for k, v in data.items()}
            data['total_cost'] = np.sum(data['cost'])
            data['t0_state'] = self.get_target_state_elite(0)[0]

            out[cfg.motion_seq][f'path_{i}'] = data

        # 5. save cost info
        info = {
            'pose_cost': [],
            'root_cost': [],
            'ee_cost': [],
            'balance_cost': [],
            'com_cost': [],
            'total_cost': [],
            'elite_inds': [],
        }
        for t in range(1, self._nIter + 1):
            dummy = self.get_info(t) # (2000, 4)
            info['pose_cost'].append(dummy[:, 0])
            info['root_cost'].append(dummy[:, 1])
            info['ee_cost'].append(dummy[:, 2])
            info['balance_cost'].append(dummy[:, 3])
            info['com_cost'].append(dummy[:, 4])
            info['total_cost'].append(self.get_cost(t))
            info['elite_inds'].append(self.get_elite_inds(t))

        info = {k: np.vstack(v) for k, v in info.items()}
        info['best_path_my_sys'] = paths_my_sys[order[0]][1:]

        time_flag = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        joblib.dump(out, osp.join(cfg.output_dir, f'{cfg.id}_{time_flag}.pkl'))
        joblib.dump(info, osp.join(cfg.info_dir, f'{cfg.id}_{time_flag}.pkl'))

