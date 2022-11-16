import os
import os.path as osp
import time
import multiprocessing
import numpy as np
import datetime
from scipy.spatial.transform import Rotation as sRot

from envs.humanoid_tracking_env import HumanoidTrackingEnv
from samcon.core.trajectory import Trajectory
from samcon.core.unique_id import UniqueId
from utils.tools import get_eta_str
from utils.logger import create_logger

def sample_worker(pid, pipe, cfg, viz, num_substep):
    """ run physics simulator in the subprocess """

    # create env
    env = HumanoidTrackingEnv(cfg, viz)

    while True:
        try:
            cmd, data = pipe.recv()
        except KeyboardInterrupt:
            print(f"subprocess {pid} worker: got KeyboardInterrupt")
            break
        
        if cmd == "sim":
            curr_iter, prev_uids, curr_uids, target_state, actions = data
            res = []
            for i in range(len(prev_uids)):
                for j in range(num_substep):
                    prev_uid = prev_uids[i]
                    curr_uid = next(curr_uids) # get a new unique_id to mark this simulation
                    act = actions[i * num_substep + j]
                    env.restore_pb_state(curr_iter - 1, prev_uid)
                    env._kin_agent.set_state(target_state)
                    simulated_state, cost, info = env.step(act)
                    env.save_pb_state(curr_iter, curr_uid)
                    res.append([prev_uid, curr_uid, target_state, simulated_state, act, cost, info])
            pipe.send(res)
        elif cmd == "init":
            curr_iter, curr_uid, init_state = data
            env._sim_agent.set_state(init_state)
            env.save_pb_state(curr_iter, curr_uid)
            pipe.send([f"subprocess {pid} init"])
        elif cmd == "close":
            pipe.send([f"subprocess {pid} close"])
            break
        else:
            raise RuntimeError(f"Got unrecognized cmd {cmd}")
    pipe.close()
    
class SamCon():
    
    def __init__(self, cfg, data_loader, num_processes, nIter, nSample, nSave):
        np.random.seed(cfg.seed)
        self._cfg = cfg
        self._data_loader = data_loader

        """ sample config """
        self._sample_timestep = 1.0 / cfg.fps_act
        self._nIter = nIter
        self._nSample = nSample
        self._nSave = nSave
        if self._nSample % self._nSave != 0:
            raise Exception('nSample should be a multiples of nSave')
        self._num_substep = self._nSample // self._nSave

        """ multiprocessing """
        self._num_processes = num_processes
        self._processes = []
        self._parent_pipes = []
        self._child_pipes = []
        for _ in range(num_processes):
            parent, child = multiprocessing.Pipe()
            self._parent_pipes.append(parent)
            self._child_pipes.append(child)
        for pid in range(num_processes):
            process = multiprocessing.Process(target=sample_worker, args=(pid, self._child_pipes[pid], cfg, False, self._num_substep))
            process.start()
            self._processes.append(process)

        # pre-compute job chunks for subprocesses
        self._jobs = np.array_split(np.arange(0, self._nSave), self._num_processes)
        self._jobs_action = []
        for x in self._jobs:
            start_id = x[0] * self._num_substep
            end_id = (x[-1] + 1) * self._num_substep
            self._jobs_action.append(np.arange(start_id, end_id))

        """ uid manager """
        self._unique_id = UniqueId()

        """ trajectory """
        self._traj = Trajectory(nIter, nSample, nSave)

        """ logger """
        self._logger = create_logger(os.path.join(cfg.log_dir, "log_sample.txt"))

    def get_batch_action(self, state):

        # noise
        if self._cfg.noise_type == 'gaussian':
            vars = self._cfg.values
            num_eulers = len(vars)
            noise = np.random.multivariate_normal( # (2000, 51)
                mean=np.zeros(num_eulers),
                cov=np.diag(vars),
                size=(self._nSample)
            )
        elif self._cfg.noise_type == 'uniform':
            limits = self._cfg.values
            num_eulers = len(limits)
            noise = np.random.random((self._nSample, num_eulers)) # sample from [0, 1) uniform distribution
            for i in range(num_eulers): # transform to [-limit, limit)
                noise[:, i] *= 2 * limits[i]
                noise[:, i] += -1 * limits[i]
        else:
            raise NotImplementedError

        # quaternion 2 euler
        state_quat = state[7:75] # (68, )
        state_euler = sRot.from_quat(state_quat.reshape(-1, 4)).as_euler("XYZ", degrees=False).reshape(-1) # (51, )

        batch_action_euler = np.zeros((self._nSample, len(state_euler))) # (2000, 51)
        batch_action_euler[:] = state_euler.copy()
        batch_action_euler += noise

        while np.any(batch_action_euler > np.pi):
            batch_action_euler[batch_action_euler > np.pi] -= 2 * np.pi
        while np.any(batch_action_euler < -np.pi):
            batch_action_euler[batch_action_euler < -np.pi] += 2 * np.pi

        # euler 2 quaternion
        batch_action_quat = sRot.from_euler("XYZ", batch_action_euler.reshape(-1, 3), degrees=False)
        batch_action_quat = batch_action_quat.as_quat().reshape(self._nSample, -1) # (2000, 68)

        # shuffle
        np.random.shuffle(batch_action_quat)

        return batch_action_quat

    def sample(self):
    
        # setup sampling start point
        startp_iter = 0
        startp_uid = next(self._unique_id.get(1))
        startp_state = self._data_loader.getSpecTimeState(t=0.0)
        os.makedirs(osp.join(self._cfg.state_dir, f'{startp_iter}'), exist_ok=True)
        self._parent_pipes[0].send(["init", [startp_iter, startp_uid, startp_state]])
        self._parent_pipes[0].recv()
        for _ in range(self._nSample):
            self._traj.push(startp_iter, [-1, startp_uid, startp_state, None, None, 0.0, None])
        self._traj.select(startp_iter)

        # begin sampling
        for t in range(1, self._nIter + 1):
            t0 = time.time()

            target_state = self._data_loader.getSpecTimeState(t * self._sample_timestep)
            batch_action = self.get_batch_action(target_state)
            curr_iter = t
            prev_uids = self._traj.get_curr_uid_elite(t-1)
            curr_uids = [self._unique_id.get(len(j) * self._num_substep) for j in self._jobs]
            os.makedirs(osp.join(self._cfg.state_dir, f'{curr_iter}'), exist_ok=True)

            # send to subprocess
            for i, parentPipe in enumerate(self._parent_pipes):
                data = [curr_iter, prev_uids[self._jobs[i]], curr_uids[i], target_state, batch_action[self._jobs_action[i]]]
                parentPipe.send(["sim", data])
            results = []

            # receive data
            for i, parentPipe in enumerate(self._parent_pipes):
                results += parentPipe.recv()

            # add into trajectory
            for res in results:
                self._traj.push(curr_iter, res)
            
            # select elite samples
            self._traj.select(curr_iter, 'greedy')

            t1 = time.time()
            self._logger.info(
                'cfg: {} | iter {}/{} | T_i {:.2f}s | ETA {}'
                .format(self._cfg.id, t, self._nIter, t1 - t0, get_eta_str(t, self._nIter, t1 - t0))
            )

        # end sampling and find the final trajectory
        self._traj.find_path_and_save(self._cfg)

        # close subprocesses
        for i, parentPipe in enumerate(self._parent_pipes):
            parentPipe.send(["close", None])

        self._logger.info(f'sampling done {datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
