import os
import sys
sys.path.append(os.getcwd())

import time
import os.path as osp
import joblib
import numpy as np
import argparse
import matplotlib.pyplot as plt

from samcon.utils.config import Config
from envs.humanoid_tracking_env import HumanoidTrackingEnv
from data_loaders.amass_motion_data import AmassMotionData
from utils.bullet_utils import isKeyTriggered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cartwheel')
    parser.add_argument('--file', type=str, default='cartwheel_ACCAD_Female1Gestures_c3d_D6- CartWheel_poses.pkl')
    args = parser.parse_args()

    cfg = Config(args.cfg, base_dir='', create_dirs=False)
    env = HumanoidTrackingEnv(cfg, True)

    # load data
    m_path = osp.join(cfg.output_dir, args.file)
    m_data = joblib.load(m_path)
    i_path = osp.join(cfg.info_dir, args.file)
    i_data = joblib.load(i_path)

    nIter = i_data['total_cost'].shape[0]
    nSample = i_data['total_cost'].shape[1]
    nSave = i_data['elite_inds'].shape[1]
    
    # save cost distribution pictures
    print(f'***** saving cost distribution pictures to {cfg.info_dir}')
    best_path_my_sys = i_data['best_path_my_sys']
    pictures = ['total_cost', 'pose_cost', 'root_cost', 'ee_cost', 'balance_cost', 'com_cost']
    for wid, pic in enumerate(pictures):
        key = pic
        line_chart_x = []
        line_chart_y = []
        plt.figure(wid, figsize=(19.2, 10.8))
        for i in range(nIter):
            x_all = np.full((nSample), i + 1)
            x_elite = np.full((nSave), i + 1 + 0.3)
            elite_inds = i_data['elite_inds'][i]
            plt.scatter(x=x_all, y=i_data[key][i], c='blue', marker='+')
            plt.scatter(x=x_elite, y=i_data[key][i][elite_inds], c='red', marker='x')

            line_chart_x.append(i + 1 + 0.3)
            line_chart_y.append(i_data[key][i][elite_inds[best_path_my_sys[i]]])
        plt.plot(line_chart_x, line_chart_y, c='lime', linewidth=3)
        plt.title(key)
        plt.savefig(osp.join(cfg.info_dir, f'{args.file[:-4]}_{key}.png'))

    # visualize reconstructed motion
    vis_target_state = True
    debug = False
    for k, v in m_data.items():
        print(f'***** visualizing reconstructed motion of seq: {k}')
        print(f'***** vis_target_state: {vis_target_state} output debug info {debug}')

        path = v['path_0'] # visualize path with min total cost

        # simulation to reproduce the trajectory
        t0_state = path['t0_state']
        seq_len = path['action'].shape[0]
        env._sim_agent.set_state(t0_state)

        simulated_states = []
        costs = []

        for i in range(seq_len):
            target_state = path['target_state'][i]
            action = path['action'][i]
            env._kin_agent.set_state(target_state)
            simulated_state, cost, _ = env.step(action)
            simulated_states.append(simulated_state)
            costs.append(cost)

        # vis
        # key Board Control: 
        # [space] pause/continue 
        # [q] quit
        enable_animation = True
        n_iter = 0
        while(env._pb_client.isConnected()):
            keys = env._pb_client.getKeyboardEvents()
            if isKeyTriggered(keys, ' '): 
                enable_animation = not enable_animation
            if isKeyTriggered(keys, 'q'): 
                break

            if enable_animation:
                target_state = path['target_state'][n_iter]
                simulated_state_of_traj = path['simulated_state'][n_iter]
                simulated_state = simulated_states[n_iter]

                env._sim_agent.set_state(simulated_state)
                if vis_target_state:
                    env._kin_agent.set_state(target_state)
                else:
                    env._kin_agent.set_state(simulated_state_of_traj)

                # compute difference
                if debug:
                    diff_ss = simulated_state - simulated_state_of_traj
                    diff_ss_ = np.dot(diff_ss, diff_ss)
                    print(f'curr iter {n_iter + 1} diff of simulated state {diff_ss_}')

                n_iter += 1
                if n_iter == seq_len:
                    n_iter = 0
            
                time.sleep(1/10)
