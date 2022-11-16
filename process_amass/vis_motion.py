import os
import sys
sys.path.append(os.getcwd())

import os.path as osp
import joblib
import pybullet as p
import pybullet_data

from utils.bullet_utils import isKeyTriggered
from utils.tools import print_all_seq_names

if __name__ == "__main__":
    dir = "./data/motion"
    in_file = "amass_bullet.pkl"
    data = joblib.load(osp.join(dir, in_file))

    """print all seq names"""
    seq_names = list(data.keys())
    save_path = osp.join(dir, 'all_seq_names.txt')
    print_all_seq_names(seq_names, save_path)

    """ load bullet env """
    viz = True
    mode = p.GUI if viz else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    plane_id = p.loadURDF('plane_implicit.urdf', [0, 0, 0], useMaximalCoordinates=True)
    body_id = p.loadURDF(
        fileName='./data/character/humanoid.urdf', 
        basePosition=[0, 0, 0], 
        globalScaling=1.0, 
        useFixedBase=False, 
        flags=p.URDF_MAINTAIN_LINK_ORDER
    )
    movable_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]

    print("please input your requested sequence:")
    seq_name = input()
    slider = p.addUserDebugParameter('frame', 0, data[seq_name]['pose'].shape[0]-1)
    while(p.isConnected()):
        keys = p.getKeyboardEvents()
        if isKeyTriggered(keys, 'q'): # input sequence
            print("please input your requested sequence:")
            seq_name = input()
            p.removeAllUserParameters()
            slider = p.addUserDebugParameter('frame', 0, data[seq_name]['pose'].shape[0]-1)
        
        n = int(p.readUserDebugParameter(slider))
        state = data[seq_name]['pose'][n]
        base_pos = state[0:3]
        base_orn = state[3:7]
        joints_orn = state[7:].reshape(-1, 4)
        p.resetBasePositionAndOrientation(body_id, base_pos, base_orn)
        p.resetJointStatesMultiDof(body_id, movable_indices, joints_orn)
