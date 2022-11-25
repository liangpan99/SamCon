import os
import sys
sys.path.append(os.getcwd())

import os.path as osp
import numpy as np
import joblib
import torch 
from tqdm import tqdm
import pybullet as p
import pybullet_data

from utils.torch_geometry_transforms import (
    angle_axis_to_quaternion
)

# extract base orientation and 17 movable Humanoid joints from 24 SMPL joints
joints_to_use = np.array(
    [0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 15, 13, 16, 18, 14, 17, 19]
)

def smpl_to_bullet(pose, trans):
    '''
        Expect pose to be batch_size x 72
        trans to be batch_size x 3
    '''
    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)
    
    pose = pose.reshape(-1, 24, 3)[:, joints_to_use, :]
    pose_quat = angle_axis_to_quaternion(pose.reshape(-1, 3)).reshape(pose.shape[0], -1, 4)
    # switch quaternion order
    # w,x,y,z -> x,y,z,w 
    pose_quat = pose_quat[:, :, [1, 2, 3, 0]]
    bullet = np.concatenate((trans, pose_quat.reshape(pose.shape[0], -1)), axis = 1)
    return bullet

if __name__ == "__main__":
    dir = "./data/motion"
    in_file = "amass_db.pkl"
    out_file = "amass_bullet.pkl"
    target_fr = 30

    """ load bullet env """
    viz = True
    mode = p.GUI if viz else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    plane_id = p.loadURDF('plane_implicit.urdf', [0, 0, 0], useMaximalCoordinates=True)
    body_id = p.loadURDF(
        fileName='./data/character/humanoid.urdf', 
        basePosition=[0, 0, 0], 
        globalScaling=1.0, 
        useFixedBase=False, 
        flags=p.URDF_MAINTAIN_LINK_ORDER
    )
    movable_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]

    """ process data """
    amass_db = joblib.load("./data/motion/amass_db.pkl")
    amass_occlusion = joblib.load("./data/motion/amass_copycat_occlusion.pkl")
    saved_data = {}
    all_data = list(amass_db.items())
    pbar = tqdm(all_data)
    for (k, v) in pbar:
        pbar.set_description(k)
        poses = v['poses']
        trans = v['trans']
        amass_fr = v['mocap_framerate']
        seq_length = poses.shape[0]

        """ filter invalid motion """
        if seq_length < 10:
            continue
        if k in amass_occlusion:
            continue
            
        """ downsample sequence from amass_fr to target_fr """
        skip = int(amass_fr/target_fr)
        poses = poses[::skip]
        trans = trans[::skip]
        seq_length = poses.shape[0]

        """ axis angle 2 quaternion """
        bullets = smpl_to_bullet(poses, trans)
        
        """ modify base position to avoid foot-ground penetration """
        p.resetBasePositionAndOrientation(body_id, bullets[0, :3], bullets[0, 3:7])
        p.resetJointStatesMultiDof(body_id, movable_indices, bullets[0, 7:].reshape(-1, 4))
        link_states = p.getLinkStates(body_id, [2, 5]) # lankle rankle
        begin_feet = min(link_states[0][0][2],  link_states[1][0][2])
        begin_root = bullets[0, 2]
        if begin_root < 0.3 and begin_feet > -0.1:
            # no need to modify
            print(f"Crawling: {k}")
        else:
            offset = 0.06 / 2 # <box size="0.0875 0.06 0.185"/>
            if begin_feet <= 0:
                # handle penetration
                bullets[:, 2] += abs(begin_feet) + offset
            else:
                # handle floating
                bullets[:, 2] -= abs(begin_feet) - offset

            # check all sequence
            new_ground_pene = []
            for s in bullets:
                p.resetBasePositionAndOrientation(body_id, s[:3], s[3:7])
                p.resetJointStatesMultiDof(body_id, movable_indices, s[7:].reshape(-1, 4))
                new_link_states = p.getLinkStates(body_id, [2, 5]) # lankle rankle
                new_ground_pene.append(min(new_link_states[0][0][2],  new_link_states[1][0][2]))
            if np.min(new_ground_pene) < -0.15:
                print(f"{k} negative sequence invalid for copycat: {np.min(new_ground_pene)}")
                continue
        
        """ add valid sequence into saved_data """
        saved_data[k] = {
            "fr": target_fr,
            "pose": bullets,
        }
    
    print(out_file, len(saved_data))
    joblib.dump(saved_data, osp.join(dir, out_file))
