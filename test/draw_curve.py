import sys
sys.path.append('./')

import os
import json
import numpy as np
import pybullet as p
from tensorboardX import SummaryWriter

from src.samcon.mocapdata import State

def getStateFromList(data):
        """ 从列表转换为collections.namedtuple类型的state
        
        Return:
            pose len=43
            velocity len=34
            pose+velocity len=77
        """

        state = State(
        # Position
        basePos = data[0:3],
        baseOrn = data[3:7],
        chestRot = data[7:11], 
        neckRot = data[11:15], 
        rightHipRot = data[15:19], 
        rightKneeRot = data[19:20], 
        rightAnkleRot = data[20:24], 
        rightShoulderRot = data[24:28], 
        rightElbowRot = data[28:29], 
        leftHipRot = data[29:33], 
        leftKneeRot = data[33:34], 
        leftAnkleRot = data[34:38], 
        leftShoulderRot = data[38:42], 
        leftElbowRot = data[42:43],

        # Velocity
        baseLinVel = data[43:46],
        baseAngVel = data[46:49],
        chestVel = data[49:52], 
        neckVel = data[52:55], 
        rightHipVel = data[55:58], 
        rightKneeVel = data[58:59], 
        rightAnkleVel = data[59:62], 
        rightShoulderVel = data[62:65], 
        rightElbowVel = data[65:66], 
        leftHipVel = data[66:69], 
        leftKneeVel = data[69:70], 
        leftAnkleVel = data[70:73], 
        leftShoulderVel = data[73:76], 
        leftElbowVel = data[76:77],
        )
        
        return state

def getListFromNamedtuple(namedtuple, is_pose=True, is_velocity=True):
        """ 从collections.namedtuple类型的state转换为列表 
        
        Return:
            pose len=43
            velocity len=34
            pose+velocity len=77
        """
        if is_pose:
            pose = list(namedtuple.basePos) + list(namedtuple.baseOrn) \
                + list(namedtuple.chestRot) + list(namedtuple.neckRot) \
                + list(namedtuple.rightHipRot) + list(namedtuple.rightKneeRot) + list(namedtuple.rightAnkleRot) \
                + list(namedtuple.rightShoulderRot) + list(namedtuple.rightElbowRot) \
                + list(namedtuple.leftHipRot) + list(namedtuple.leftKneeRot) + list(namedtuple.leftAnkleRot) \
                + list(namedtuple.leftShoulderRot) + list(namedtuple.leftElbowRot)
        else:
            pose = []

        if is_velocity:
            velocity = list(namedtuple.baseLinVel) + list(namedtuple.baseAngVel) \
                + list(namedtuple.chestVel) + list(namedtuple.neckVel) \
                + list(namedtuple.rightHipVel) + list(namedtuple.rightKneeVel) + list(namedtuple.rightAnkleVel) \
                + list(namedtuple.rightShoulderVel) + list(namedtuple.rightElbowVel) \
                + list(namedtuple.leftHipVel) + list(namedtuple.leftKneeVel) + list(namedtuple.leftAnkleVel) \
                + list(namedtuple.leftShoulderVel) + list(namedtuple.leftElbowVel)
        else:
            velocity = []
        
        return pose+velocity

def main():
    # prepare tensorboardX
    curve_save_folder = './example/tensorboardX/info'
    ref_writer = SummaryWriter(os.path.join(curve_save_folder, 'reference'))
    sam_writer = SummaryWriter(os.path.join(curve_save_folder, 'sampled'))

    # load samcon result txt data
    data_path = './example/run.txt'
    with open(data_path, 'r') as f:
            data = json.load(f)
    
    nIter = data['info']['nIter']
    nSaveFinal = data['info']['nSaveFinal']
    sampleTimeStep = data['info']['sampleTimeStep']
    simTimeStep = data['info']['simulationTimeStep']

    print(f'nIter={nIter}\nnSaveFinal={nSaveFinal}\nsamleTimeStep={sampleTimeStep}\nsiTimeStep={simTimeStep}\n')

    path_id = 0
    key = f'path_{path_id}'
    reference_states = data[key]['reference_states']
    sampled_states = data[key]['sampled_states']
    simulated_states = data[key]['simulated_states']
    assert len(reference_states) == nIter
    assert len(sampled_states) == nIter
    assert len(simulated_states) == nIter

    # traversal and draw
    joint_names = ['chest', 'neck', 
                'rightHip', 'rightKnee', 'rightAnkle', 'rightShoulder', 'rightElbow',
                'leftHip', 'leftKnee', 'leftAnkle', 'leftShoulder', 'leftElbow'
                ]
    joint_dofs = [4, 4, 
                4, 1, 4, 4, 1,
                4, 1, 4, 4, 1
                ]
    
    dummy_dof = 7

    for t in range(1, nIter+1):
        reference_state = reference_states[t-1] # type: list
        sampled_state = sampled_states[t-1] # type: list
        simulated_state = simulated_states[t-1] # type: list

        count_dof = dummy_dof
        for j in range(len(joint_names)):
            name = joint_names[j]
            dof = joint_dofs[j]

            # queternion
            pos_ref = reference_state[count_dof:count_dof+dof]
            pos_sam = sampled_state[count_dof:count_dof+dof]
            count_dof += dof

            if dof == 4:
                pos_ref = p.getEulerFromQuaternion(pos_ref)
                pos_sam = p.getEulerFromQuaternion(pos_sam)

                ref_writer.add_scalar(f'{name}/X', pos_ref[0], t)
                ref_writer.add_scalar(f'{name}/Y', pos_ref[1], t)
                ref_writer.add_scalar(f'{name}/Z', pos_ref[2], t)
                sam_writer.add_scalar(f'{name}/X', pos_sam[0], t)
                sam_writer.add_scalar(f'{name}/Y', pos_sam[1], t)
                sam_writer.add_scalar(f'{name}/Z', pos_sam[2], t)

            if dof == 1:
                ref_writer.add_scalar(f'{name}/X', pos_ref[0], t)
                sam_writer.add_scalar(f'{name}/X', pos_sam[0], t)

    ref_writer.close()
    sam_writer.close()
      

if __name__ == '__main__':
    
    main()
