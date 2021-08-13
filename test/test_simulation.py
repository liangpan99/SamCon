import sys
sys.path.append('./')

import time
import math
import json
import pybullet as p
import pybullet_data

from src.mocapdata import PybulletMocapData
from src.simulation import HumanoidStablePD

def isKeyTriggered(keys, key):
  o = ord(key)
  if o in keys:
    return keys[ord(key)] & p.KEY_WAS_TRIGGERED
  return False

def main(**args):

    # 读取参数
    cameraArgs = {
        'cameraDistance': args.get('cameraDistance'),
        'cameraYaw': args.get('cameraYaw'),
        'cameraPitch': args.get('cameraPitch'),
        'cameraTargetPosition': args.get('cameraTargetPosition')
    }
    simTimeStep = args.get('simTimeStep')
    sampleTimeStep = args.get('sampleTimeStep')
    fps = args.get('displayFPS')


    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, -9.8, 0)
    p.resetDebugVisualizerCamera(**cameraArgs)
    p.setPhysicsEngineParameter(numSolverIterations=10)
    p.setPhysicsEngineParameter(numSubSteps=2)
    p.setTimeStep(simTimeStep)


    # 加载urdf
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0]) # 绕x轴旋转-90度
    plane = p.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)
    p.changeDynamics(plane, linkIndex=-1, lateralFriction=0.9)

    humanoidPD = HumanoidStablePD(pybullet_client=p, timeStep=simTimeStep)
    humanoid = humanoidPD._sim_model
    
    # 加载数据集
    path = pybullet_data.getDataPath() + args.get('data_path')
    motionData = PybulletMocapData(path=path, pybullet_client=p)


    animating = False
    i = 0
    temp_state = 0
    while(p.isConnected()):
        keys = p.getKeyboardEvents()
        if isKeyTriggered(keys, ' '): # 开始动画（连续仿真）
            animating = not animating
        if isKeyTriggered(keys, 'r'): # 重置 reset
            i = 0
        
        if animating:
            if i == 0:
                start_state = motionData.getSpecTimeState(t=0)
            else:
                start_state = temp_state
            
            targetT = (i + 1) * sampleTimeStep
            target_state = motionData.getSpecTimeState(t=targetT)
            humanoidPD.resetState(start_state, target_state)
            sim_target_state, cost = humanoidPD.simulation(False, target_state, sampleTimeStep, fps)
            
            temp_state = sim_target_state
            i += 1
        


if __name__ == '__main__':
    args = {
        'cameraDistance': 2,
        'cameraYaw': 180,
        'cameraPitch': -50,
        'cameraTargetPosition': [0, 1, 1],

        'data_path': "/data/motions/humanoid3d_walk.txt",

        'sampleTimeStep': 1./10,
        'simTimeStep': 1./2000,
        'displayFPS': 600,
    }
    main(**args)