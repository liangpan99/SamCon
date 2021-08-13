import sys
sys.path.append('./')

import time
import math
import json
import numpy as np
import pybullet as p
import pybullet_data

from src.samcon import SamCon

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
    nIter = args.get('nIter')
    nSample = args.get('nSample')
    nSave = args.get('nSave')
    nSaveFinal = args.get('nSaveFinal')
    savePath = args.get('save_path')


    p.connect()
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
    
    path = pybullet_data.getDataPath() + args.get('data_path')
    samcon = SamCon(p, simTimeStep, sampleTimeStep, savePath)

    animating = False
    while(p.isConnected()):
        keys = p.getKeyboardEvents()
        if isKeyTriggered(keys, ' '): # 开始动画（连续仿真）
            animating = not animating
        
        if animating:
          #samcon.test(savePath, fps)
          samcon.learn(nIter, nSample, nSave, nSaveFinal, dataPath=path, displayFPS=fps)

        


if __name__ == '__main__':
    args = {
        'cameraDistance': 2,
        'cameraYaw': 180,
        'cameraPitch': -50,
        'cameraTargetPosition': [0, 1, 1],

        'data_path': "/data/motions/humanoid3d_walk.txt",
        'save_path': './data/reconstructed_motion/result.txt',

        'sampleTimeStep': 1./10,
        'simTimeStep': 1./2000,
        'displayFPS': 10000,

        'nIter': 10,
        'nSample': 1400,
        'nSave': 200,
        'nSaveFinal': 30
    }
    main(**args)