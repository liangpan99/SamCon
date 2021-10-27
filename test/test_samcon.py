import sys
sys.path.append('./')

import time
import math
import json
import numpy as np
import pybullet as p
import pybullet_data

from src.samcon.samcon import Samcon
from src.samcon.mocapdata import PybulletMocapData

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
    useFPS = args.get('useFPS')
    fps = args.get('displayFPS')
    nIter = args.get('nIter')
    nSample = args.get('nSample')
    nSave = args.get('nSave')
    nSaveFinal = args.get('nSaveFinal')
    savePath = args.get('save_path')
    useGUI = args.get('useGUI')
    dataFolder = args.get('data_folder')
    dataName = args.get('data_name')

    if useGUI:
      p.connect(p.GUI)
    else:
      p.connect(p.DIRECT)

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
    
    path = pybullet_data.getDataPath() + dataFolder + dataName
    samcon = Samcon(p, simTimeStep, sampleTimeStep, savePath)
    mocap_data = PybulletMocapData(path, p)
    nIter = int(mocap_data.getCycleTime() / sampleTimeStep)

    print(f'data_name: {dataName}, nIter: {nIter}')

    animating = False
    while(p.isConnected()):
        keys = p.getKeyboardEvents()
        if isKeyTriggered(keys, ' '): # 开始动画（连续仿真）
            animating = not animating
        
        if animating:
          samcon.test(savePath, fps, useFPS)
          # samcon.learn(nIter, nSample, nSave, nSaveFinal, dataPath=path, displayFPS=fps, useFPS=useFPS)

      

if __name__ == '__main__':
    flag = 1
    args = {
        'useGUI': flag,

        'cameraDistance': 1,
        'cameraYaw': 180,
        'cameraPitch': -50,
        'cameraTargetPosition': [0, 1, 1],

        'data_folder': "/data/motions/",
        'data_name': "humanoid3d_run.txt",
        'save_path': './example/run.txt',

        'sampleTimeStep': 1./10,
        'simTimeStep': 1./240,
        'useFPS': 1,
        'displayFPS': 120,

        'nIter': 7,
        'nSample': 1400,
        'nSave': 200,
        'nSaveFinal': 10
    }
    main(**args)
