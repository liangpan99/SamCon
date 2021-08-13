""" 测试角速度计算 """
import sys
sys.path.append('./')

import math
import json
import pybullet as p
import pybullet_data

from src.simulation import HumanoidStablePD

def main(**args):

    # 读取参数
    cameraArgs = {
        'cameraDistance': args.get('cameraDistance'),
        'cameraYaw': args.get('cameraYaw'),
        'cameraPitch': args.get('cameraPitch'),
        'cameraTargetPosition': args.get('cameraTargetPosition')
    }

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.setGravity(0, -9.8, 0)
    p.resetDebugVisualizerCamera(**cameraArgs)

    # 加载urdf
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0]) # 绕x轴旋转-90度
    p.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)
    
    # 加载数据集
    path = pybullet_data.getDataPath() + "/data/motions/humanoid3d_backflip.txt"
    with open(path, 'r') as f:
        motion_dict = json.load(f)
    numFrames = len(motion_dict['Frames'])

    startId = 10
    endId = 2
    frameData = motion_dict['Frames'][startId]
    frameDataNext = motion_dict['Frames'][endId]

    # 数据集中的都是单位四元数 那么四元数的逆就等于其共轭
    baseOrnStart = [frameData[5], frameData[6], frameData[7], frameData[4]]
    baseOrnNext = [frameDataNext[5], frameDataNext[6], frameDataNext[7], frameDataNext[4]]
    
    dorn_1 = p.getDifferenceQuaternion(baseOrnStart, baseOrnNext)

    # start的共轭四元数 代表转轴相同 旋转角相反
    baseOrnStartConjugate = [-baseOrnStart[0], -baseOrnStart[1], -baseOrnStart[2], baseOrnStart[3]]
    _, dorn_2 = p.multiplyTransforms([0, 0, 0], baseOrnNext,
                                     [0, 0, 0], baseOrnStartConjugate)
    print(dorn_1)
    print(dorn_2)

    # print('baseOrnStart:  ', p.getEulerFromQuaternion(baseOrnStart))
    # print('baseOrnNext :  ', p.getEulerFromQuaternion(baseOrnNext))
    # print('baseOrnNext 轴角 :  ', p.getAxisAngleFromQuaternion(baseOrnNext))

    _, dorn_1_recover = p.multiplyTransforms([0, 0, 0], dorn_1,
                                        [0, 0, 0], baseOrnStart)
    # print('方式1 相对旋转的欧拉角')
    # print(p.getEulerFromQuaternion(dorn_1))
    # print('用方式1计算出的相对旋转，来进行恢复：')
    # print(p.getEulerFromQuaternion(dorn_1_recover))

    # axis, angle = p.getAxisAngleFromQuaternion(dorn_1_recover)
    # print(axis, angle)

    _, dorn_2_recover = p.multiplyTransforms([0, 0, 0], dorn_2,
                                        [0, 0, 0], baseOrnStart)
    # print('方式2 相对旋转的欧拉角')
    # print(p.getEulerFromQuaternion(dorn_2))
    # print('用方式2计算出的相对旋转，来进行恢复：')
    # print(p.getEulerFromQuaternion(dorn_2_recover))

    # axis, angle = p.getAxisAngleFromQuaternion(dorn_2_recover)
    # print(axis, angle)

    while(p.isConnected()):
        pass


        



if __name__ == '__main__':
    args = {
        'cameraDistance': 2,
        'cameraYaw': 180,
        'cameraPitch': -50,
        'cameraTargetPosition': [0, 1, 1],
    }
    main(**args)