""" 
    humanoid.urdf 模型的相关参数
"""

import os

class HumanoidConfig():

    """ load URDF

    """
    fileName = os.path.join(os.getcwd(), 'data/urdf/humanoid.urdf')
    basePos = [0, 0, 0]
    globalScale = 0.25


    """ joint 关节序号

        0, 8, 14 不可以控制，类型为fixed
        其余为可以控制的关节，类型分为revolute 1-DoF和spherical 3-DoF两种
    """
    root = 0            # fixed
    rightWrist = 8      # fixed
    leftWrist = 14      # fixed

    chest = 1           # 3 DoF spherical
    neck = 2            # 3 DoF
    rightHip = 3        # 3 DoF 
    rightKnee = 4       # 1 DoF revolute 
    rightAnkle = 5      # 3 DoF
    rightShoulder = 6   # 3 DoF
    rightElbow = 7      # 1 DoF revolute 
    leftHip = 9         # 3 DoF
    leftKnee = 10       # 1 DoF revolute 
    leftAnkle = 11      # 3 DoF
    leftShoulder = 12   # 3 DoF
    leftElbow = 13      # 1 DoF revolute 

    ##### 可以控制的关节序号
    controllableJointIndicesAll = [
        chest, neck, 
        rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow, 
        leftHip, leftKnee, leftAnkle, leftShoulder, leftElbow
    ]

    ##### Link 肢体序号 (与关节序号对应)
    linkIndicesAll = [
        root, chest, neck, 
        rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow, rightWrist, 
        leftHip, leftKnee, leftAnkle, leftShoulder, leftElbow, leftWrist
    ]


    """ PD controller

        预先设定好的Kp Kd系数
        len=43 basePos 3 + baseOrn 4 + 3-DoF jointOrn 4x8=32 + 1-DoF jointOrn 1x4=4

        预先设定好的MaxForces
        长度同Kp Kd  
    """
    Kp = [
        0, 0, 0,                # 3个basePos 因为不能对其控制，所以系数设为0
        0, 0, 0, 0,             # 4个baseOrient
        1000, 1000, 1000, 1000, # chest
        100, 100, 100, 100,     # neck
        500, 500, 500, 500,     # right Hip
        500,                    # right Knee
        400, 400, 400, 400,     # right Ankle
        400, 400, 400, 400,     # right Shoulder
        300,                    # right Elbow
        500, 500, 500, 500,     # left Hip
        500,                    # left Knee
        400, 400, 400, 400,     # left Ankle
        400, 400, 400, 400,     # left Shoulder
        300,                    # left Elbow
    ]

    Kd = [
        0, 0, 0, 
        0, 0, 0, 0, 
        100, 100, 100, 100,     # chest
        10, 10, 10, 10,         # neck
        50, 50, 50, 50,         # right Hip
        50,                     # right Knee
        40, 40, 40, 40,         # right Ankle
        40, 40, 40, 40,         # right Shoulder
        30,                     # right Elbow
        50, 50, 50, 50,         # left Hip
        50,                     # left Knee
        40, 40, 40, 40,         # left Ankle
        40, 40, 40, 40,         # left Shoulder
        30,                     # left Elbow
    ]

    maxForces = [
        0, 0, 0,                # base position
        0, 0, 0, 0,             # base orientation
        200, 200, 200, 200,     # chest
        50, 50, 50, 50,         # neck
        200, 200, 200, 200,     # right hip
        150,                    # right knee
        90, 90, 90, 90,         # right ankle
        100, 100, 100, 100,     # right shoulder
        60,                     # right elbow
        200, 200, 200, 200,     # left hip
        150,                    # left knee
        90, 90, 90, 90,         # left ankle
        100, 100, 100, 100,     # left shoulder
        60,                     # left elbow
    ]

    maxForcesInf = [
        0, 0, 0,                                                    # base position
        0, 0, 0, 0,                                                 # base orientation
        float('inf'), float('inf'), float('inf'), float('inf'),     # chest
        float('inf'), float('inf'), float('inf'), float('inf'),     # neck
        float('inf'), float('inf'), float('inf'), float('inf'),     # right hip
        float('inf'),                                               # right knee
        float('inf'), float('inf'), float('inf'), float('inf'),     # right ankle
        float('inf'), float('inf'), float('inf'), float('inf'),     # right shoulder
        float('inf'),                                               # right elbow
        float('inf'), float('inf'), float('inf'), float('inf'),     # left hip
        float('inf'),                                               # left knee
        float('inf'), float('inf'), float('inf'), float('inf'),     # left ankle
        float('inf'), float('inf'), float('inf'), float('inf'),     # left shoulder
        float('inf'),                                               # left elbow
    ]

    """ 采样窗口大小 """
    samplingWindow = [
        [0.1, 0.1, 0.4],      # chest
        [0.2, 0.2, 0.2],      # neck
        [0.3, 0.1, 0.8],      # right hip
        [0.4],                # right knee
        [0.1, 0.2, 0.8],      # right ankle
        [0.2, 0.2, 0.2],      # right shoulder
        [0.0],                # right elbow
        [0.3, 0.1, 0.8],      # left hip
        [0.4],                # left knee
        [0.1, 0.2, 0.8],      # left ankle
        [0.2, 0.2, 0.2],      # left shoulder
        [0.0],                # left elbow
    ]


    """ cost 权值 """
    pose_weight = 8
    root_weight = 5
    end_effector_weight = 30
    balance_weight = 20
    