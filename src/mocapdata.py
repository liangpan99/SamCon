import math
import os
import json
import numpy as np

from config.humanoid_config import HumanoidConfig as c

from collections import namedtuple
State = namedtuple('State', [
    # Position
    'basePos',
    'baseOrn',
    'chestRot', 
    'neckRot', 
    'rightHipRot', 
    'rightKneeRot', 
    'rightAnkleRot', 
    'rightShoulderRot', 
    'rightElbowRot', 
    'leftHipRot', 
    'leftKneeRot', 
    'leftAnkleRot', 
    'leftShoulderRot', 
    'leftElbowRot',

    # Velocity
    'baseLinVel',
    'baseAngVel',
    'chestVel', 
    'neckVel', 
    'rightHipVel', 
    'rightKneeVel', 
    'rightAnkleVel', 
    'rightShoulderVel', 
    'rightElbowVel', 
    'leftHipVel', 
    'leftKneeVel', 
    'leftAnkleVel', 
    'leftShoulderVel', 
    'leftElbowVel',
])

class HumanoidPoseInterpolator(object):

  def __init__(self):
    pass

  def Reset(self,
            basePos=[0, 0, 0],
            baseOrn=[0, 0, 0, 1],
            chestRot=[0, 0, 0, 1],
            neckRot=[0, 0, 0, 1],
            rightHipRot=[0, 0, 0, 1],
            rightKneeRot=[0],
            rightAnkleRot=[0, 0, 0, 1],
            rightShoulderRot=[0, 0, 0, 1],
            rightElbowRot=[0],
            leftHipRot=[0, 0, 0, 1],
            leftKneeRot=[0],
            leftAnkleRot=[0, 0, 0, 1],
            leftShoulderRot=[0, 0, 0, 1],
            leftElbowRot=[0],
            baseLinVel=[0, 0, 0],
            baseAngVel=[0, 0, 0],
            chestVel=[0, 0, 0],
            neckVel=[0, 0, 0],
            rightHipVel=[0, 0, 0],
            rightKneeVel=[0],
            rightAnkleVel=[0, 0, 0],
            rightShoulderVel=[0, 0, 0],
            rightElbowVel=[0],
            leftHipVel=[0, 0, 0],
            leftKneeVel=[0],
            leftAnkleVel=[0, 0, 0],
            leftShoulderVel=[0, 0, 0],
            leftElbowVel=[0]):

    self._basePos = basePos
    self._baseLinVel = baseLinVel
    self._baseOrn = baseOrn
    self._baseAngVel = baseAngVel

    self._chestRot = chestRot
    self._chestVel = chestVel
    self._neckRot = neckRot
    self._neckVel = neckVel

    self._rightHipRot = rightHipRot
    self._rightHipVel = rightHipVel
    self._rightKneeRot = rightKneeRot
    self._rightKneeVel = rightKneeVel
    self._rightAnkleRot = rightAnkleRot
    self._rightAnkleVel = rightAnkleVel

    self._rightShoulderRot = rightShoulderRot
    self._rightShoulderVel = rightShoulderVel
    self._rightElbowRot = rightElbowRot
    self._rightElbowVel = rightElbowVel

    self._leftHipRot = leftHipRot
    self._leftHipVel = leftHipVel
    self._leftKneeRot = leftKneeRot
    self._leftKneeVel = leftKneeVel
    self._leftAnkleRot = leftAnkleRot
    self._leftAnkleVel = leftAnkleVel

    self._leftShoulderRot = leftShoulderRot
    self._leftShoulderVel = leftShoulderVel
    self._leftElbowRot = leftElbowRot
    self._leftElbowVel = leftElbowVel

  def ComputeLinVel(self, posStart, posEnd, deltaTime): 
    """ 根据xyz计算线速度 """
    vel = [
        (posEnd[0] - posStart[0]) / deltaTime, 
        (posEnd[1] - posStart[1]) / deltaTime,
        (posEnd[2] - posStart[2]) / deltaTime
    ]
    return vel

  def ComputeAngVel(self, ornStart, ornEnd, deltaTime, bullet_client):
    """ 用于计算base的角速度 """
    dorn = bullet_client.getDifferenceQuaternion(ornStart, ornEnd)
    axis, angle = bullet_client.getAxisAngleFromQuaternion(dorn)
    angVel = [
        (axis[0] * angle) / deltaTime, 
        (axis[1] * angle) / deltaTime,
        (axis[2] * angle) / deltaTime
    ]
    return angVel

  def ComputeAngVelRel(self, ornStart, ornEnd, deltaTime, bullet_client):
    """ 用于计算关节的角速度"""
    ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]] # 共轭四元数，几何意义：旋转轴不变，旋转角相反
    pos_diff, q_diff = bullet_client.multiplyTransforms(positionA=[0, 0, 0], orientationA=ornStartConjugate,
                                                        positionB=[0, 0, 0], orientationB=ornEnd)
    axis, angle = bullet_client.getAxisAngleFromQuaternion(q_diff)
    angVel = [
        (axis[0] * angle) / deltaTime, 
        (axis[1] * angle) / deltaTime,
        (axis[2] * angle) / deltaTime
    ]
    return angVel

  def GetStatePosVel(self):
    """ 返回一个自定义类型, 包括Pose和速度 """
    state = State(
        # Position
        basePos = self._basePos,
        baseOrn = self._baseOrn,
        chestRot = self._chestRot, 
        neckRot = self._neckRot, 
        rightHipRot = self._rightHipRot, 
        rightKneeRot = self._rightKneeRot, 
        rightAnkleRot = self._rightAnkleRot, 
        rightShoulderRot = self._rightShoulderRot, 
        rightElbowRot = self._rightElbowRot, 
        leftHipRot = self._leftHipRot, 
        leftKneeRot = self._leftKneeRot, 
        leftAnkleRot = self._leftAnkleRot, 
        leftShoulderRot = self._leftShoulderRot, 
        leftElbowRot = self._leftElbowRot,

        # Velocity
        baseLinVel = self._baseLinVel,
        baseAngVel = self._baseAngVel,
        chestVel = self._chestVel, 
        neckVel = self._neckVel, 
        rightHipVel = self._rightHipVel, 
        rightKneeVel = self._rightKneeVel, 
        rightAnkleVel = self._rightAnkleVel, 
        rightShoulderVel = self._rightShoulderVel, 
        rightElbowVel = self._rightElbowVel, 
        leftHipVel = self._leftHipVel, 
        leftKneeVel = self._leftKneeVel, 
        leftAnkleVel = self._leftAnkleVel, 
        leftShoulderVel = self._leftShoulderVel, 
        leftElbowVel = self._leftElbowVel,
    )

    return state

  def Slerp(self, frameFraction, frameData, frameDataNext, bullet_client):
    """
        Args:
            frameFraction -- 量化current frame相较于start frame走了多远
            frameData, frameDataNext -- 直接从txt中读的数据
                                        其中四元数顺序为x,y,z,w
        
    """

    keyFrameDuration = frameData[0]

    ##### Base Position
    basePos1Start = [frameData[1], frameData[2], frameData[3]]
    basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
    self._basePos = [
        basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
        basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
        basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
    ]
    self._baseLinVel = self.ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
    
    ##### Base Orientation
    baseOrn1Start = [frameData[4], frameData[5], frameData[6], frameData[7]]
    baseOrn1Next = [frameDataNext[4], frameDataNext[5], frameDataNext[6], frameDataNext[7]]
    self._baseOrn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
    self._baseAngVel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration, bullet_client)

    ##### Chest
    chestRotStart = [frameData[8], frameData[9], frameData[10], frameData[11]]
    chestRotEnd = [frameDataNext[8], frameDataNext[9], frameDataNext[10], frameDataNext[11]]
    self._chestRot = bullet_client.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)
    self._chestVel = self.ComputeAngVelRel(chestRotStart, chestRotEnd, keyFrameDuration, bullet_client)

    ##### Neck
    neckRotStart = [frameData[12], frameData[13], frameData[14], frameData[15]]
    neckRotEnd = [frameDataNext[12], frameDataNext[13], frameDataNext[14], frameDataNext[15]]
    self._neckRot = bullet_client.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)
    self._neckVel = self.ComputeAngVelRel(neckRotStart, neckRotEnd, keyFrameDuration, bullet_client)

    ##### Right Hip
    rightHipRotStart = [frameData[16], frameData[17], frameData[18], frameData[19]]
    rightHipRotEnd = [frameDataNext[16], frameDataNext[17], frameDataNext[18], frameDataNext[19]]
    self._rightHipRot = bullet_client.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd, frameFraction)
    self._rightHipVel = self.ComputeAngVelRel(rightHipRotStart, rightHipRotEnd, keyFrameDuration, bullet_client)

    ##### Right Knee
    rightKneeRotStart = [frameData[20]]
    rightKneeRotEnd = [frameDataNext[20]]
    self._rightKneeRot = [
        rightKneeRotStart[0] + frameFraction * (rightKneeRotEnd[0] - rightKneeRotStart[0])
    ]
    self._rightKneeVel = [(rightKneeRotEnd[0] - rightKneeRotStart[0]) / keyFrameDuration]

    ##### Right Ankle
    rightAnkleRotStart = [frameData[21], frameData[22], frameData[23], frameData[24]]
    rightAnkleRotEnd = [frameDataNext[21], frameDataNext[22], frameDataNext[23], frameDataNext[24]]
    self._rightAnkleRot = bullet_client.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd, frameFraction)
    self._rightAnkleVel = self.ComputeAngVelRel(rightAnkleRotStart, rightAnkleRotEnd, keyFrameDuration, bullet_client)

    ##### Right Shoulder
    rightShoulderRotStart = [frameData[25], frameData[26], frameData[27], frameData[28]]
    rightShoulderRotEnd = [frameDataNext[25], frameDataNext[26], frameDataNext[27], frameDataNext[28]]
    self._rightShoulderRot = bullet_client.getQuaternionSlerp(rightShoulderRotStart, rightShoulderRotEnd, frameFraction)
    self._rightShoulderVel = self.ComputeAngVelRel(rightShoulderRotStart, rightShoulderRotEnd, keyFrameDuration, bullet_client)

    ##### Right Elbow
    rightElbowRotStart = [frameData[29]]
    rightElbowRotEnd = [frameDataNext[29]]
    self._rightElbowRot = [
        rightElbowRotStart[0] + frameFraction * (rightElbowRotEnd[0] - rightElbowRotStart[0])
    ]
    self._rightElbowVel = [(rightElbowRotEnd[0] - rightElbowRotStart[0]) / keyFrameDuration]

    ##### Left Hip
    leftHipRotStart = [frameData[30], frameData[31], frameData[32], frameData[33]]
    leftHipRotEnd = [frameDataNext[30], frameDataNext[31], frameDataNext[32], frameDataNext[33]]
    self._leftHipRot = bullet_client.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd, frameFraction)
    self._leftHipVel = self.ComputeAngVelRel(leftHipRotStart, leftHipRotEnd, keyFrameDuration, bullet_client)

    ##### Left Knee
    leftKneeRotStart = [frameData[34]]
    leftKneeRotEnd = [frameDataNext[34]]
    self._leftKneeRot = [
        leftKneeRotStart[0] + frameFraction * (leftKneeRotEnd[0] - leftKneeRotStart[0])
    ]
    self._leftKneeVel = [(leftKneeRotEnd[0] - leftKneeRotStart[0]) / keyFrameDuration]

    ##### Left Ankle
    leftAnkleRotStart = [frameData[35], frameData[36], frameData[37], frameData[38]]
    leftAnkleRotEnd = [frameDataNext[35], frameDataNext[36], frameDataNext[37], frameDataNext[38]]
    self._leftAnkleRot = bullet_client.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd, frameFraction)
    self._leftAnkleVel = self.ComputeAngVelRel(leftAnkleRotStart, leftAnkleRotEnd, keyFrameDuration, bullet_client)

    ##### Left Shoulder
    leftShoulderRotStart = [frameData[39], frameData[40], frameData[41], frameData[42]]
    leftShoulderRotEnd = [frameDataNext[39], frameDataNext[40], frameDataNext[41], frameDataNext[42]]
    self._leftShoulderRot = bullet_client.getQuaternionSlerp(leftShoulderRotStart, leftShoulderRotEnd, frameFraction)
    self._leftShoulderVel = self.ComputeAngVelRel(leftShoulderRotStart, leftShoulderRotEnd, keyFrameDuration, bullet_client)

    ##### Left Elbow
    leftElbowRotStart = [frameData[43]]
    leftElbowRotEnd = [frameDataNext[43]]
    self._leftElbowRot = [
        leftElbowRotStart[0] + frameFraction * (leftElbowRotEnd[0] - leftElbowRotStart[0])
    ]
    self._leftElbowVel = [(leftElbowRotEnd[0] - leftElbowRotStart[0]) / keyFrameDuration]

    state = self.GetStatePosVel()
    return state


class PybulletMocapData():
    """只使用于pybullet提供txt数据集"""

    def __init__(self, path, pybullet_client):
        # 加载数据集
        assert os.path.exists(path)
        self._path = path
        self._mocap_data = []
        with open(path, 'r') as f:
            self._mocap_data = json.load(f)
        
        # 对数据集进行后处理, 调整四元数的顺序 w,x,y,z -> x,y,z,w
        self.postProcess()

        self._pb_client = pybullet_client
        self._poseInterpolator = HumanoidPoseInterpolator()
        
    def DataPath(self):
        return self._path

    def NumFrames(self):
        return len(self._mocap_data['Frames'])
    
    def KeyFrameDuration(self):
        return self._mocap_data['Frames'][0][0]

    def postProcess(self):
        """ 调整四元数的顺序 """
        numFrames = self.NumFrames()
        for i in range(numFrames):
            oldFrameData = self._mocap_data['Frames'][i]
            newFrameData = []

            # keyFrameDuration
            newFrameData += [oldFrameData[0]]
            # base position
            newFrameData += [oldFrameData[1], oldFrameData[2], oldFrameData[3]]
            # base orientation
            newFrameData += [oldFrameData[5], oldFrameData[6], oldFrameData[7], oldFrameData[4]]
            # 12 joints
            newFrameData += [oldFrameData[9], oldFrameData[10], oldFrameData[11], oldFrameData[8]]
            newFrameData += [oldFrameData[13], oldFrameData[14], oldFrameData[15], oldFrameData[12]]
            newFrameData += [oldFrameData[17], oldFrameData[18], oldFrameData[19], oldFrameData[16]]
            newFrameData += [oldFrameData[20]]
            newFrameData += [oldFrameData[22], oldFrameData[23], oldFrameData[24], oldFrameData[21]]
            newFrameData += [oldFrameData[26], oldFrameData[27], oldFrameData[28], oldFrameData[25]]
            newFrameData += [oldFrameData[29]]
            newFrameData += [oldFrameData[31], oldFrameData[32], oldFrameData[33], oldFrameData[30]]
            newFrameData += [oldFrameData[34]]
            newFrameData += [oldFrameData[36], oldFrameData[37], oldFrameData[38], oldFrameData[35]]
            newFrameData += [oldFrameData[40], oldFrameData[41], oldFrameData[42], oldFrameData[39]]
            newFrameData += [oldFrameData[43]]

            self._mocap_data['Frames'][i] = newFrameData

    def getCycleTime(self):
        """ 计算运动序列的持续时间, 单位是秒 """
        keyFrameDuration = self.KeyFrameDuration()
        cycleTime = keyFrameDuration * (self.NumFrames() - 1)
        return cycleTime

    def calcCycleCount(self, curTime, cycleTime):
        """ 计算循环次数
        
        Args:
            curTime -- 当前时间
            cycleTime -- 运动序列的持续时间(一次循环所需要的时间)
        Return:
            count -- 次数，如果仿真时间小于一次循环所需要的时间，将会返回0

        """
        phases = curTime / cycleTime
        count = math.floor(phases)
        return count

    def computeCycleOffset(self):
        """ 计算数据集中第一帧和最后一帧的basePosition(xyz)的偏移量 """
        firstFrame = 0
        lastFrame = self.NumFrames() - 1
        frameData = self._mocap_data['Frames'][firstFrame]
        frameDataNext = self._mocap_data['Frames'][lastFrame]

        basePosStart = [frameData[1], frameData[2], frameData[3]]
        basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        cycleOffset = [
            basePosEnd[0] - basePosStart[0], 
            basePosEnd[1] - basePosStart[1],
            basePosEnd[2] - basePosStart[2]
        ]
        return cycleOffset


    def getSpecTimeState(self, t):
        """ 获取指定时刻的State, 包括速度 """
        
        # 将时间t转换为frame, frameNext, frameFraction
        curTime = t
        keyFrameDuration = self.KeyFrameDuration()
        cycleTime = self.getCycleTime()
        cycleCount = self.calcCycleCount(curTime, cycleTime)
        frameTime = curTime - cycleCount * cycleTime
        if frameTime < 0:
            frameTime += cycleTime
        
        frame = int(frameTime / keyFrameDuration)
        frameNext = frame + 1
        if frameNext >= self.NumFrames():
            frameNext = frame
        frameFraction = (frameTime - frame * keyFrameDuration) / keyFrameDuration

        # 取出帧的数据, 进行插值
        frameData = self._mocap_data['Frames'][frame]
        frameDataNext = self._mocap_data['Frames'][frameNext]

        self._poseInterpolator.Slerp(frameFraction, frameData, frameDataNext, self._pb_client)

        cycleOffset = self.computeCycleOffset()
        oldPos = self._poseInterpolator._basePos
        self._poseInterpolator._basePos = [
            oldPos[0] + cycleCount * cycleOffset[0],
            oldPos[1] + cycleCount * cycleOffset[1],
            oldPos[2] + cycleCount * cycleOffset[2]
        ] 

        return self._poseInterpolator.GetStatePosVel()