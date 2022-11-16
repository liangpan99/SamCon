import joblib
import math
import numpy as np
import pybullet as p

from utils.bullet_utils import (
    computeLinVel,
    computeAngVel,
    computeAngVelRel
)

class AmassMotionData():

    def __init__(self, path, seq):
        
        # load motion
        data = joblib.load(path)[seq]

        self._duration = 1.0 / data['fr']
        self._motion = data['pose']

    def numFrames(self):
        return self._motion.shape[0]

    def keyFrameDuration(self):
        return self._duration

    def getCycleTime(self):
        """ compute cycle time of the motion sequence """
        keyFrameDuration = self.keyFrameDuration()
        cycleTime = keyFrameDuration * (self.numFrames() - 1)
        return cycleTime

    def calcCycleCount(self, curTime, cycleTime):
        phases = curTime / cycleTime
        count = math.floor(phases)
        return count
    
    def computeCycleOffset(self):
        """ compute base position (xyz) offset between first and last frame """
        frameData = self._motion[0]
        frameDataNext = self._motion[-1]
        basePosStart = frameData[:3]
        basePosEnd = frameDataNext[:3]
        cycleOffset = basePosEnd - basePosStart
        return cycleOffset
    
    def getSpecTimeState(self, t):
        """ get humanoid state of specified time """
        
        # transform t to frame, frameNext, frameFraction
        curTime = t
        keyFrameDuration = self.keyFrameDuration()
        cycleTime = self.getCycleTime()
        cycleCount = self.calcCycleCount(curTime, cycleTime)
        frameTime = curTime - cycleCount * cycleTime
        if frameTime < 0:
            frameTime += cycleTime
        
        frame = int(frameTime / keyFrameDuration)
        frameNext = frame + 1
        if frameNext >= self.numFrames():
            frameNext = frame
        frameFraction = (frameTime - frame * keyFrameDuration) / keyFrameDuration

        # interpolation
        frameData = self._motion[frame]
        frameDataNext = self._motion[frameNext]
        state = self.interpolator(frameFraction, frameData, frameDataNext)

        # modify basePos for support long-time motion
        cycleOffset = self.computeCycleOffset()
        oldBasePos = state[0:3]
        newBasePos = oldBasePos + cycleCount * cycleOffset
        state[0:3] = newBasePos

        return state

    def interpolator(self, frameFraction, frameData, frameDataNext):
        keyFrameDuration = self.keyFrameDuration()

        ##### Base Position
        basePos1Start = frameData[0:3]
        basePos1End = frameDataNext[0:3]
        basePos = basePos1Start + frameFraction * (basePos1End - basePos1Start)
        baseLinVel = computeLinVel(basePos1Start, basePos1End, keyFrameDuration)

        ##### Base Orientation
        baseOrn1Start = frameData[3:7]
        baseOrn1Next = frameDataNext[3:7]
        baseOrn = p.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
        baseAngVel = computeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration)

        ##### Joints
        jointsData = frameData[7:].reshape(-1, 4)
        jointsDataNext = frameDataNext[7:].reshape(-1, 4)
        jointsRot = []
        jointsVel = []
        for rotStart, rotEnd in zip(jointsData, jointsDataNext):
            rot = p.getQuaternionSlerp(rotStart, rotEnd, frameFraction)
            vel = computeAngVelRel(rotStart, rotEnd, keyFrameDuration)
            jointsRot.append(rot)
            jointsVel.append(vel)
        jointsRot = np.concatenate(jointsRot)
        jointsVel = np.concatenate(jointsVel)

        state = np.concatenate((basePos, baseOrn, jointsRot, baseLinVel, baseAngVel, jointsVel))
        
        return state
