import pybullet as p

def getEulerFromQuaternion(quat):
    return p.getEulerFromQuaternion(quat)

def getQuaternionFromEuler(euler):
    return p.getQuaternionFromEuler(euler)

def computeLinVel(posStart, posEnd, deltaTime): 
    vel = [
        (posEnd[0] - posStart[0]) / deltaTime, 
        (posEnd[1] - posStart[1]) / deltaTime,
        (posEnd[2] - posStart[2]) / deltaTime
    ]
    return vel

def computeAngVel(ornStart, ornEnd, deltaTime):
    """ compute angular velocity in world coordinate (for base) """
    ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]]
    _, dorn = p.multiplyTransforms(
        positionA=[0, 0, 0], orientationA=ornEnd,
        positionB=[0, 0, 0], orientationB=ornStartConjugate
    )
    axis, angle = p.getAxisAngleFromQuaternion(dorn)
    angVel = [
        (axis[0] * angle) / deltaTime, 
        (axis[1] * angle) / deltaTime,
        (axis[2] * angle) / deltaTime
    ]
    return angVel

def computeAngVelRel(ornStart, ornEnd, deltaTime):
    """ compute angular velocity in local coordinate (for joints) """
    ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]]
    _, dorn = p.multiplyTransforms(
        positionA=[0, 0, 0], orientationA=ornStartConjugate,
        positionB=[0, 0, 0], orientationB=ornEnd
    )
    axis, angle = p.getAxisAngleFromQuaternion(dorn)
    angVel = [
        (axis[0] * angle) / deltaTime, 
        (axis[1] * angle) / deltaTime,
        (axis[2] * angle) / deltaTime
    ]
    return angVel

def isKeyTriggered(keys, key):
    o = ord(key)
    if o in keys:
        return keys[ord(key)] & p.KEY_WAS_TRIGGERED
    return False