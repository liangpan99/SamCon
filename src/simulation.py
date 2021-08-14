from ntpath import join
import time
import math
from typing import List
from pybullet_utils import pd_controller_stable

from config.humanoid_config import HumanoidConfig as c
from src.mocapdata import State

class HumanoidStablePD():

    def __init__(self, pybullet_client, timeStep):
        self._pb_client = pybullet_client
        self._stablePD = pd_controller_stable.PDControllerStableMultiDof(self._pb_client)
        self._kpOrg = c.Kp
        self._kdOrg = c.Kd
        self._jointIndicesAll = c.controllableJointIndicesAll
        self._jointDofCounts = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]
        self._totalDofs = 7
        for dof in self._jointDofCounts:
            self._totalDofs += dof
        self._simTimeStep = timeStep

        # 从URDF文件中加载模型
        self._sim_model = self._pb_client.loadURDF(
            fileName = c.fileName,
            basePosition = c.basePos,
            globalScaling = c.globalScale,
            useFixedBase = False,
            flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        )

        self._kin_model = self._pb_client.loadURDF(
            fileName = c.fileName,
            basePosition = c.basePos,
            globalScaling = c.globalScale,
            useFixedBase = True,
            flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        )

        ## 设置参数
        # self._sim_model 摩擦力
        self._pb_client.changeDynamics(self._sim_model, -1, lateralFriction=0.9)
        for i in range(self._pb_client.getNumJoints(self._sim_model)):
            self._pb_client.changeDynamics(self._sim_model, i, lateralFriction=0.9)
        
        # self._kin_model 允许碰撞 设置透明度
        self._pb_client.changeDynamics(
            self._kin_model,
            -1,
            activationState=self._pb_client.ACTIVATION_STATE_SLEEP +
                            self._pb_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                            self._pb_client.ACTIVATION_STATE_DISABLE_WAKEUP)
        alpha = 0.7
        for j in range(self._pb_client.getNumJoints(self._kin_model)):
            self._pb_client.changeVisualShape(self._kin_model, j, rgbaColor=[1, 1, 1, alpha])
            self._pb_client.setCollisionFilterGroupMask(
                self._kin_model,
                j,
                collisionFilterGroup=0,
                collisionFilterMask=0)
        
        # self._sim_model 初始化关节电机
        for j in self._jointIndicesAll:
            self._pb_client.setJointMotorControl2(
                self._sim_model,
                j,
                self._pb_client.POSITION_CONTROL,
                targetPosition=0,
                positionGain=0,
                targetVelocity=0,
                force=0)
            self._pb_client.setJointMotorControlMultiDof(
                self._sim_model,
                j,
                self._pb_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[0, 0, 0]
            )

    def initPose(self, pose, phys_model, is_initBase, is_initVel):
        """ 初始化物理模型的pose和velocity

        Args:
            pose -- 自定义类型 collections.namedtuple
            phys_model -- 待初始化的机器人Id
            is_initBase -- 是否对base进行初始化
            is_initVel -- 是否对速度进行初始化
        
        """
        if is_initVel:
            if is_initBase:
                self._pb_client.resetBasePositionAndOrientation(phys_model, pose.basePos, pose.baseOrn)
                self._pb_client.resetBaseVelocity(phys_model, pose.baseLinVel, pose.baseAngVel)
            
            indices = self._jointIndicesAll

            jointPositions = [pose.chestRot, pose.neckRot, 
                              pose.rightHipRot, pose.rightKneeRot, pose.rightAnkleRot, 
                              pose.rightShoulderRot, pose.rightElbowRot, 
                              pose.leftHipRot, pose.leftKneeRot, pose.leftAnkleRot, 
                              pose.leftShoulderRot, pose.leftElbowRot]
            jointVelocities = [pose.chestVel, pose.neckVel, 
                               pose.rightHipVel, pose.rightKneeVel, pose.rightAnkleVel, 
                               pose.rightShoulderVel, pose.rightElbowVel, 
                               pose.leftHipVel, pose.leftKneeVel, pose.leftAnkleVel, 
                               pose.leftShoulderVel, pose.leftElbowVel]
            
            self._pb_client.resetJointStatesMultiDof(phys_model, indices,
                                            jointPositions, jointVelocities)
        else:
            if is_initBase:
                self._pb_client.resetBasePositionAndOrientation(phys_model, pose.basePos, pose.baseOrn)
            
            indices = self._jointIndicesAll

            jointPositions = [pose.chestRot, pose.neckRot, 
                              pose.rightHipRot, pose.rightKneeRot, pose.rightAnkleRot, 
                              pose.rightShoulderRot, pose.rightElbowRot, 
                              pose.leftHipRot, pose.leftKneeRot, pose.leftAnkleRot, 
                              pose.leftShoulderRot, pose.leftElbowRot]
            
            self._pb_client.resetJointStatesMultiDof(phys_model, indices,
                                            jointPositions)


    def resetState(self, start_state, end_state):
        """ 重置状态 (state: position + velocity)"""
        if end_state != None:
            self._desiredState = end_state

            # 用end_state来重置self._kin_model
            self.initPose(end_state, self._kin_model, is_initBase=True, is_initVel=True)

        if start_state != None:
            # 用start_state来重置self._sim_model
            # 允许修改base, 因为start_state本身就是仿真的结果
            self.initPose(start_state, self._sim_model, is_initBase=True, is_initVel=True)


    def computePDForces(self, desiredPositions, desiredVelocities, maxForces):
        """ 使用pybullet官方实现的stablePD计算torques
            Compute torques from the stable PD controller.
            
            只需传入target，观测值在ComputePD函数内部进行读取
        """

        if desiredVelocities == None:
            desiredVelocities = [0] * self._totalDofs
        
        taus = self._stablePD.computePD(bodyUniqueId=self._sim_model,
                                        jointIndices=self._jointIndicesAll,
                                        desiredPositions=desiredPositions,
                                        desiredVelocities=desiredVelocities,
                                        kps=self._kpOrg,
                                        kds=self._kdOrg,
                                        maxForces=maxForces,
                                        timeStep=self._simTimeStep)
        return taus
        
    def applyPDForces(self, taus):
        """ Apply pre-computed torques

        """
        dofIndex = 7
        scaling = 1
        forces = []

        for index in range(len(self._jointIndicesAll)):

            if self._jointDofCounts[index] == 4:
                force = [
                    scaling * taus[dofIndex + 0],
                    scaling * taus[dofIndex + 1],
                    scaling * taus[dofIndex + 2]
                ]
            if self._jointDofCounts[index] == 1:
                force = [
                    scaling * taus[dofIndex],
                ]
            forces.append(force)
            dofIndex += self._jointDofCounts[index]

        self._pb_client.setJointMotorControlMultiDofArray(self._sim_model,
                                                        self._jointIndicesAll,
                                                        self._pb_client.TORQUE_CONTROL,
                                                        forces=forces)


    def simulation(self, desiredPosition, sampleTimeStep, displayFPS):
        """ 执行仿真
        
        Args:
            sampleTimeStep -- 采样时间 用于自动计算仿真次数
        Returns:
            simulatedState -- 仿真结果 是collections.namedtuple自定义类型
            cost
        
        """

        # base不能控制, 因此需要7个0
        desiredPosition = [0, 0, 0, 0, 0, 0, 0] \
            + list(desiredPosition.chestRot) + list(desiredPosition.neckRot) \
            + list(desiredPosition.rightHipRot) + list(desiredPosition.rightKneeRot) + list(desiredPosition.rightAnkleRot) \
            + list(desiredPosition.rightShoulderRot) + list(desiredPosition.rightElbowRot) \
            + list(desiredPosition.leftHipRot) + list(desiredPosition.leftKneeRot) + list(desiredPosition.leftAnkleRot) \
            + list(desiredPosition.leftShoulderRot) + list(desiredPosition.leftElbowRot)

        
        numSim = int(sampleTimeStep / self._simTimeStep)
        for i in range(numSim):
            taus = self.computePDForces(desiredPosition, desiredVelocities=None, maxForces=c.maxForces)
            self.applyPDForces(taus)
            self._pb_client.stepSimulation()
            time.sleep(1./displayFPS) # 显示FPS

        ## 返回仿真结果及cost
        # 仿真结果是base的平移,旋转,线速度和角速度, 以及各个可控关节的旋转和角速度
        # cost是一个标量

        sim_basePos, sim_baseOrn = self._pb_client.getBasePositionAndOrientation(self._sim_model) 
        sim_baseLinVel, sim_baseAngVel = self._pb_client.getBaseVelocity(self._sim_model)
        sim_jointStates = self._pb_client.getJointStatesMultiDof(self._sim_model, c.linkIndicesAll)

        simulatedState = State(
            # Position
            basePos = sim_basePos,
            baseOrn = sim_baseOrn,
            chestRot = sim_jointStates[c.chest][0], 
            neckRot = sim_jointStates[c.neck][0], 
            rightHipRot = sim_jointStates[c.rightHip][0], 
            rightKneeRot = sim_jointStates[c.rightKnee][0], 
            rightAnkleRot = sim_jointStates[c.rightAnkle][0], 
            rightShoulderRot = sim_jointStates[c.rightShoulder][0], 
            rightElbowRot = sim_jointStates[c.rightElbow][0], 
            leftHipRot = sim_jointStates[c.leftHip][0], 
            leftKneeRot = sim_jointStates[c.leftKnee][0], 
            leftAnkleRot = sim_jointStates[c.leftAnkle][0], 
            leftShoulderRot = sim_jointStates[c.leftShoulder][0], 
            leftElbowRot = sim_jointStates[c.leftElbow][0],

            # Velocity
            baseLinVel = sim_baseLinVel,
            baseAngVel = sim_baseAngVel,
            chestVel = sim_jointStates[c.chest][1], 
            neckVel = sim_jointStates[c.neck][1], 
            rightHipVel = sim_jointStates[c.rightHip][1], 
            rightKneeVel = sim_jointStates[c.rightKnee][1], 
            rightAnkleVel = sim_jointStates[c.rightAnkle][1], 
            rightShoulderVel = sim_jointStates[c.rightShoulder][1], 
            rightElbowVel = sim_jointStates[c.rightElbow][1], 
            leftHipVel = sim_jointStates[c.leftHip][1], 
            leftKneeVel = sim_jointStates[c.leftKnee][1], 
            leftAnkleVel = sim_jointStates[c.leftAnkle][1], 
            leftShoulderVel = sim_jointStates[c.leftShoulder][1], 
            leftElbowVel = sim_jointStates[c.leftElbow][1],
        )

        cost = self.computeCost()
        
        return simulatedState, cost

    def computeCost(self):
        """ 计算cost
        
        目前包含两项:
            1. 可控joints的旋转和速度
            2. root的平移,旋转,线速度,角速度

        """
        pose_err = 0
        vel_err = 0
        root_err = 0

        # Pose Control

        num_joints = self._pb_client.getNumJoints(self._kin_model)
        mJointWeights = [
        0.20833, 0.10416, 0.0625, 0.10416, 0.0625, 0.041666666666666671, 0.0625, 0.0416, 0.00,
        0.10416, 0.0625, 0.0416, 0.0625, 0.0416, 0.0000
        ]

        jointIndicesControllable = c.controllableJointIndicesAll # 只计算可以控制的joint, 因为fixed类型的joint读出来的信息都是0
        jointIndicesAll = range(num_joints)
        simJointStates = self._pb_client.getJointStatesMultiDof(self._sim_model, jointIndicesAll)
        kinJointStates = self._pb_client.getJointStatesMultiDof(self._kin_model, jointIndicesAll)
        
        for j in jointIndicesControllable:
            curr_pose_err = 0
            curr_vel_err = 0
            w = mJointWeights[j]
            simJointInfo = simJointStates[j]
            kinJointInfo = kinJointStates[j]

            if len(simJointInfo[0]) == 1: # joint type: revolute 1-Dof
                angle = simJointInfo[0][0] - kinJointInfo[0][0]
                curr_pose_err = angle * angle
                velDiff = simJointInfo[1][0] - kinJointInfo[1][0]
                curr_vel_err = velDiff * velDiff
            if len(simJointInfo[0]) == 4: # joint type: spherical 3-Dof
                diffQuat = self._pb_client.getDifferenceQuaternion(simJointInfo[0], kinJointInfo[0])
                axis, angle = self._pb_client.getAxisAngleFromQuaternion(diffQuat)
                curr_pose_err = angle * angle
                diffVel = [
                    simJointInfo[1][0] - kinJointInfo[1][0], 
                    simJointInfo[1][1] - kinJointInfo[1][1],
                    simJointInfo[1][2] - kinJointInfo[1][2]
                ]
                curr_vel_err = diffVel[0] * diffVel[0] + diffVel[1] * diffVel[1] + diffVel[2] * diffVel[2]
            pose_err += w * curr_pose_err
            vel_err += w * curr_vel_err
        

        # Root Control 
        # 包括线速度, 角速度, 平移, 旋转
        # 实际上是用的都是base的信息, 但该urdf root joint和base位于同一点
        # 而且root为fixed, 读出来的信息都是0

        rootPosSim, rootOrnSim = self._pb_client.getBasePositionAndOrientation(self._sim_model)
        rootPosKin, rootOrnKin = self._pb_client.getBasePositionAndOrientation(self._kin_model)
        linVelSim, angVelSim = self._pb_client.getBaseVelocity(self._sim_model)
        rootJointInfo = simJointStates
        # don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
        # see also issue https://github.com/bulletphysics/bullet3/issues/2401 
        linVelKin = self._desiredState.baseLinVel
        angVelKin = self._desiredState.baseAngVel

        root_pos_diff = [
            rootPosSim[0] - rootPosKin[0], rootPosSim[1] - rootPosKin[1], rootPosSim[2] - rootPosKin[2]
        ]
        root_pos_err = root_pos_diff[0] * root_pos_diff[0] + root_pos_diff[1] * root_pos_diff[1] + root_pos_diff[2] * root_pos_diff[2]

        root_rot_diffQuat = self._pb_client.getDifferenceQuaternion(rootOrnSim, rootOrnKin)
        axis, angle = self._pb_client.getAxisAngleFromQuaternion(root_rot_diffQuat)
        root_rot_err = mJointWeights[c.root] * angle * angle

        root_vel_diff = [
            linVelSim[0] - linVelKin[0], linVelSim[1] - linVelKin[1], linVelSim[2] - linVelKin[2]
        ]   
        root_vel_err = root_vel_diff[0] * root_vel_diff[0] + root_vel_diff[1] * root_vel_diff[1] + root_vel_diff[2] * root_vel_diff[2]
        
        root_ang_vel_diff = [
            angVelSim[0] - angVelKin[0], angVelSim[1] - angVelKin[1], angVelSim[2] - angVelKin[2]
        ] 
        root_ang_vel_err = root_ang_vel_diff[0] * root_ang_vel_diff[0] + root_ang_vel_diff[1] * root_ang_vel_diff[1] + root_ang_vel_diff[2] * root_ang_vel_diff[2]
        root_ang_vel_err =  mJointWeights[c.root] * root_ang_vel_err

        root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err
        

        # final cost加权系数
        pose_w = 0.5
        vel_w = 0.05
        root_w = 0.2

        # normalize
        total_w = pose_w + vel_w
        pose_w /= total_w
        vel_w /= total_w
        root_w /= total_w

        # scale系数
        pose_scale = 2
        vel_scale = 0.1
        root_scale = 5


        pose_cost = math.exp(+pose_scale * pose_err)
        vel_cost = math.exp(+vel_scale * vel_err)
        root_cost = math.exp(+root_scale * root_err)

        final_cost = pose_w * pose_cost + vel_w * vel_cost + root_w * root_cost

        return final_cost
