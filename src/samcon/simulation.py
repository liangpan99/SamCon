import time
import math
import numpy as np
from pybullet_utils import pd_controller_stable

from config.humanoid_config import HumanoidConfig as c

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

        # Load humanoid.urdf
        char_create_flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        self_collision = True
        if self_collision:
            char_create_flags = char_create_flags|\
                                self._pb_client.URDF_USE_SELF_COLLISION|\
                                self._pb_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self._sim_model = self._pb_client.loadURDF(
            fileName = c.fileName,
            basePosition = c.basePos,
            globalScaling = c.globalScale,
            useFixedBase = False,
            flags = char_create_flags
        )

        self._kin_model = self._pb_client.loadURDF(
            fileName = c.fileName,
            basePosition = c.basePos,
            globalScaling = c.globalScale,
            useFixedBase = True,
            flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        )

        ## Set parameters
        # self._sim_model: Friction, Motor
        self._pb_client.changeDynamics(self._sim_model, -1, linearDamping = 0, angularDamping = 0)
        self._pb_client.changeDynamics(self._sim_model, -1, lateralFriction = 0.9)
        for i in range(self._pb_client.getNumJoints(self._sim_model)):
            self._pb_client.changeDynamics(self._sim_model, i, lateralFriction = 0.9)
        
        for j in self._jointIndicesAll:
            self._pb_client.setJointMotorControl2(
                self._sim_model,
                j,
                self._pb_client.POSITION_CONTROL,
                targetPosition = 0,
                positionGain = 0,
                targetVelocity = 0,
                force= 0 )
            self._pb_client.setJointMotorControlMultiDof(
                self._sim_model,
                j,
                self._pb_client.POSITION_CONTROL,
                targetPosition = [0, 0, 0, 1],
                targetVelocity = [0, 0, 0],
                positionGain = 0,
                velocityGain = 1,
                force = [0, 0, 0]
            )
        
        # self._kin_model: allow collision, transparency
        self._pb_client.changeDynamics(self._kin_model, -1, linearDamping = 0, angularDamping = 0)
        alpha = 0.7
        for j in range(-1, self._pb_client.getNumJoints(self._kin_model)):
            self._pb_client.changeVisualShape(self._kin_model, j, rgbaColor = [1, 1, 1, alpha])
            self._pb_client.setCollisionFilterGroupMask(
                self._kin_model,
                j,
                collisionFilterGroup = 0,
                collisionFilterMask = 0)
            self._pb_client.changeDynamics(
                self._kin_model,
                j,
                activationState = self._pb_client.ACTIVATION_STATE_SLEEP +
                                self._pb_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                                self._pb_client.ACTIVATION_STATE_DISABLE_WAKEUP)

    def initPose(self, state, phys_model, is_initBase, is_initVel):
        """ Reset humanoid's position and velocity

        Args:
            state -- len=77 list
            phys_model -- int
            is_initBase -- bool
            is_initVel -- bool
        
        """
        if is_initVel:
            if is_initBase:
                self._pb_client.resetBasePositionAndOrientation(phys_model, state[0:3], state[3:7])
                self._pb_client.resetBaseVelocity(phys_model, state[43:46], state[46:49])
            
            indices = self._jointIndicesAll

            jointPositions = [state[7:11], state[11:15], 
                              state[15:19], state[19:20], state[20:24], 
                              state[24:28], state[28:29], 
                              state[29:33], state[33:34], state[34:38], 
                              state[38:42], state[42:43]]
            jointVelocities = [state[49:52], state[52:55], 
                               state[55:58], state[58:59], state[59:62], 
                               state[62:65], state[65:66], 
                               state[66:69], state[69:70], state[70:73], 
                               state[73:76], state[76:77]]
            
            self._pb_client.resetJointStatesMultiDof(phys_model, indices,
                                            jointPositions, jointVelocities)
        else:
            if is_initBase:
                self._pb_client.resetBasePositionAndOrientation(phys_model, state[0:3], state[3:7])
            
            indices = self._jointIndicesAll

            jointPositions = [state[7:11], state[11:15], 
                              state[15:19], state[19:20], state[20:24], 
                              state[24:28], state[28:29], 
                              state[29:33], state[33:34], state[34:38], 
                              state[38:42], state[42:43]]
            
            self._pb_client.resetJointStatesMultiDof(phys_model, indices,
                                            jointPositions)


    def resetState(self, start_state, end_state):
        """ Reset state (state: position + velocity) """
        if end_state != None:
            self._desiredState = end_state

            # Use end_state to reset the kinematic humanoid
            self.initPose(end_state, self._kin_model, is_initBase=True, is_initVel=True)

        if start_state != None:
            # Use start_state to reset the simulated humanoid
            self.initPose(start_state, self._sim_model, is_initBase=True, is_initVel=True)


    def computePDForces(self, desiredPositions, desiredVelocities, maxForces):
        """ Compute torques from the stable PD controller. """

        if desiredVelocities == None:
            desiredVelocities = [0] * self._totalDofs
        
        taus = self._stablePD.computePD(bodyUniqueId = self._sim_model,
                                        jointIndices = self._jointIndicesAll,
                                        desiredPositions = desiredPositions,
                                        desiredVelocities = desiredVelocities,
                                        kps = self._kpOrg,
                                        kds = self._kdOrg,
                                        maxForces = maxForces,
                                        timeStep = self._simTimeStep)
        return taus
        
    def applyPDForces(self, taus):
        """ Apply pre-computed torques """
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
                                                        forces = forces)


    def simulation(self, desiredPosition, sampleTimeStep, displayFPS, useFPS):
        """ Execute simulation
        
        Args:
            desiredPosition -- len=77 list
            sampleTimeStep -- int -- for calculate simulation times
            displayFPS -- int
            useFPS -- bool

        Returns:
            simulatedState -- collections.namedtuple State
            cost -- int
        """

        # Base position cannot be controlled, so we need 7 zeros
        desiredPosition = [0, 0, 0, 0, 0, 0, 0] + desiredPosition[7:43]
        
        numSim = int(sampleTimeStep / self._simTimeStep)
        for i in range(numSim):
            taus = self.computePDForces(desiredPosition, desiredVelocities = None, maxForces = c.maxForces)
            self.applyPDForces(taus)
            self._pb_client.stepSimulation()
            if useFPS:
                time.sleep(1./displayFPS) # display FPS


        # Get simulatedState
        sim_basePos, sim_baseOrn = self._pb_client.getBasePositionAndOrientation(self._sim_model) 
        sim_baseLinVel, sim_baseAngVel = self._pb_client.getBaseVelocity(self._sim_model)
        sim_jointStates = self._pb_client.getJointStatesMultiDof(self._sim_model, c.linkIndicesAll)

        simulatedState = []

        # Base pos orn len=7
        simulatedState.extend(list(sim_basePos))
        simulatedState.extend(list(sim_baseOrn))

        # Joint rotation len=8*4+4*1=36
        for i in c.controllableJointIndicesAll:
            simulatedState.extend(list(sim_jointStates[i][0]))
        
        # Base lin ang vel len=6
        simulatedState.extend(list(sim_baseLinVel))
        simulatedState.extend(list(sim_baseAngVel))

        # Joint ang vel len=8*3+4*1=28
        for i in c.controllableJointIndicesAll:
            simulatedState.extend(list(sim_jointStates[i][1]))

        cost = self.computeCost()
        
        return simulatedState, cost

    def computeCost(self):
        """ Compute cost between kinematic humanoid and simulated humanoid.
        
        including 4 components:
            1. controllable joints
            2. root
            3. end-effector
            4. balance
        """

        pose_weight = c.pose_weight
        root_weight = c.root_weight
        end_effector_weight = c.end_effector_weight
        balance_weight = c.balance_weight

        pose_cost = self.computePoseCost(pose_weight)
        root_cost = self.computeRootCost(root_weight)
        end_effector_cost = self.computeEndEffectorCost(end_effector_weight)
        balance_cost = self.ComputeBalanceCost(balance_weight)

        final_cost = pose_cost + root_cost + end_effector_cost + balance_cost

        # print(f'pose_cost={pose_cost}  root_cost={root_cost}  end_effector_cost={end_effector_cost}  balance_cost={balance_cost}')

        return final_cost

    def computeCOMposVel(self, uid: int):
        """Compute center-of-mass position and velocity."""
        pb = self._pb_client
        num_joints = 15
        jointIndices = range(num_joints)
        link_states = pb.getLinkStates(uid, jointIndices, computeLinkVelocity = 1)
        link_pos = np.array([s[0] for s in link_states])
        link_vel = np.array([s[-2] for s in link_states])
        tot_mass = 0.
        masses = []
        for j in jointIndices:
            mass_, *_ = pb.getDynamicsInfo(uid, j)
            masses.append(mass_)
            tot_mass += mass_
        masses = np.asarray(masses)[:, None]
        com_pos = np.sum(masses * link_pos, axis = 0) / tot_mass
        com_vel = np.sum(masses * link_vel, axis = 0) / tot_mass
        return com_pos, com_vel

    def ComputeBalanceCost(self, weight):

        error = 0.0

        end_effectors = [c.rightAnkle, c.rightWrist, c.leftAnkle, c.leftWrist]
        num = len(end_effectors)

        simLinkStates = self._pb_client.getLinkStates(self._sim_model, c.linkIndicesAll)
        kinLinkStates = self._pb_client.getLinkStates(self._kin_model, c.linkIndicesAll)

        sim_com_pos, _ = self.computeCOMposVel(self._sim_model)
        kin_com_pos, _ = self.computeCOMposVel(self._kin_model)

        # only compute position, 
        # because we cannot read the link velocity of kinematic humanoid
        for index in end_effectors:
            sim_link_state = simLinkStates[index]
            kin_link_state = kinLinkStates[index]

            sim_link_pos = sim_link_state[0] # in world frame
            kin_link_pos = kin_link_state[0]

            # calculate relative coordinates in xz plane
            sim_link_posRel = [
                sim_com_pos[0] - sim_link_pos[0],
                0.0,
                sim_com_pos[2] - sim_link_pos[2],
            ]

            kin_link_posRel = [
                kin_com_pos[0] - kin_link_pos[0],
                0.0,
                kin_com_pos[2] - kin_link_pos[2],
            ]

            diff = [
                sim_link_posRel[0] - kin_link_posRel[0],
                sim_link_posRel[1] - kin_link_posRel[1],
                sim_link_posRel[2] - kin_link_posRel[2],
            ]

            error += math.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])

        if num > 0:
            error /= num
        
        return weight * error

    def computePoseCost(self, weight):
        """ Compute all controllable joints' rotation and angle velocity. """
        
        pose_err = 0.0
        vel_err = 0.0

        # We usually just use wi=1, and just the weights to produce more motion variants.
        # plz see the SamCon paper
        mJointWeights = [
            0.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0
        ]
        assert len(mJointWeights) == len(c.linkIndicesAll)

        # only compute controllable joints, 
        # because all information of fixed joint is 0.
        jointIndicesControllable = c.controllableJointIndicesAll
        num = len(jointIndicesControllable)

        simJointStates = self._pb_client.getJointStatesMultiDof(self._sim_model, c.linkIndicesAll)
        kinJointStates = self._pb_client.getJointStatesMultiDof(self._kin_model, c.linkIndicesAll)
        
        for j in jointIndicesControllable:
            curr_pose_err = 0
            curr_vel_err = 0
            w = mJointWeights[j]
            simJointInfo = simJointStates[j]
            kinJointInfo = kinJointStates[j]

            if len(simJointInfo[0]) == 1: # joint type: revolute 1-Dof
                angle = simJointInfo[0][0] - kinJointInfo[0][0]
                curr_pose_err = math.sqrt(angle * angle)
                velDiff = simJointInfo[1][0] - kinJointInfo[1][0]
                curr_vel_err = math.sqrt(velDiff * velDiff)
            if len(simJointInfo[0]) == 4: # joint type: spherical 3-Dof
                diffQuat = self._pb_client.getDifferenceQuaternion(simJointInfo[0], kinJointInfo[0])
                axis, angle = self._pb_client.getAxisAngleFromQuaternion(diffQuat)
                curr_pose_err = math.sqrt(angle * angle)
                diffVel = [
                    simJointInfo[1][0] - kinJointInfo[1][0], 
                    simJointInfo[1][1] - kinJointInfo[1][1],
                    simJointInfo[1][2] - kinJointInfo[1][2]
                ]
                curr_vel_err = math.sqrt(diffVel[0] * diffVel[0] + diffVel[1] * diffVel[1] + diffVel[2] * diffVel[2])
            pose_err += w * curr_pose_err
            vel_err += 0.1 * curr_vel_err

        error = (pose_err + vel_err) / num
        
        return weight * error

    def computeRootCost(self, weight):
        """ Root Orientation and Angle Velocity """

        rot_error = 0.0
        angVel_error = 0.0

        sim_base_pos, sim_base_orn = self._pb_client.getBasePositionAndOrientation(self._sim_model)
        kin_base_pos, kin_base_orn = self._pb_client.getBasePositionAndOrientation(self._kin_model)
        sim_base_linVel, sim_base_angVel = self._pb_client.getBaseVelocity(self._sim_model)
        # don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
        # see also issue https://github.com/bulletphysics/bullet3/issues/2401
        kin_base_linVel = self._desiredState[43:46]
        kin_base_angVel = self._desiredState[46:49]

        diffQuat = self._pb_client.getDifferenceQuaternion(sim_base_orn, kin_base_orn)
        axis, angle = self._pb_client.getAxisAngleFromQuaternion(diffQuat)
        rot_error += math.sqrt(angle * angle)

        diffAngVel = [
            sim_base_angVel[0] - kin_base_angVel[0],
            sim_base_angVel[1] - kin_base_angVel[1],
            sim_base_angVel[2] - kin_base_angVel[2]
        ]

        angVel_error += math.sqrt(
            diffAngVel[0] * diffAngVel[0] + diffAngVel[1] * diffAngVel[1] + diffAngVel[2] * diffAngVel[2]
        )

        return weight * (rot_error + 0.1 * angVel_error)
        


    def computeEndEffectorCost(self, weight):
        """ End-effector Cost """

        error = 0.0

        end_effectors = [c.rightAnkle, c.rightWrist, c.leftAnkle, c.leftWrist]
        num = len(end_effectors)

        simLinkStates = self._pb_client.getLinkStates(self._sim_model, c.linkIndicesAll)
        kinLinkStates = self._pb_client.getLinkStates(self._kin_model, c.linkIndicesAll)

        for index in end_effectors:
            sim_link_state = simLinkStates[index]
            kin_link_state = kinLinkStates[index]

            sim_link_pos = sim_link_state[0] # in world frame
            kin_link_pos = kin_link_state[0]

            diff = abs(sim_link_pos[1] - kin_link_pos[1]) # get y-axis error
            error += diff

        if num > 0:
            error /= num
        
        return weight * error
