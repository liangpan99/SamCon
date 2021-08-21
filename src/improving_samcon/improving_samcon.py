import time
import json
import math
import numpy as np

from src.samcon.simulation import HumanoidStablePD
from src.samcon.mocapdata import PybulletMocapData, State
from config.humanoid_config import HumanoidConfig as c

INIT_BULLET_STATE_INDEX = 0
END_BULLET_STATE_INDEX = 1
TARGET_STATE_INDEX = 2
SAMPLED_TARGET_STATE_INDEX = 3
SIM_TARGET_STATE_INDEX = 4
COST_INDEX = 5

offset = 0.0 # sample time offset

class ImprovingSamcon():

    def __init__(self, pybullet_client, simTimeStep, sampleTimeStep, savePath):
        self._pb_client = pybullet_client
        self._humanoid = HumanoidStablePD(self._pb_client, simTimeStep)
        self._simTimeStep = simTimeStep
        self._sampleTimeStep = sampleTimeStep
        self._savePath = savePath

    def learn(self, nIter, nSample, nSave, nSaveFinal, dataPath, displayFPS, useFPS):
        
        print('SamCon Start Learn')

        self._mocap_data = PybulletMocapData(dataPath, self._pb_client)
        print('Motion Sequence Cycle Time = ', self._mocap_data.getCycleTime())

        self._nIter = nIter
        self._nSample = nSample
        self._nSave = nSave
        self._nSaveFinal = nSaveFinal

        assert self._nSample % self._nSave == 0
        assert self._nSaveFinal <= self._nSave

        startTime = time.clock()

        """ 
        Data form in SM:
        [initBulletState, endBulletState, targetState, sampledTargetState, simTargetState, cost]
        """
        SM = []

        mean = [0] * c.joint_dof_total
        cov = [c.default_sampling_window] * c.joint_dof_total

        # initialize SM[0]
        SM.append([])
        firstInitState = self._mocap_data.getSpecTimeState(t = 0.0 + offset)
        for i in range(self._nSave):
            SM[0].append([0.0, firstInitState, 0.0, 0.0, 0.0, 0.0])

        for t in range(1, self._nIter + 1):
            
            nInitState = len(SM[t - 1])
            nSampleEachInitState = int(self._nSample / nInitState)

            target_state = self._mocap_data.getSpecTimeState(t * self._sampleTimeStep + offset)
            S = []
            cost_list = []
            for state_set in SM[t - 1]:
                init_bullet_state = state_set[END_BULLET_STATE_INDEX]

                for i in range(nSampleEachInitState):
                    
                    # load last time simulation result
                    # use pybullet API: restoreState to ensure the continuity between two successive simulations 
                    if t == 1:
                        self._humanoid.resetState(init_bullet_state, None)
                    if t != 1:
                        self._pb_client.restoreState(init_bullet_state)

                    sampled_target_state = self.sample(target_state, mean, cov) 
                    self._humanoid.resetState(None, target_state)
                    sim_target_state, cost = self._humanoid.simulation(sampled_target_state, self._sampleTimeStep, displayFPS, useFPS)
                    end_bullet_state = self._pb_client.saveState()
                    S.append([init_bullet_state, end_bullet_state, target_state, sampled_target_state, sim_target_state, cost])
                    
                    cost_list.append(cost)
            
            ## select nSave of the nSample samples to save
            # 1. discard samples lying in the top 40% of the cost distribution
            cost_list = np.array(cost_list)
            cost_order = cost_list.argsort() # sort from small to large
            numDiscard = int(self._nSample * 0.6)
            cost_order = cost_order[0 : numDiscard]
            # 2. find new cost_min, cost_max
            cost_min = cost_list[cost_order[0]]
            cost_max = cost_list[cost_order[-1]]
            # 3. select nSave samples
            indices_list = []
            for x_i in range(self._nSave):
                x = x_i / self._nSave
                cost_target = cost_min + (cost_max - cost_min) * pow(x, 6)
                
                idx = (np.abs(cost_list - cost_target)).argmin()
                indices_list.append(idx)

            SM.append([])
            for index in indices_list:
                SM[t].append(S[index])


            time_so_far = (time.clock() - startTime)
            print('iter: {:d}  time_so_far:  {:.2f}s'.format(t, time_so_far))

        ## select nSaveFinal of nSave paths to save
        # 1. search nSave paths
        # from the element of SM at the last time step of a motion, we backtrack nSave paths
        path_list = []
        for pathId in range(self._nSave):
            path = []
            path.append(pathId)

            currTime_initBulletState = SM[-1][pathId][INIT_BULLET_STATE_INDEX]
            for i in range(self._nIter - 1): # only need to compare nIter-1 times
                lastTime = -2 - i
                for j in range(self._nSave):
                    lastTime_endBulletState = SM[lastTime][j][END_BULLET_STATE_INDEX]

                    # if the current time's init state equal to the last time's end state,
                    # it proves that two samples belong to the same path.
                    if lastTime_endBulletState == currTime_initBulletState:
                        path.insert(0, j)
                        currTime_initBulletState = SM[lastTime][j][INIT_BULLET_STATE_INDEX]
                        break
                    
            path_list.append(path)

        # 2. calculate the total cost of each path
        total_cost_list = []
        for pathId in range(self._nSave):
            total_cost = 0
            for t in range(1, self._nIter + 1):
                savedSample_index = path_list[pathId][t - 1]
                total_cost += SM[t][savedSample_index][COST_INDEX]
            total_cost_list.append(total_cost)
        total_cost_list = np.array(total_cost_list)

        # 3. save nSaveFinal paths with minimum total cost
        order = total_cost_list.argsort() 
        order = order[0 : self._nSaveFinal]
        self.save(SM, path_list, total_cost_list, order)

        print('SamCon End Learn')
        exit()


    def test(self, dataPath, displayFPS, useFPS):

        print('SamCon Start Test')

        # load txt file
        with open(dataPath, 'r') as f:
            data = json.load(f)

        nIter = data['info']['nIter']
        nSaveFinal = data['info']['nSaveFinal']
        sampleTimeStep = data['info']['sampleTimeStep']
        simTimeStep = data['info']['simulationTimeStep']

        for pathId in range(nSaveFinal):
            key = f'path_{pathId}'
            path_cost = data[key]['cost']
            t0_state = data[key]['t0_state']
            reference_states = data[key]['reference_states']
            sampled_states = data[key]['sampled_states']
            simulated_states = data[key]['simulated_states']

            assert len(reference_states) == nIter
            assert len(sampled_states) == nIter
            assert len(simulated_states) == nIter

            self._humanoid.resetState(start_state = self.getStateFromList(t0_state), end_state = None)
            
            for t in range(1, nIter + 1):
                print('iter:  ', t)
                reference_state = reference_states[t - 1]
                reference_state = self.getStateFromList(reference_state)

                sampled_state = sampled_states[t - 1]
                sampled_state = self.getStateFromList(sampled_state)

                simulated_state = simulated_states[t - 1]
                simulated_state = self.getStateFromList(simulated_state)

                self._humanoid.resetState(start_state = None, end_state = reference_state)
                state, _ = self._humanoid.simulation(sampled_state, sampleTimeStep, displayFPS, useFPS)

                state_1 = self.getListFromNamedtuple(state)
                state_2 = self.getListFromNamedtuple(simulated_state)
                state_1 = np.array(state_1)
                state_2 = np.array(state_2)
                print('error: ', (state_1-state_2).sum())
                # time.sleep(0.5)
    
    def save(self, SM, path_list, pathCost_list, savedIndices):
        """ save as txt file """
        assert savedIndices.shape[0] == self._nSaveFinal
        assert len(path_list) == self._nSave
        assert len(pathCost_list) == self._nSave

        data = {
            
        }

        info = {
            'dataset_path': self._mocap_data.DataPath(),
            'nIter': self._nIter,
            'nSample': self._nSample,
            'nSave': self._nSave,
            'nSaveFinal': self._nSaveFinal,
            'sampleTimeStep': self._sampleTimeStep,
            'simulationTimeStep': self._simTimeStep
        }
        data['info'] = info

        for i in range(self._nSaveFinal):
            pathId = savedIndices[i]
            
            # save the init state at t=0
            t0_state = SM[0][0][END_BULLET_STATE_INDEX]
            t0_state = self.getListFromNamedtuple(t0_state)

            # save the other states after t=0
            reference_states = []
            sampled_states = []
            simulated_states = []
            for t in range(1, self._nIter + 1):
                savedSample_index = path_list[pathId][t - 1]
                reference_state = SM[t][savedSample_index][TARGET_STATE_INDEX]
                sampled_state = SM[t][savedSample_index][SAMPLED_TARGET_STATE_INDEX]
                simulated_state = SM[t][savedSample_index][SIM_TARGET_STATE_INDEX]

                reference_state = self.getListFromNamedtuple(reference_state)
                sampled_state = self.getListFromNamedtuple(sampled_state)
                simulated_state = self.getListFromNamedtuple(simulated_state)

                reference_states.append(reference_state)
                sampled_states.append(sampled_state)
                simulated_states.append(simulated_state)


            dataUnit = {
                'cost': pathCost_list[pathId],
                't0_state': t0_state,
                'reference_states': reference_states,
                'sampled_states': sampled_states,
                'simulated_states': simulated_states,
            }

            data[f'path_{i}'] = dataUnit
    
        f = open(self._savePath, 'w')
        json.dump(data, f)
        f.close()
        print(f'file saved! [{self._savePath}]')
                
    
    def genModifiedQuaternion(self, quaternion, diff):
        if len(quaternion) == 4:
            euler = self._pb_client.getEulerFromQuaternion(quaternion)
            eulerDiff = (
                euler[0] + diff[0], 
                euler[1] + diff[1], 
                euler[2] + diff[2]
            )
            quaternionDiff = self._pb_client.getQuaternionFromEuler(eulerDiff)
    
        if len(quaternion) == 1:
            quaternionDiff = [diff[0] + quaternion[0]]

        return quaternionDiff

    def sample(self, state, mean, cov):
        """ Perform random sampling on rotation of all movable joints to produce a new pose. 
        
        Args:
            state
            mean -- list
            cov  -- list
        """

        assert len(mean) == c.joint_dof_total
        assert len(cov) == c.joint_dof_total

        cov_mat = np.diag(cov)
        errors = np.random.multivariate_normal(mean, cov_mat, size=(1))

        modified_state = state._replace(
            basePos = [0, 0, 0],
            baseOrn = [0, 0, 0, 1],

            chestRot=self.genModifiedQuaternion(state.chestRot, errors[0, 0 : 3]),
            neckRot=self.genModifiedQuaternion(state.neckRot, errors[0, 3 : 6]),
            rightHipRot=self.genModifiedQuaternion(state.rightHipRot, errors[0, 6 : 9]),
            rightKneeRot=self.genModifiedQuaternion(state.rightKneeRot, errors[0, 9 : 10]),
            rightAnkleRot=self.genModifiedQuaternion(state.rightAnkleRot, errors[0, 10 : 13]),
            rightShoulderRot=self.genModifiedQuaternion(state.rightShoulderRot, errors[0, 13 : 16]),
            rightElbowRot=self.genModifiedQuaternion(state.rightElbowRot, errors[0, 16 : 17]),
            leftHipRot=self.genModifiedQuaternion(state.leftHipRot, errors[0, 17 : 20]),
            leftKneeRot=self.genModifiedQuaternion(state.leftKneeRot, errors[0, 20 : 21]),
            leftAnkleRot=self.genModifiedQuaternion(state.leftAnkleRot, errors[0, 21 : 24]),
            leftShoulderRot=self.genModifiedQuaternion(state.leftShoulderRot, errors[0, 24 : 27]),
            leftElbowRot=self.genModifiedQuaternion(state.leftElbowRot, errors[0, 27 : 28]),

            baseLinVel = [0, 0, 0],
            baseAngVel = [0, 0, 0],
            chestVel = [0, 0, 0], 
            neckVel = [0, 0, 0], 
            rightHipVel = [0, 0, 0], 
            rightKneeVel = [0], 
            rightAnkleVel = [0, 0, 0], 
            rightShoulderVel = [0, 0, 0], 
            rightElbowVel = [0], 
            leftHipVel = [0, 0, 0], 
            leftKneeVel = [0], 
            leftAnkleVel = [0, 0, 0], 
            leftShoulderVel = [0, 0, 0], 
            leftElbowVel = [0],
        )

        return modified_state

    def getListFromNamedtuple(self, namedtuple, is_pose=True, is_velocity=True):
        """ Type transition: from collections.namedtuple to list.
        
        Return:
            pose len=43
            velocity len=34
            pose+velocity len=77
        """

        if is_pose:
            pose = list(namedtuple.basePos) + list(namedtuple.baseOrn) \
                 + list(namedtuple.chestRot) + list(namedtuple.neckRot) \
                 + list(namedtuple.rightHipRot) + list(namedtuple.rightKneeRot) + list(namedtuple.rightAnkleRot) \
                 + list(namedtuple.rightShoulderRot) + list(namedtuple.rightElbowRot) \
                 + list(namedtuple.leftHipRot) + list(namedtuple.leftKneeRot) + list(namedtuple.leftAnkleRot) \
                 + list(namedtuple.leftShoulderRot) + list(namedtuple.leftElbowRot)
        else:
            pose = []

        if is_velocity:
            velocity = list(namedtuple.baseLinVel) + list(namedtuple.baseAngVel) \
                 + list(namedtuple.chestVel) + list(namedtuple.neckVel) \
                 + list(namedtuple.rightHipVel) + list(namedtuple.rightKneeVel) + list(namedtuple.rightAnkleVel) \
                 + list(namedtuple.rightShoulderVel) + list(namedtuple.rightElbowVel) \
                 + list(namedtuple.leftHipVel) + list(namedtuple.leftKneeVel) + list(namedtuple.leftAnkleVel) \
                 + list(namedtuple.leftShoulderVel) + list(namedtuple.leftElbowVel)
        else:
            velocity = []
        
        return pose + velocity

    def getStateFromList(self, data):
        """ Type transition: from list to collections.namedtuple. """

        state = State(
        # Position
        basePos = data[0 : 3],
        baseOrn = data[3 : 7],
        chestRot = data[7 : 11], 
        neckRot = data[11 : 15], 
        rightHipRot = data[15 : 19], 
        rightKneeRot = data[19 : 20], 
        rightAnkleRot = data[20 : 24], 
        rightShoulderRot = data[24 : 28], 
        rightElbowRot = data[28 : 29], 
        leftHipRot = data[29 : 33], 
        leftKneeRot = data[33 : 34], 
        leftAnkleRot = data[34 : 38], 
        leftShoulderRot = data[38 : 42], 
        leftElbowRot = data[42 : 43],

        # Velocity
        baseLinVel = data[43 : 46],
        baseAngVel = data[46 : 49],
        chestVel = data[49 : 52], 
        neckVel = data[52 : 55], 
        rightHipVel = data[55 : 58], 
        rightKneeVel = data[58 : 59], 
        rightAnkleVel = data[59 : 62], 
        rightShoulderVel = data[62 : 65], 
        rightElbowVel = data[65 : 66], 
        leftHipVel = data[66 : 69], 
        leftKneeVel = data[69 : 70], 
        leftAnkleVel = data[70 : 73], 
        leftShoulderVel = data[73 : 76], 
        leftElbowVel = data[76 : 77],
        )
        
        return state
