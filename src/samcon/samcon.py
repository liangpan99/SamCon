import time
import json
import math
import numpy as np
from numpy.core.einsumfunc import _parse_possible_contraction

from src.samcon.simulation import HumanoidStablePD
from src.samcon.mocapdata import PybulletMocapData
from config.humanoid_config import HumanoidConfig as c

INIT_BULLET_STATE_INDEX = 0
END_BULLET_STATE_INDEX = 1
TARGET_STATE_INDEX = 2
COMPENSATED_TARGET_STATE_INDEX = 3
SAMPLED_TARGET_STATE_INDEX = 4
SIM_TARGET_STATE_INDEX = 5
COST_INDEX = 6

time_offset = 0.0 # sample time offset

class Samcon():

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
        [initBulletState, endBulletState, targetState, compensatedTargetState, sampledTargetState, simTargetState, cost]
        """
        SM = []

        # initialize SM[0]
        SM.append([])
        firstInitState = self._mocap_data.getSpecTimeState(t = 0.0 + time_offset)
        for i in range(self._nSave):
            SM[0].append([0.0, firstInitState, 0.0, 0.0, 0.0, 0.0, 0.0])

        for t in range(1, self._nIter + 1):
            
            nInitState = len(SM[t - 1])
            nSampleEachInitState = int(self._nSample / nInitState)

            target_state = self._mocap_data.getSpecTimeState(t * self._sampleTimeStep + time_offset)
            S = []
            cost_list = []
            for state_set in SM[t - 1]:
                init_bullet_state = state_set[END_BULLET_STATE_INDEX]

                for i in range(nSampleEachInitState):
                    
                    # load last time simulation result
                    # use pybullet API: restoreState to ensure the continuity between two successive simulations 
                    if t == 1:
                        self._humanoid.resetState(init_bullet_state, None)
                        temp_state = self._pb_client.saveState()
                    if t != 1:
                        self._pb_client.restoreState(init_bullet_state)

                    # add feedforward offsets to target_state
                    compensated_target_state = target_state[:] # create a new list
                    self._humanoid.resetState(None, target_state)
                    feedforward_offsets, _ = self._humanoid.simulation(target_state, self._sampleTimeStep, displayFPS, useFPS)
                    start_idx = 7
                    for z in range(len(c.samplingWindow)):
                        if len(c.samplingWindow[z]) == 3:
                            quat = target_state[start_idx:start_idx+4]
                            euler = self._pb_client.getEulerFromQuaternion(quat)
                            offset = feedforward_offsets[start_idx:start_idx+4]
                            offset = self._pb_client.getEulerFromQuaternion(offset)

                            compensated_euler = (
                                euler[0] + euler[0] - offset[0],
                                euler[1] + euler[1] - offset[1],
                                euler[2] + euler[2] - offset[2]
                            )
                            compensated_quat = self._pb_client.getQuaternionFromEuler(compensated_euler)
                            
                            compensated_target_state[start_idx+0] = compensated_quat[0]
                            compensated_target_state[start_idx+1] = compensated_quat[1]
                            compensated_target_state[start_idx+2] = compensated_quat[2]
                            compensated_target_state[start_idx+3] = compensated_quat[3]

                            start_idx += 4

                        elif len(c.samplingWindow[z]) == 1:
                            euler = target_state[start_idx:start_idx+1]
                            offset = feedforward_offsets[start_idx:start_idx+1]

                            compensated_euler = euler[0] + euler[0] - offset[0]
                            compensated_target_state[start_idx] = compensated_euler
                            start_idx += 1

                        else:
                            raise RuntimeError('wrong samplingwindow')

                    # revise base pos&orn for visulization
                    for z in range(7):
                        compensated_target_state[z] = target_state[z]

                    if t == 1:
                        self._pb_client.restoreState(temp_state)
                    if t != 1:
                        self._pb_client.restoreState(init_bullet_state)

                    sampled_target_state = self.sample(compensated_target_state) # sampling based on the compensated target
                    self._humanoid.resetState(None, target_state) # still use target_state when calc cost
                    sim_target_state, cost = self._humanoid.simulation(sampled_target_state, self._sampleTimeStep, displayFPS, useFPS)
                    end_bullet_state = self._pb_client.saveState()
                    S.append([init_bullet_state, end_bullet_state, target_state, compensated_target_state, sampled_target_state, sim_target_state, cost])
                    
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
            compensated_reference_states = data[key]['compensated_reference_states']
            sampled_states = data[key]['sampled_states']
            simulated_states = data[key]['simulated_states']

            assert len(reference_states) == nIter
            assert len(compensated_reference_states) == nIter
            assert len(sampled_states) == nIter
            assert len(simulated_states) == nIter

            self._humanoid.resetState(start_state = t0_state, end_state = None)
            
            for t in range(1, nIter + 1):
                print('iter:  ', t)
                reference_state = reference_states[t - 1]
                compensated_reference_state = compensated_reference_states[t - 1]
                sampled_state = sampled_states[t - 1]
                simulated_state = simulated_states[t - 1]

                self._humanoid.resetState(start_state = None, end_state = reference_state)
                state, _ = self._humanoid.simulation(sampled_state, sampleTimeStep, displayFPS, useFPS)

                state_1 = np.array(state)
                state_2 = np.array(simulated_state)
                print('error: ', np.abs(state_1-state_2).sum())
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

            # save the other states after t=0
            reference_states = []
            compensated_reference_states = []
            sampled_states = []
            simulated_states = []
            for t in range(1, self._nIter + 1):
                savedSample_index = path_list[pathId][t - 1]
                reference_state = SM[t][savedSample_index][TARGET_STATE_INDEX]
                compensated_reference_state = SM[t][savedSample_index][COMPENSATED_TARGET_STATE_INDEX]
                sampled_state = SM[t][savedSample_index][SAMPLED_TARGET_STATE_INDEX]
                simulated_state = SM[t][savedSample_index][SIM_TARGET_STATE_INDEX]

                reference_states.append(reference_state)
                compensated_reference_states.append(compensated_reference_state)
                sampled_states.append(sampled_state)
                simulated_states.append(simulated_state)


            dataUnit = {
                'cost': pathCost_list[pathId],
                't0_state': t0_state,
                'reference_states': reference_states,
                'compensated_reference_states': compensated_reference_states,
                'sampled_states': sampled_states,
                'simulated_states': simulated_states,
            }

            data[f'path_{i}'] = dataUnit
    
        f = open(self._savePath, 'w')
        json.dump(data, f)
        f.close()
        print(f'file saved! [{self._savePath}]')
                

    def genRandomFromUniform(self, samplingWindow):
        """ Generate random number from uniform distribution. """
        diff = []
        for i in samplingWindow:
            diff.append(np.random.uniform(low = -i, high = i, size = (1)))
        return diff

    def sample(self, state):
        """ Perform random sampling on rotation of all movable joints to produce a new pose. """

        state = np.array(state, dtype=np.float32)
        state[43:77] = 0

        start_idx = 7
        for i in range(len(c.samplingWindow)):

            if len(c.samplingWindow[i]) == 3:
                quat = state[start_idx:start_idx+4]
                euler = self._pb_client.getEulerFromQuaternion(quat)
                delta = self.genRandomFromUniform(c.samplingWindow[i])

                sampled_euler = (
                euler[0] + delta[0],
                euler[1] + delta[1],
                euler[2] + delta[2]
                )

                sampled_quat = self._pb_client.getQuaternionFromEuler(sampled_euler)
                state[start_idx:start_idx+4] = list(sampled_quat)

                start_idx += 4

            elif len(c.samplingWindow[i]) == 1:
                euler = state[start_idx:start_idx+1]
                delta = self.genRandomFromUniform(c.samplingWindow[i])

                sampled_euler = euler[0] + delta[0]
                state[start_idx:start_idx+1] = sampled_euler
                
                start_idx += 1
            
            else:
                raise RuntimeError('wrong samplingwindow!')

        return state.tolist()
