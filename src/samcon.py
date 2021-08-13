import json
import numpy as np
from numpy.lib.npyio import save

from src.simulation import HumanoidStablePD
from src.mocapdata import PybulletMocapData
from config.humanoid_config import HumanoidConfig as c

class SamCon():

    def __init__(self, pybullet_client, simTimeStep, sampleTimeStep, savePath):
        self._pb_client = pybullet_client
        self._humanoid = HumanoidStablePD(self._pb_client, simTimeStep)
        self._simTimeStep = simTimeStep
        self._sampleTimeStep = sampleTimeStep
        self._savePath = savePath


    def learn(self, nIter, nSample, nSave, nSaveFinal, dataPath, displayFPS):

        self._mocap_data = PybulletMocapData(dataPath, self._pb_client)

        self._nIter = nIter
        self._nSample = nSample
        self._nSave = nSave
        self._nSaveFinal = nSaveFinal

        assert self._nSample % self._nSave == 0

        SM = [[]]
        offset = 0.0
        # initialize SM[0]
        for i in range(self._nSave):
            SM[0].append([self._mocap_data.getSpecTimeState(t=0.0 + offset), 0.0])


        for t in range(1, self._nIter+1):

            nInitState = len(SM[t-1])
            nSampleEachInitState = int(self._nSample / nInitState)

            target_state = self._mocap_data.getSpecTimeState(t*self._sampleTimeStep+offset)
            SM_element = []
            for init_state in SM[t-1]:
                init_state = init_state[0]
                S = []
                for i in range(nSampleEachInitState):

                    desiredPosition = self.sample(target_state) # 只修改pose
                    self._humanoid.resetState(init_state, target_state)
                    sim_target_state, cost = self._humanoid.simulation(False, desiredPosition, self._sampleTimeStep, displayFPS)
                    S.append([sim_target_state, cost])
                
                # 从该init_state对应的nSampleEachInitState挑nSave个最小的保存到SM中
                cost_min = 99999
                cost_min_index = 99999
                for i in range(nSampleEachInitState):
                    cost_curr = S[i][1]
                    if cost_curr < cost_min:
                        cost_min = cost_curr
                        cost_min_index = i
                SM_element.append(S[cost_min_index])
            SM.append(SM_element)
        
        # 对SM的nSave个path进行遍历，选择nSaveFinal个cost最小的作为最终的结果
        total_cost_list = []
        for i in range(self._nSave):
            
            total_cost = 0
            for j in range(self._nIter+1):
                total_cost += SM[j][i][1]
            total_cost_list.append(total_cost)
        
        total_cost_list = np.array(total_cost_list)
        order = total_cost_list.argsort() # 按小到大排序 只返回索引

        order = order[0:self._nSaveFinal] # 取前nSaveFinal个最小的
        
        # 保存
        self.save(SM, total_cost_list, order)
        exit()

    def test(self, dataPath, displayFPS):
        
        # 加载txt文件
        with open(dataPath, 'r') as f:
            data = json.load(f)

        datasetPath = data['info']['dataset_path']
        mocap_data = PybulletMocapData(datasetPath, self._pb_client)

        nIter = data['info']['nIter']
        nSaveFinal = data['info']['nSaveFinal']
        sampleTimeStep = data['info']['sampleTimeStep']
        simTimeStep = data['info']['simulationTimeStep']

        for pathId in range(nSaveFinal):
            
            total_cost = data[f'path_{pathId}']['cost']
            frames = data[f'path_{pathId}']['frames']
            numFrames = len(frames)
            assert numFrames-1 == nIter

            init_state = mocap_data.getSpecTimeState(t=0)
            self._humanoid.resetState(start_state=init_state, end_state=None)

            for t in range(1, nIter+1):
                target_state = mocap_data.getSpecTimeState(t*sampleTimeStep)
                self._humanoid.resetState(start_state=None, end_state=target_state)
                self._humanoid.simulation(True, frames[t], sampleTimeStep, displayFPS)

        

    def genRandomFromUniform(self, samplingWindow):
        """ 从均匀分布中生成随机数, 返回一个和samplingWindow大小一样的列表 """
        diff = []
        for i in samplingWindow:
            diff.append(np.random.uniform(low=-i, high=i, size=(1)))
        return diff
    
    def genModifiedQuaternion(self, quaternion, samplingWindow):
        if len(quaternion) == 4:

            euler = self._pb_client.getEulerFromQuaternion(quaternion)
            diff = self.genRandomFromUniform(samplingWindow)
            eulerDiff = (
                euler[0] + diff[0], 
                euler[1] + diff[1], 
                euler[2] + diff[2]
            )
            quaternionDiff = self._pb_client.getQuaternionFromEuler(eulerDiff)
    
        if len(quaternion) == 1:
            diff = self.genRandomFromUniform(samplingWindow)
            quaternionDiff = diff[0] + quaternion[0]

        return quaternionDiff

    def sample(self, state):

        modified_state = state._replace(
            chestRot=self.genModifiedQuaternion(state.chestRot, c.samplingWindow[0]),
            neckRot=self.genModifiedQuaternion(state.neckRot, c.samplingWindow[1]),
            rightHipRot=self.genModifiedQuaternion(state.rightHipRot, c.samplingWindow[2]),
            rightKneeRot=self.genModifiedQuaternion(state.rightKneeRot, c.samplingWindow[3]),
            rightAnkleRot=self.genModifiedQuaternion(state.rightAnkleRot, c.samplingWindow[4]),
            rightShoulderRot=self.genModifiedQuaternion(state.rightShoulderRot, c.samplingWindow[5]),
            rightElbowRot=self.genModifiedQuaternion(state.rightElbowRot, c.samplingWindow[6]),
            leftHipRot=self.genModifiedQuaternion(state.leftHipRot, c.samplingWindow[7]),
            leftKneeRot=self.genModifiedQuaternion(state.leftKneeRot, c.samplingWindow[8]),
            leftAnkleRot=self.genModifiedQuaternion(state.leftAnkleRot, c.samplingWindow[9]),
            leftShoulderRot=self.genModifiedQuaternion(state.leftShoulderRot, c.samplingWindow[10]),
            leftElbowRot=self.genModifiedQuaternion(state.leftElbowRot, c.samplingWindow[11])
        )

        return modified_state
        
    
    def save(self, SM, total_cost_list, order):
        """ 只保存pose和cost 用 """
        assert order.shape[0] == self._nSaveFinal
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
            index = order[i]
            frames = []
            
            for f in range(self._nIter+1):
                state = SM[f][index][0]
                # pose为长为43的列表
                pose = list(state.basePos) + list(state.baseOrn) \
                    + list(state.chestRot) + list(state.neckRot) \
                    + list(state.rightHipRot) + list(state.rightKneeRot) + list(state.rightAnkleRot) \
                    + list(state.rightShoulderRot) + list(state.rightElbowRot) \
                    + list(state.leftHipRot) + list(state.leftKneeRot) + list(state.leftAnkleRot) \
                    + list(state.leftShoulderRot) + list(state.leftElbowRot)

                frames.append(pose)

            dataUnit = {
                'cost': total_cost_list[index],
                'frames': frames
            }

            data[f'path_{i}'] = dataUnit
        print(data)
        f = open(self._savePath, 'w')
        json.dump(data, f)
        f.close()
        print(f'file saved! [{self._savePath}]')
