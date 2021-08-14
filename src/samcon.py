import json
import numpy as np

from src.simulation import HumanoidStablePD
from src.mocapdata import PybulletMocapData, State
from config.humanoid_config import HumanoidConfig as c

INIT_STATE_INDEX = 0
TARGET_STATE_INDEX = 1
SIM_TARGET_STATE_INDEX = 2
COST_INDEX = 3

offset = 0.0 # 时间偏移

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
        assert self._nSaveFinal <= self._nSave

        """
        SM = [
            t=0 [ [0.0, 0.0, initState, 0.0], [], ..., nSave ],
            t=1 [ [initState, targetState, simTargetState, cost], [], ..., nSave],
            t=2 [],
        ]
        
        每次仿真nSample次，然后一起计算cost

        """

        SM = []

        # initialize SM[0]
        SM.append([])
        firstInitState = self._mocap_data.getSpecTimeState(t=0.0+offset)
        for i in range(self._nSave):
            SM[0].append([0.0, 0.0, firstInitState, 0.0]) # [initState, targetState, simTargetState, cost]

        for t in range(1, self._nIter+1):
            print('t=', t)
            nInitState = len(SM[t-1])
            nSampleEachInitState = int(self._nSample / nInitState)

            target_state = self._mocap_data.getSpecTimeState(t*self._sampleTimeStep+offset)
            S = []
            cost_list = []
            for init_state in SM[t-1]:
                init_state = init_state[SIM_TARGET_STATE_INDEX]
                
                for i in range(nSampleEachInitState):

                    sampled_target_state = self.sample(target_state) # 只修改pose
                    self._humanoid.resetState(init_state, target_state)
                    sim_target_state, cost = self._humanoid.simulation(sampled_target_state, self._sampleTimeStep, displayFPS)
                    S.append([init_state, target_state, sim_target_state, cost])
                    cost_list.append(cost)
            
            # 从nSample个样本中挑nSave个最小的保存
            # TBD: 构建cost分布
            cost_list = np.array(cost_list)
            cost_order = cost_list.argsort() # 从小到大排序 返回索引
            cost_order = cost_order[0:self._nSave] # 取前nSave个

            SM.append([])
            for index in cost_order:
                SM[t].append(S[index])

        ## 从SM的尾部开始回溯, 找到nSave个path, 求他们的total_cost, 保存nSaveFinal个total_cost最小的结果
        # 找path
        path_list = []
        for pathId in range(self._nSave):
            path = []
            path.append(pathId)

            currTime_initState = SM[-1][pathId][INIT_STATE_INDEX]
            currTime_initState = np.array(self.getListFromNamedtuple(currTime_initState))
            for i in range(self._nIter-1): # 只需要比较nIter-1次
                lastTime = -2 - i
                for j in range(self._nSave):
                    lastTime_simTargetState = SM[lastTime][j][SIM_TARGET_STATE_INDEX]
                    lastTime_simTargetState = np.array(self.getListFromNamedtuple(lastTime_simTargetState))
                    # 当前时刻的初始状态等于上一时刻的仿真结果, 则说明两者属于同一条path
                    # 存在潜在的BUG: 两者的值恰巧相等, 实际上并无对应关系
                    # TBD: 使用唯一标记来标识每次的仿真
                    if int((lastTime_simTargetState - currTime_initState).sum()) == 0:
                        path.insert(0, j)
                        currTime_initState = SM[lastTime][j][INIT_STATE_INDEX]
                        currTime_initState = np.array(self.getListFromNamedtuple(currTime_initState))
                        break
                    
            path_list.append(path)


        # 求每条path的total_cost
        total_cost_list = []
        for pathId in range(self._nSave):
            total_cost = 0
            for t in range(1, self._nIter+1):
                savedSample_index = path_list[pathId][t-1]
                total_cost += SM[t][savedSample_index][COST_INDEX]
            total_cost_list.append(total_cost)
        
        total_cost_list = np.array(total_cost_list)
        order = total_cost_list.argsort() # 从小到大排序 返回索引
        order = order[0:self._nSaveFinal] # 取前nSaveFinal个最小的
        
        self.save(SM, path_list, total_cost_list, order)


       
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
            key = f'path_{pathId}'
            path_cost = data[key]['cost']
            t0_state = data[key]['t0_state']
            reference_states = data[key]['reference_states']
            simulated_states = data[key]['simulated_states']

            assert len(reference_states) == nIter
            assert len(simulated_states) == nIter

            self._humanoid.resetState(start_state=self.getStateFromList(t0_state), end_state=None)

            for t in range(nIter+1):
                reference_state = reference_states[t-1]
                reference_state = self.getStateFromList(reference_state)

                simulated_state = simulated_states[t-1]
                simulated_state = self.getStateFromList(simulated_state)

                self._humanoid.resetState(start_state=None, end_state=reference_state)
                self._humanoid.simulation(simulated_state, sampleTimeStep, displayFPS)

        

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
        
    
    def save(self, SM, path_list, pathCost_list, savedIndices):
        """ 保存为.txt """
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
            

            # 保存t=0的初始状态
            t0_state = SM[0][0][SIM_TARGET_STATE_INDEX]
            t0_state = self.getListFromNamedtuple(t0_state)

            # 保存t=0之后时刻的reference states, simulated states
            reference_states = []
            simulated_states = []
            for t in range(1, self._nIter+1):
                savedSample_index = path_list[pathId][t-1]
                reference_state = SM[t][savedSample_index][TARGET_STATE_INDEX]
                simulated_state = SM[t][savedSample_index][SIM_TARGET_STATE_INDEX]

                reference_state = self.getListFromNamedtuple(reference_state)
                simulated_state = self.getListFromNamedtuple(simulated_state)

                reference_states.append(reference_state)
                simulated_states.append(simulated_state)


            dataUnit = {
                'cost': pathCost_list[pathId],
                't0_state': t0_state,
                'reference_states': reference_states,
                'simulated_states': simulated_states,
            }

            data[f'path_{i}'] = dataUnit
    
        f = open(self._savePath, 'w')
        json.dump(data, f)
        f.close()
        print(f'file saved! [{self._savePath}]')

    def getListFromNamedtuple(self, namedtuple, is_pose=True, is_velocity=True):
        """ 从collections.namedtuple类型的state转换为列表 
        
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
        
        return pose+velocity


    def getStateFromList(self, data):
        """ 从列表转换为collections.namedtuple类型的state
        
        Return:
            pose len=43
            velocity len=34
            pose+velocity len=77
        """

        state = State(
        # Position
        basePos = data[0:3],
        baseOrn = data[3:7],
        chestRot = data[7:11], 
        neckRot = data[11:15], 
        rightHipRot = data[15:19], 
        rightKneeRot = data[19:20], 
        rightAnkleRot = data[20:24], 
        rightShoulderRot = data[24:28], 
        rightElbowRot = data[28:29], 
        leftHipRot = data[29:33], 
        leftKneeRot = data[33:34], 
        leftAnkleRot = data[34:38], 
        leftShoulderRot = data[38:42], 
        leftElbowRot = data[42:43],

        # Velocity
        baseLinVel = data[43:46],
        baseAngVel = data[46:49],
        chestVel = data[49:52], 
        neckVel = data[52:55], 
        rightHipVel = data[55:58], 
        rightKneeVel = data[58:59], 
        rightAnkleVel = data[59:62], 
        rightShoulderVel = data[62:65], 
        rightElbowVel = data[65:66], 
        leftHipVel = data[66:69], 
        leftKneeVel = data[69:70], 
        leftAnkleVel = data[70:73], 
        leftShoulderVel = data[73:76], 
        leftElbowVel = data[76:77],
        )
        
        return state
