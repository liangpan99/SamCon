import numpy as np

class HumanoidStablePD():

    def __init__(self, pybullet_client, char_info, model_file, scale, self_collision=True, kinematic_only=False):
        self._pb_client = pybullet_client
        
        """ load urdf """
        char_create_flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        if self_collision:
            char_create_flags = char_create_flags|\
                                self._pb_client.URDF_USE_SELF_COLLISION|\
                                self._pb_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self._body_id = self._pb_client.loadURDF(
            fileName=model_file,
            basePosition=[0, 0, 0],
            globalScaling=scale,
            useFixedBase=False,
            flags=char_create_flags
        )

        """ pre-compute information about the humanoid """
        # 1 base + 19 links + 19 joints
        self._num_joint = self._pb_client.getNumJoints(self._body_id)
        self._link_indices = range(-1, self._num_joint) # include base link (it has mass and collision entity)
        self._joint_indices = range(self._num_joint) # all joints
        self._joint_indices_movable = [] # movable (controllable) joints
        self._joint_type = []
        for j in self._joint_indices:
            joint_info = self._pb_client.getJointInfo(self._body_id, j)
            self._joint_type.append(joint_info[2])
        for j in self._joint_indices:
            joint_type = self.get_joint_type(j)
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                self._joint_indices_movable.append(j)
            elif joint_type == self._pb_client.JOINT_REVOLUTE: 
                self._joint_indices_movable.append(j)
            elif joint_type == self._pb_client.JOINT_FIXED: 
                continue
            else:
                raise NotImplementedError()

        """ setup dynamics or kinematics """
        if kinematic_only:
            self.setup_kinematics()
        else:
            self.setup_dynamics()

        """ kps kds max_forces """
        self._kps = char_info['kps']
        self._kds = char_info['kds']
        self._max_forces = []
        for i, j in enumerate(self._joint_indices_movable):
            joint_type = self.get_joint_type(j)
            if joint_type == self._pb_client.JOINT_REVOLUTE:
                max_force = np.array([char_info['max_forces'][i]])
            elif joint_type == self._pb_client.JOINT_SPHERICAL:
                max_force = np.ones(3) * char_info['max_forces'][i]
            self._max_forces.append(max_force)
                
    def setup_dynamics(self):
        # Settings for the simulation self._body_id
        self._pb_client.changeDynamics(self._body_id, -1, linearDamping=0, angularDamping=0)
        
        # set dynamics parameters for all links (1 base + 19 others)
        for k in self._link_indices:
            self._pb_client.changeDynamics(
                self._body_id, 
                k, 
                lateralFriction=0.8, 
                spinningFriction=0.3, 
                jointDamping=0.0, 
                restitution=0.0,
                # linearDamping=0.0,
                # angularDamping=0.0
            )
        
        # set motor
        for j in self._joint_indices_movable:
            self._pb_client.setJointMotorControl2(
                self._body_id, 
                j, 
                self._pb_client.POSITION_CONTROL,
                targetPosition=0,
                positionGain=0,
                targetVelocity=0,
                force=0
            )
            self._pb_client.setJointMotorControlMultiDof(
                self._body_id,
                j,
                self._pb_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[0, 0, 0]
            )

    def setup_kinematics(self):
        # Settings for the kinematic self._body_id so that it does not affect the simulation
        self._pb_client.changeDynamics(self._body_id, -1, linearDamping=0, angularDamping=0)

        alpha = 0.7
        for k in self._link_indices:
            self._pb_client.changeVisualShape(self._body_id, k, rgbaColor=[1, 1, 1, alpha])
            self._pb_client.setCollisionFilterGroupMask(
                self._body_id,
                k,
                collisionFilterGroup=0,
                collisionFilterMask=0
            )
            self._pb_client.changeDynamics(
                self._body_id,
                k,
                activationState=self._pb_client.ACTIVATION_STATE_SLEEP +
                                self._pb_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                                self._pb_client.ACTIVATION_STATE_DISABLE_WAKEUP
            )

    def get_joint_type(self, idx):
        return self._joint_type[idx]

    def actuate(self, targetPositions):
        """ stable PD control """
        targetPositions = np.array(targetPositions, dtype=np.float64).reshape(-1, 4)
        self._pb_client.setJointMotorControlMultiDofArray(
            self._body_id,
            self._joint_indices_movable,
            self._pb_client.STABLE_PD_CONTROL,
            targetPositions=targetPositions,
            forces=self._max_forces,
            positionGains=self._kps,
            velocityGains=self._kds
        )

    def set_state(self, state):
        ''' 
        Set all state information of the given body. States should include
        pQvw of the base link and p and v for the all joints
        '''      
        state = np.array(state, dtype=np.float64)
        p = state[0:3]
        Q = state[3:7]
        ps = state[7:75].reshape(-1, 4)
        v = state[75:78]
        w = state[78:81]
        vs = state[81:132].reshape(-1, 3)

        assert len(self._joint_indices_movable) == len(ps)

        self.set_base_pQvw(p, Q, v, w)
        self.set_joint_pv(self._joint_indices_movable, ps, vs)

    def get_state(self):
        ''' 
        Return all state information of the given body. This includes 
        pQvw of the base link and p and v for the all joints
        '''        
        p, Q, v, w = self.get_base_pQvw()
        ps, vs = self.get_joint_pv(self._joint_indices_movable)
        state = np.concatenate(
            [p, Q, np.concatenate(ps), v, w, np.concatenate(vs)]
        )
        return state

    def compute_com_pos_vel(self):
        """ compute center-of-mass position and velocity """
        total_mass = 0.0
        com_pos = np.zeros(3)
        com_vel = np.zeros(3)

        for i in self._link_indices:
            di = self._pb_client.getDynamicsInfo(self._body_id, i)
            mass = di[0]
            if i == -1:
                p, _, v, _ = self.get_base_pQvw()
            else:
                p, _, v, _ = self.get_link_pQvw([i])
            total_mass += mass
            com_pos += mass * p
            com_vel += mass * v

        com_pos /= total_mass
        com_vel /= total_mass

        return com_pos, com_vel
    
    def set_base_pQvw(self, p, Q, v=None, w=None):
        ''' 
        Set positions, orientations, linear and angular velocities of the base link.
        ''' 
        self._pb_client.resetBasePositionAndOrientation(self._body_id, p, Q)
        if v is not None and w is not None:
            self._pb_client.resetBaseVelocity(self._body_id, v, w)

    def get_base_pQvw(self):
        ''' 
        Returns position, orientation, linear and angular velocities of the base link.
        ''' 
        p, Q = self._pb_client.getBasePositionAndOrientation(self._body_id)
        p, Q = np.array(p), np.array(Q)
        v, w = self._pb_client.getBaseVelocity(self._body_id)
        v, w = np.array(v), np.array(w)
        return p, Q, v, w

    def get_link_pQvw(self, indices=None):
        ''' 
        Returns positions, orientations, linear and angular velocities given link indices.
        Please use get_base_pQvw for the base link.
        ''' 
        if indices is None:
            indices = range(self._num_joint)
        
        num_indices = len(indices)
        assert num_indices > 0
        
        ls = self._pb_client.getLinkStates(self._body_id, indices, computeLinkVelocity=True)

        ps = [np.array(ls[j][0]) for j in range(num_indices)]
        Qs = [np.array(ls[j][1]) for j in range(num_indices)]
        vs = [np.array(ls[j][6]) for j in range(num_indices)]
        ws = [np.array(ls[j][7]) for j in range(num_indices)]

        if num_indices == 1:
            return ps[0], Qs[0], vs[0], ws[0]
        else:
            return ps, Qs, vs, ws

    def set_joint_pv(self, indices, ps, vs):
        ''' 
        Set positions and velocities given joint indices.
        Please note that the values are locally repsented w.r.t. its parent joint
        '''        
        self._pb_client.resetJointStatesMultiDof(self._body_id, indices, ps, vs)

    def get_joint_pv(self, indices=None):
        ''' 
        Return positions and velocities given joint indices.
        Please note that the values are locally repsented w.r.t. its parent joint
        '''        
        if indices is None:
            indices = range(self._num_joint)

        num_indices = len(indices)
        assert num_indices > 0

        js = self._pb_client.getJointStatesMultiDof(self._body_id, indices)

        ps = []
        vs = []
        for j in range(num_indices):
            p = np.array(js[j][0])
            v = np.array(js[j][1])
            ps.append(p)
            vs.append(v)

        if num_indices == 1:
            return ps[0], vs[0]
        else:
            return ps, vs
    