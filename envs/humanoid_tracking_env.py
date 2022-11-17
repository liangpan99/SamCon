import os
import sys
sys.path.append(os.getcwd())

import os.path as osp
import numpy as np
import pybullet as p
import pybullet_data

from envs.humanoid_stable_pd import HumanoidStablePD
from utils.bullet_client import BulletClient

class HumanoidTrackingEnv():

    def __init__(self, cfg, viz):
        self._cfg = cfg
        self._viz = viz

        """ pybullet client """
        mode = p.GUI if viz else p.DIRECT
        self._pb_client = BulletClient(mode)
        self._pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        """ timestep for simulator & action """
        self._dt_sim = 1.0 / cfg.fps_sim
        self._dt_act = 1.0 / cfg.fps_act
        if cfg.fps_sim % cfg.fps_act != 0:
            raise Exception('FPS_SIM should be a multiples of FPS_ACT')
        self._num_substep = cfg.fps_sim // cfg.fps_act

        """ setup physics scene """
        self.setup_physics_scene(
            cfg.char_info, 
            cfg.model_file, 
            cfg.scale, 
            cfg.self_collision
        )

    def setup_physics_scene(self,  char_info, model_file, scale, self_collision):
        self._pb_client.resetSimulation()

        """ setup simulator parameters """
        self._pb_client.setPhysicsEngineParameter(numSubSteps=2)
        self._pb_client.setPhysicsEngineParameter(numSolverIterations=10)
        self._pb_client.setGravity(0, 0, -9.8) # coordinate system is Z-up
        self._pb_client.setTimeStep(self._dt_sim)

        # for deterministic simulation
        # see https://github.com/bulletphysics/bullet3/issues/2982
        self._pb_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        if self._viz:
            self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_GUI, 0)
            # cameraArgs = {
            #     'cameraDistance': 1,
            #     'cameraYaw': 0,
            #     'cameraPitch': 0,
            #     'cameraTargetPosition': [0, 1, 1]
            # }
            # self._pb_client.resetDebugVisualizerCamera(**cameraArgs)

        """ create ground plane """
        self.create_ground()

        """ create humanoid """
        self._sim_agent = HumanoidStablePD(self._pb_client, char_info, model_file, scale, self_collision, kinematic_only=False)
        self._kin_agent = HumanoidStablePD(self._pb_client, char_info, model_file, scale, self_collision, kinematic_only=True)

        # trick for make contact solver stable
        # it's important for perfectly reproduce the trajectory when running eval_samcon.py
        self._pb_client.performCollisionDetection()

    def create_ground(self):
        # create ground plane
        self._plane_id = self._pb_client.loadURDF('plane_implicit.urdf', [0, 0, 0], useMaximalCoordinates=True)
        self._pb_client.changeDynamics(self._plane_id, linkIndex=-1, lateralFriction=0.9)

    def compute_total_cost(self):
        pose_w, root_w, ee_w, balance_w, com_w = self._cfg.cost_weights
        pose_cost = self.compute_pose_cost()
        root_cost = self.compute_root_cost()
        ee_cost = self.compute_ee_cost()
        balance_cost = self.compute_balance_cost()
        com_cost = self.compute_com_cost()
        total_cost = pose_w * pose_cost + \
                     root_w * root_cost + \
                     ee_w * ee_cost + \
                     balance_w * balance_cost + \
                     com_w * com_cost
        return total_cost, pose_cost, root_cost, ee_cost, balance_cost, com_cost

    def compute_pose_cost(self):
        """ pose + angular velocity of internal joints in local coordinate """
        error = 0.0
        joint_weights = self._cfg.joint_weights
        sim_joint_ps, sim_joint_vs = self._sim_agent.get_joint_pv(self._sim_agent._joint_indices_movable)
        kin_joint_ps, kin_joint_vs = self._kin_agent.get_joint_pv(self._kin_agent._joint_indices_movable)

        for i in range(len(joint_weights)):
            _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_joint_ps[i], kin_joint_ps[i])
            )
            diff_pos_vel = sim_joint_vs[i] - kin_joint_vs[i]
            error += joint_weights[i] * (1.0 * np.dot(diff_pose_pos, diff_pose_pos) + 0.1 * np.dot(diff_pos_vel, diff_pos_vel))
        error /= len(joint_weights)            
        return error

    def compute_root_cost(self):
        """ orientation + angular velocity of root in world coordinate """
        error = 0.0
        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = self._sim_agent.get_base_pQvw()
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = self._kin_agent.get_base_pQvw()

        diff_root_p = sim_root_p - kin_root_p
        diff_root_p = diff_root_p[:2] # only consider XY-component

        _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
            self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q)
        )
        diff_root_v = sim_root_v - kin_root_v
        diff_root_w = sim_root_w - kin_root_w
        # error += 1.0 * np.dot(diff_root_p, diff_root_p) + \
        #          1.0 * np.dot(diff_root_Q, diff_root_Q) + \
        #          0.1 * np.dot(diff_root_v, diff_root_v) + \
        #          0.1 * np.dot(diff_root_w, diff_root_w)
        error += 1.0 * np.dot(diff_root_Q, diff_root_Q) + \
                 0.1 * np.dot(diff_root_w, diff_root_w)
        return error

    def compute_ee_cost(self):
        """ end-effectors (height) in world coordinate """
        error = 0.0
        end_effectors = self._cfg.end_effectors
        sim_ps, _, _, _ = self._sim_agent.get_link_pQvw(end_effectors)
        kin_ps, _, _, _ = self._kin_agent.get_link_pQvw(end_effectors)
        
        for sim_p, kin_p in zip(sim_ps, kin_ps):
            diff_pos = sim_p - kin_p
            diff_pos = diff_pos[-1] # only consider Z-component (height)
            error += np.dot(diff_pos, diff_pos)

        error /= len(end_effectors)
        return error

    def compute_balance_cost(self):
        """ balance cost plz see the SamCon paper """
        error = 0.0
        sim_com_pos, sim_com_vel = self._sim_agent.compute_com_pos_vel()
        kin_com_pos, kin_com_vel = self._kin_agent.compute_com_pos_vel()
        end_effectors = self._cfg.end_effectors
        sim_ps, _, _, _ = self._sim_agent.get_link_pQvw(end_effectors)
        kin_ps, _, _, _ = self._kin_agent.get_link_pQvw(end_effectors)

        for i in range(len(end_effectors)):
            sim_planar_vec = sim_com_pos - sim_ps[i]
            kin_planar_vec = kin_com_pos - kin_ps[i]
            diff_planar_vec = sim_planar_vec - kin_planar_vec
            diff_planar_vec = diff_planar_vec[:2] # only consider XY-component
            error += np.dot(diff_planar_vec, diff_planar_vec)
        error /= len(end_effectors) * self._cfg.height

        # diff_com_vel = sim_com_vel - kin_com_vel
        # error += 1.0 * np.dot(diff_com_vel, diff_com_vel)

        return error
    
    def compute_com_cost(self):
        """ CoM (position linVel) in world coordinate """
        error = 0.0
        sim_com_pos, sim_com_vel = self._sim_agent.compute_com_pos_vel()
        kin_com_pos, kin_com_vel = self._kin_agent.compute_com_pos_vel()
        diff_com_pos = sim_com_pos - kin_com_pos
        diff_com_vel = sim_com_vel - kin_com_vel
        error += 1.0 * np.dot(diff_com_pos, diff_com_pos) + \
                 0.1 * np.dot(diff_com_vel, diff_com_vel)
        return error

    def save_pb_state(self, dir, filename):
        self._pb_client.saveBullet(osp.join(self._cfg.state_dir, f'{dir}', f'{filename}.bullet'))
    
    def restore_pb_state(self, dir, filename):
        self._pb_client.restoreState(fileName=osp.join(self._cfg.state_dir, f'{dir}', f'{filename}.bullet'))

    def step(self, action):
        s = self._kin_agent.get_state()

        for _ in range(self._num_substep):
            self._sim_agent.actuate(action)
            self._pb_client.stepSimulation()

        # after stepSimulation(), the velocities of kin_agent will be zero, so we need to reset it
        # see https://github.com/bulletphysics/bullet3/issues/2401
        self._kin_agent.set_state(s)

        simulated_state = self._sim_agent.get_state()
        total_cost, pose_cost, root_cost, ee_cost, balance_cost, com_cost = self.compute_total_cost()

        info = [pose_cost, root_cost, ee_cost, balance_cost, com_cost]

        return simulated_state, total_cost, info
