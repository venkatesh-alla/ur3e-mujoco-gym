import os

from gymnasium.utils.ezpickle import EzPickle
import numpy as np

from ur3e_env import MujocoUR3eEnv, goal_distance

MODEL_XML_PATH = "/misis/project_multi_skill_rl/rl_skills/3d_models_main/ur3e/ur3e_push.xml"

class UR3ePushEnv(MujocoUR3eEnv, EzPickle):

    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoUR3eEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=40,
            gripper_extra_height=0.1,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.10,
            target_range=0.10,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    ### Reward function for push task

    def compute_reward(self, achieved_goal, goal, info):

        # Piece-wise reward signal scaled appropriately
        d = goal_distance(achieved_goal, goal)

        # Gripper to object distance (if scalar, make array)
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:TCP")
        object_pos = self._utils.get_site_xpos(self.model, self.data, "object0") 
        gripper_to_object = np.linalg.norm(grip_pos - object_pos, axis=-1)

        # If scalar input, convert to arrays
        is_scalar_input = np.isscalar(d) or (np.shape(d) == ()) or (np.shape(d) == (1,))
        if is_scalar_input:
            d = np.array([d])
            gripper_to_object = np.array([gripper_to_object])

        if self.reward_type == "sparse":
            reward = -(d > self.distance_threshold).astype(np.float32)
        else:
            reach_threshold = 0.03
            distance_threshold = self.distance_threshold  

            reach_mask = (gripper_to_object > reach_threshold)
            # penalty_reach = -2.0 * gripper_to_object  
            penalty_reach = -2.0 * np.tanh(4 * gripper_to_object)

            if not hasattr(self, '_prev_goal_dist'):
                self._prev_goal_dist = np.copy(d)

            # Delta reward for making progress towards goal
            delta_progress = self._prev_goal_dist - d
            reward_progress = 5.0 * delta_progress

            # Success bonus when the object is close enough to the goal
            success_mask = (d <= distance_threshold)
            success_bonus = 10.0 * success_mask.astype(np.float32)

            # Total reward composition
            reward = np.where(reach_mask, penalty_reach, reward_progress)
            reward += success_bonus
            self._prev_goal_dist = np.copy(d)

        reward = np.asarray(reward, dtype=np.float32)

        # Return a float if scalar input, else return full array
        if is_scalar_input:
            return float(reward[0])  # shape was (1,)
        else:
            return reward  # shape (batch_size,)

