import os

from gymnasium.utils.ezpickle import EzPickle

from ur3e_env import MujocoUR3eEnv, goal_distance
import numpy as np

MODEL_XML_PATH = "/misis/project_multi_skill_rl/rl_skills/3d_models_main/ur3e/ur3e_slide.xml"


class UR3eSlideEnv(MujocoUR3eEnv, EzPickle):
    

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
            target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1,
            target_range=0.3,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def reset(self, seed=None, options=None):
        obs = super().reset(seed=seed, options=options)

        # Cache the initial object position and reset others
        self._initial_object_pos = self._utils.get_site_xpos(self.model, self.data, "object0").copy()
        self._prev_object_pos = None
        self._prev_goal_dist = None

        return obs

    
    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)

        # Get positions
        object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")        # shape (batch, 3)
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:TCP")       # shape (batch, 3)
        gripper_to_object = np.linalg.norm(grip_pos - object_pos, axis=-1)              # shape (batch,)
        
        # Ensure array shape
        is_scalar_input = np.isscalar(d) or np.shape(d) == () or np.shape(d) == (1,)
        if is_scalar_input:
            d = np.array([d])
            gripper_to_object = np.array([gripper_to_object])

        # Phase split: check if object has moved significantly
        object_moved = np.linalg.norm(object_pos - self._initial_object_pos, axis=-1) > 0.002
        # Init reward array
        reward = np.zeros_like(d, dtype=np.float32)

        if self.reward_type == "sparse":
            reward = -(d > self.distance_threshold).astype(np.float32)
        else:
            # --- Phase 1: Approach ---
            reach_penalty = -2.0 * np.tanh(4 * gripper_to_object)  # shape (batch,)

            # --- Phase 2: Slide ---
            if self._prev_object_pos is None:
                self._prev_object_pos = np.copy(object_pos)
            if self._prev_goal_dist is None:
                self._prev_goal_dist = np.copy(d)

            goal_dir = goal - object_pos
            goal_dir_unit = goal_dir / (np.linalg.norm(goal_dir, axis=-1, keepdims=True) + 1e-6)

            delta_pos = object_pos - self._prev_object_pos
            direction_bonus = 40.0 * np.sum(delta_pos * goal_dir_unit, axis=-1)
            direction_bonus = np.clip(direction_bonus, 0.0, None)  # Clip negative values to 0

            delta_progress = self._prev_goal_dist - d
            reward_progress = 100.0 * delta_progress
            reward_progress = np.clip(reward_progress, 0.0, None)  # Clip negative values to 0

            # proximity bonus
            d_clipped = np.clip(d, self.distance_threshold, None)
            proximity_bonus = 5.0 * np.exp(-5.0 * d_clipped)

            success_mask = (d <= self.distance_threshold)
            success_bonus = 5.0 * success_mask.astype(np.float32)

            # Combine rewards using masks (no in-place += to avoid shape mismatch)
            reward = (~object_moved) * reach_penalty + object_moved * (direction_bonus + reward_progress + proximity_bonus + success_bonus)            

            # Update state
            self._prev_object_pos = np.copy(object_pos)
            self._prev_goal_dist = np.copy(d)

        return float(reward[0]) if is_scalar_input else reward
