import os
from gymnasium.utils.ezpickle import EzPickle
from ur3e_pickplace_env import MujocoUR3eEnv, goal_distance
import numpy as np

MODEL_XML_PATH = "/misis/project_multi_skill_rl/rl_skills/3d_models_main/ur3e/ur3e_pickplace.xml"

class UR3ePickPlaceEnv(MujocoUR3eEnv, EzPickle):
    def __init__(self, reward_type="sparse", **kwargs):
        self._last_grasp_reward = np.zeros(1)
        self.cumulative_reach_penalty = np.zeros(1)
        self.cumulative_grasp_reward = np.zeros(1)
        self.cumulative_lift_reward = np.zeros(1)
        self.cumulative_final_reward = np.zeros(1)
        self.cumulative_progress_bonus = np.zeros(1)
        self.cumulative_proximity_bonus = np.zeros(1)
        self.cumulative_success_bonus = np.zeros(1)
        self.cumulative_air_goal_transport_phase_bonus = np.zeros(1)
        self.cumulative_table_goal_transport_phase_bonus = np.zeros(1)
        self.cumulative_gripper_opening_bonus = np.zeros(1)
        self.cumulative_release_bonus = np.zeros(1)
        self.cumulative_gripper_near_penalty = np.zeros(1)

        initial_qpos = {
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoUR3eEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=40,
            gripper_extra_height=0.1,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.10,
            target_range=0.10,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def reset(self, seed=None, options=None):
        self._last_grasp_reward = 0.0

        obs = super().reset(seed=seed, options=options)

        # Reset cumulative rewards and bonuses
        self.cumulative_reach_penalty = np.zeros(1)
        self.cumulative_grasp_reward = np.zeros(1)
        self.cumulative_lift_reward = np.zeros(1)
        self.cumulative_final_reward = np.zeros(1)
        self.cumulative_progress_bonus = np.zeros(1)
        self.cumulative_proximity_bonus = np.zeros(1)
        self.cumulative_success_bonus = np.zeros(1)
        self.cumulative_air_goal_transport_phase_bonus = np.zeros(1)
        self.cumulative_table_goal_transport_phase_bonus = np.zeros(1)
        self.cumulative_gripper_opening_bonus = np.zeros(1)
        self.cumulative_release_bonus = np.zeros(1)
        self.cumulative_gripper_near_penalty = np.zeros(1)
        

        # Resetting variables and flags
        self._prev_object_pos = None
        self._prev_goal_dist = None
        # self._prev_gripper_grasp_scalar = None

        # persisting
        self.lift_phase_success = np.array([False])
        self.persist_lift_phase_success = np.array([False])

        return obs
        
    def _is_success(self, achieved_goal, desired_goal):
        if achieved_goal.ndim == 1:
            d = goal_distance(achieved_goal[0:3], desired_goal[0:3])
        else:
            d = goal_distance(achieved_goal[:,0:3], desired_goal[:,0:3])
        return (d < self.distance_threshold).astype(np.float32)
    
    def compute_reward(self, achieved_goal, goal, info):
        # Constants
        OBJECT_TCP_DIST_THRESHOLD = 0.04  # meters
        TABLE_Z_HEIGHT = 0.4
        LIFT_HEIGHT_THRESHOLD = TABLE_Z_HEIGHT + 0.03
        
        if achieved_goal.ndim == 1:
            d = goal_distance(achieved_goal[0:3], goal[0:3])
            grip_pos = achieved_goal[3:6]
            object_pos = achieved_goal[0:3]
            gripper_grasp_scalar = achieved_goal[6]
            object_z = object_pos[2]
            goal_z = goal[2]
        else:
            d = goal_distance(achieved_goal[:,0:3], goal[:,0:3])
            grip_pos = achieved_goal[:,3:6]
            object_pos = achieved_goal[:,0:3]
            gripper_grasp_scalar = achieved_goal[:,6]
            object_z = object_pos[:, 2]
            goal_z = goal[:, 2]

        # Flags initialization
        # lift_phase_success = np.zeros_like(d, dtype=bool)
        air_goal_transport_flag = np.zeros_like(d, dtype=bool)
        table_goal_transport_flag = np.zeros_like(d, dtype=bool)
        gripper_retreat_flag = np.zeros_like(d, dtype=bool)
        
        # Measures
        assert grip_pos.shape == object_pos.shape 
        gripper_to_object_dist = np.linalg.norm(grip_pos - object_pos, axis=-1)

        gripper_closure_norm = gripper_grasp_scalar / 255.0

        # Ensure array shape
        is_scalar_input = np.isscalar(d) or np.shape(d) == () or np.shape(d) == (1,)
        if is_scalar_input:
            d = np.array([d])
            gripper_to_object_dist = np.array([gripper_to_object_dist])
            object_z = np.array([object_z])
            goal_z = np.array([goal_z])
            gripper_closure_norm = np.array([gripper_closure_norm])
            gripper_grasp_scalar = np.array([gripper_grasp_scalar])

        # Initialize reward array
        reward = np.zeros_like(d, dtype=np.float32)

        if self.reward_type == "sparse":
            reward = -(d > self.distance_threshold).astype(np.float32)
            
        else:
            # Flags
            object_attached = gripper_to_object_dist < OBJECT_TCP_DIST_THRESHOLD
            object_lifted = object_z > LIFT_HEIGHT_THRESHOLD
            goal_on_table = goal_z < 0.46
            is_gripper_open = gripper_grasp_scalar < 50

            # lift phase success flag

            self.lift_phase_success = np.where(
                ~self.persist_lift_phase_success,
                object_attached & object_lifted & ((object_z - TABLE_Z_HEIGHT) > 0.07),
                self.lift_phase_success  # keep original value if persist_lift_phase_success is True
            )
            
            # Update transport flags
            air_goal_transport_flag = self.lift_phase_success & ~goal_on_table
            table_goal_transport_flag = self.lift_phase_success & goal_on_table

            # Reward Components
            reach_penalty = -2.0 * np.tanh(4.0 * gripper_to_object_dist)
            grasp_reward = np.where(
                object_attached & ~object_lifted,
                gripper_closure_norm,
                np.where(object_lifted, getattr(self, "_last_grasp_reward", 0.0), 0.0)
            )
            self._last_grasp_reward = np.where(object_attached & ~object_lifted, grasp_reward, 0.0)

            lift_reward = np.where(
                object_attached & ~(air_goal_transport_flag | table_goal_transport_flag),
                np.where(
                    (object_z - TABLE_Z_HEIGHT) <= 0.1,
                    50.0 * (object_z - TABLE_Z_HEIGHT),
                    0.0
                ), 
                0.0
            )        
            
            #reward = reach_penalty + grasp_reward + lift_reward
            # Phase change bonuses
            air_goal_transport_phase_bonus = np.where(air_goal_transport_flag & ~goal_on_table, 5.0, 0.0)
            table_goal_transport_phase_bonus = np.where(table_goal_transport_flag & goal_on_table, 5.0, 0.0)

            # Bonus and penalty initialization
            progress_bonus = np.zeros_like(d, dtype=np.float32)
            proximity_bonus = np.zeros_like(d, dtype=np.float32)
            success_bonus = np.zeros_like(d, dtype=np.float32)
            gripper_opening_bonus = np.zeros_like(d, dtype=np.float32)
            release_bonus = np.zeros_like(d, dtype=np.float32)
            gripper_near_penalty = np.zeros_like(d, dtype=np.float32)
            delta_progress = np.zeros_like(d, dtype=np.float32)

            table_goal_transport_reward = np.zeros_like(d, dtype=np.float32)
            air_goal_transport_reward = np.zeros_like(d, dtype=np.float32)

            # Goal in air rewards scheme - Delta progress, progress bonus, proximity bonus, success bonus

            # Adding air goal transport phase bonus
            air_goal_transport_reward = np.where(air_goal_transport_flag & ~goal_on_table, air_goal_transport_phase_bonus, 0.0)
            
            # if self._prev_goal_dist is None:
            #     self._prev_goal_dist = np.copy(d)

            # delta_progress = self._prev_goal_dist - d
            # self._prev_goal_dist = d  # Update every step after computing delta

            
            update_mask = air_goal_transport_flag | table_goal_transport_flag
            # Initialize to zeros (or another default) if None
            if self._prev_goal_dist is None:
                self._prev_goal_dist = np.zeros_like(d)

            # Now safe to compute (only updates where mask=True)
            delta_progress = np.where(update_mask, self._prev_goal_dist - d, 0.0)
            self._prev_goal_dist = np.where(update_mask, d, self._prev_goal_dist)

            progress_bonus = np.where(
                air_goal_transport_flag & ~goal_on_table,
                np.clip(100.0 * delta_progress, 0.0, None),
                0.0
            )

            d_clipped = np.clip(d, self.distance_threshold, None)
            proximity_bonus = np.where(
                air_goal_transport_flag & ~goal_on_table,
                5.0 * np.exp(-5.0 * d_clipped),
                0.0
            )

            success_mask = (d <= self.distance_threshold)
            success_bonus = np.where(
                air_goal_transport_flag & ~goal_on_table,
                5.0 * success_mask.astype(np.float32),
                0.0
            )

            # Goal on table reward scheme - Progress bonus, proximity bonus, gripper opening bonus, release bonus, gripper near penalty

            # Goal on table phase bonus
            table_goal_transport_reward = np.where(table_goal_transport_flag & goal_on_table, table_goal_transport_phase_bonus, 0.0)

            progress_bonus = np.where(
                table_goal_transport_flag & goal_on_table,
                np.clip(100.0 * delta_progress, 0.0, None),
                progress_bonus
            )

            d_clipped = np.clip(d, self.distance_threshold, None)
            proximity_bonus = np.where(
                table_goal_transport_flag & goal_on_table,
                5.0 * np.exp(-5.0 * d_clipped),
                proximity_bonus
            )

            # Add a new persistent state variable to track initialization for each env
            if not hasattr(self, "_prev_gripper_grasp_scalar"):
                # Initialize a placeholder value for _prev_gripper_grasp_scalar
                self._prev_gripper_grasp_scalar = np.zeros_like(gripper_grasp_scalar)

            # Define the condition that triggers the logic
            release_logic_condition = table_goal_transport_flag & goal_on_table & (d < 0.06)

            # Calculate the delta based on the previous and current values
            gripper_opening_delta = self._prev_gripper_grasp_scalar - gripper_grasp_scalar

            # Calculate the bonus, but only for the environments where the logic condition is met
            gripper_opening_bonus = np.where(
                release_logic_condition & (gripper_opening_delta > 1.0),
                5.0,
                0.0
            )

            # update the previous gripper scalar ONLY for the environments
            # where the logic condition is currently true.
            self._prev_gripper_grasp_scalar = np.where(
                release_logic_condition,
                np.copy(gripper_grasp_scalar),
                self._prev_gripper_grasp_scalar  # Keep the old value if the condition is false
            )


            release_success_mask = is_gripper_open & (d <= self.distance_threshold) & ~object_attached
            release_bonus = np.where(table_goal_transport_flag & release_success_mask, 10.0, 0.0)

            gripper_retreat_flag = np.where(
                table_goal_transport_flag & release_success_mask & ~gripper_retreat_flag,
                True,
                gripper_retreat_flag
            )

            gripper_near_penalty = np.where(
                table_goal_transport_flag & gripper_retreat_flag & (gripper_to_object_dist < 0.1),
                -60.0 * (0.1 - gripper_to_object_dist),
                0.0
            )


            # Persist lift_phase_success flag
            self.persist_lift_phase_success = np.where(
                table_goal_transport_flag & goal_on_table & (d < 0.08) & ~is_gripper_open & ~self.persist_lift_phase_success,
                True,
                self.persist_lift_phase_success
            )



            self.lift_phase_success = np.where(
                table_goal_transport_flag & self.persist_lift_phase_success,
                True,
                self.lift_phase_success
            )

            # Ensure all arrays are of type float32
            reward = reward.astype(np.float32)
            progress_bonus = progress_bonus.astype(np.float32)
            proximity_bonus = proximity_bonus.astype(np.float32)
            success_bonus = success_bonus.astype(np.float32)
            gripper_opening_bonus = gripper_opening_bonus.astype(np.float32)
            release_bonus = release_bonus.astype(np.float32)
            gripper_near_penalty = gripper_near_penalty.astype(np.float32)
            
            reward = reach_penalty + grasp_reward + lift_reward + table_goal_transport_reward + air_goal_transport_reward + progress_bonus + proximity_bonus + success_bonus + gripper_opening_bonus + release_bonus + gripper_near_penalty

            # Update cumulative rewards
            self.cumulative_reach_penalty += np.sum(reach_penalty)
            self.cumulative_grasp_reward += np.sum(grasp_reward)
            self.cumulative_lift_reward += np.sum(lift_reward)
            self.cumulative_progress_bonus += np.sum(progress_bonus)
            self.cumulative_proximity_bonus += np.sum(proximity_bonus)
            self.cumulative_success_bonus += np.sum(success_bonus)
            self.cumulative_air_goal_transport_phase_bonus += np.sum(air_goal_transport_phase_bonus)
            self.cumulative_table_goal_transport_phase_bonus += np.sum(table_goal_transport_phase_bonus)
            self.cumulative_gripper_opening_bonus += np.sum(gripper_opening_bonus)
            self.cumulative_release_bonus += np.sum(release_bonus)
            self.cumulative_gripper_near_penalty += np.sum(gripper_near_penalty)

            self.cumulative_final_reward += np.sum(reward)

        return reward[0].astype(np.float32) if is_scalar_input else reward.astype(np.float32)
