from typing import Callable, Optional
import os

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper
from gymnasium.wrappers import FlattenObservation

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.env_util import make_vec_env
#from gymnasium import wrapper

import sys
sys.path.append('/misis/project_multi_skill_rl/rl_skills/envs')
#sys.path.append('/content/drive/MyDrive/project_multi_skill_rl/rl_skills/envs')
import mujoco_UR3e_reach
import mujoco_UR3e_pick_place
import mujoco_UR3e_slide
import mujoco_UR3e_push

try:
    import UR3e
except ImportError:
    pass

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    register(
        id=f"UR3eReach{suffix}-v1",
        entry_point="mujoco_UR3e_reach:UR3eReachEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
        )

    register(
        id=f"UR3ePickPlace{suffix}-v1",
        entry_point="mujoco_UR3e_pick_place:UR3ePickPlaceEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
        )
    register(
        id=f"UR3ePush{suffix}-v1",
        entry_point="mujoco_UR3e_push:UR3ePushEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
        )        
    register(
        id=f"UR3eSlide{suffix}-v1",
        entry_point="mujoco_UR3e_slide:UR3eSlideEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
        )

