import sys
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

sys.path.append('/misis/project_multi_skill_rl/rl_skills/envs')

import mujoco_UR3e_reach


for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    register(
        id=f"UR3eReach{suffix}-v1",
        entry_point="mujoco_UR3e_reach:UR3eReachEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
        )


env = gym.make("UR3eReach-v1", render_mode="human")
obs, _ = env.reset()
for episode in range(100):
    obs, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

env.close()