import os

from gymnasium.utils.ezpickle import EzPickle

from ur3e_env import MujocoUR3eEnv

MODEL_XML_PATH = "/misis/project_multi_skill_rl/rl_skills/3d_models_main/ur3e/ur3e_reach.xml"

class UR3eReachEnv(MujocoUR3eEnv, EzPickle):    

    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoUR3eEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=40,
            gripper_extra_height=0.35,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

