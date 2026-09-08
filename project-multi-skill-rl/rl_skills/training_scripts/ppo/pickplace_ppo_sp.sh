#!/bin/bash

python3 -m rl_zoo3.train \
    --env UR3ePickPlace-v1 \
    --algo ppo \
    --conf-file /misis/project_multi_skill_rl/rl_skills/hyperparams/ppo.yml \
    --hyperparams n_envs:10 n_timesteps:20000000 \
    --n-eval-envs 2 \
    --eval-episodes 10 \
    --n-evaluations 20 \
    --eval-freq 100000 \
    --log-interval 1 \
    --vec-env=subproc \
    -f /misis/project_multi_skill_rl/rl_skills/models \
    --progress \
    --track \
    --wandb-project-name UR3e_PickPlace_PPO \
    --seed 42
