#!/bin/bash

python3 -m rl_zoo3.train \
    --env UR3eSlide-v1 \
    --algo sac \
    --conf-file /misis/project_multi_skill_rl/rl_skills/hyperparams/sac.yml \
    --hyperparams train_freq:8 gradient_steps:32 batch_size:4096 optimize_memory_usage:False n_envs:32 \
    --n-eval-envs 5 \
    --eval-episodes 10 \
    --n-evaluations 20 \
    --eval-freq 100000 \
    --log-interval 10 \
    --vec-env=subproc \
    -f /misis/project_multi_skill_rl/rl_skills/models \
    --progress \
    --seed 42 \
    --track \
    --wandb-project-name UR3e_Slide_SAC_her \
    --save-replay-buffer
