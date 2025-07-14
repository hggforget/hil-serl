#!/home/jd/miniconda3/envs/hilserl/bin/python python

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
export DISPLAY=:0 && \

/home/jd/miniconda3/envs/hilserl/bin/python  examples/train_rlpd_bostrapped.py "$@" \
    --exp_name=take_cup \
    --checkpoint_path=/mnt/hil-serl/tasks/take_cup/init/boostrapped_run_4 \
    --bc_checkpoint_path=/mnt/hil-serl/tasks/take_cup/init/bc_minmax_noforzen_1e-4_split_ckpt \
    --demo_path=/mnt/hil-serl/tasks/take_cup/init/demos/grasp_clip/rlpd_demos.pickle \
    --log_dir=/mnt/hil-serl/logs/rlpd_bostrapped \
    --pretrain_critic_step=2500 \
    --bootstrap \
    --learner \ 