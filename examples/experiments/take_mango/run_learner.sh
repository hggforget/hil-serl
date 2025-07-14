#!/home/jd/miniconda3/envs/hilserl/bin/python python

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export DISPLAY=:0 && \

/home/jd/miniconda3/envs/hilserl/bin/python  ../../train_rlpd.py "$@" \
    --exp_name=take_mango \
    --checkpoint_path=/mnt/hil-serl/first_run \
    --bc_checkpoint_path=/mnt/hil-serl/examples/experiments/take_cup/bc_ckpt \
    --demo_path=/mnt/hil-serl/pickles/rlpd/demos_cam_body_relative_obs.pickle \
    --log_dir=logs/rlpd \
    --learner \ 