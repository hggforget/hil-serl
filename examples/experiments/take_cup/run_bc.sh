export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
export DISPLAY=:0 && \

/home/jd/miniconda3/envs/hilserl/bin/python  examples/train_bc.py "$@" \
    --exp_name=take_cup \
    --demo_path=/mnt/hil-serl/tasks/take_cup/init/demos/grasp_clip/bc_demos.pickle \
    --bc_checkpoint_path=/mnt/hil-serl/tasks/take_cup/init/bc_latest_ckpt \
    --log_dir=/mnt/hil-serl/logs/bc \
    # --eval_n_trajs=100 \
