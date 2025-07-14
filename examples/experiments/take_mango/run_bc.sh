export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export DISPLAY=:0 && \

/home/jd/miniconda3/envs/hilserl/bin/python  ../../train_bc.py "$@" \
    --exp_name=take_mango \
    --demo_path=/mnt/hil-serl/pickles/bc/demos_take_mango_bc_relative_init.pickle \
    --bc_checkpoint_path=/mnt/hil-serl/examples/experiments/take_cup/bc_init_norm_minmax_noforzen_1e-4_ckpt \
    --log_dir=/mnt/hil-serl/logs/bc \
    # --eval_n_trajs=100 \