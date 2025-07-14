export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
export DISPLAY=:0 && \

python examples/train_rlpd_bostrapped.py "$@" \
     --exp_name=take_cup \
    --checkpoint_path=/mnt/hil-serl/tasks/take_cup/init/boostrapped_run_4 \
    --bc_checkpoint_path=/mnt/hil-serl/tasks/take_cup/init/bc_minmax_noforzen_1e-4_split_ckpt \
     --actor \