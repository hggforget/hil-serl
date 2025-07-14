export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export DISPLAY=:0 && \

python ../../train_rlpd.py "$@" \
     --exp_name=take_mango \
     --checkpoint_path=/mnt/hil-serl/first_run \
     --bc_checkpoint_path=/mnt/hil-serl/examples/experiments/take_cup/bc_ckpt \
     --actor \