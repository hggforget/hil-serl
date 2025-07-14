#!/home/jd/miniconda3/envs/hilserl/bin/python python

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export DISPLAY=:0 && \

/home/jd/miniconda3/envs/hilserl/bin/python ../../train_rlpd.py "$@" \
    --exp_name=take_mango \
    --checkpoint_path=first_run \
    --demo_path=... \
    --learner \