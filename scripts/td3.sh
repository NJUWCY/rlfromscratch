python run_td3.py \
  algorithm=td3 \
  train_action_deterministic=false \
  env=mujoco \
  env.name=Hopper-v5 \
  env.num_training_envs=1 \
  save_interval=100000 \
  test_interval=10000 \
  train_log_interval=10000 \
  total_epoch=1500000 \
  algorithm.buffer_size=1000000 \
  interact_per_epoch=1 \
  start_train_step=25000 \
  "$@"