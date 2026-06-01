python run_sac.py \
  algorithm=sac \
  train_action_deterministic=false \
  env=mujoco \
  env.name=HalfCheetah-v5 \
  env.num_training_envs=1 \
  save_interval=100000 \
  test_interval=10000 \
  train_log_interval=50000 \
  total_epoch=5000000 \
  algorithm.buffer_size=1000000 \
  algorithm.learn_temp=true \
  interact_per_epoch=1 \
  "$@"