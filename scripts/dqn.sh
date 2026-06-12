python run_dqn.py \
  algorithm=dqn \
  train_action_deterministic=true \
  env=atari \
  env.name=PongNoFrameskip-v4 \
  env.num_training_envs=1 \
  save_interval=100000 \
  test_interval=10000 \
  train_log_interval=10000 \
  total_epoch=1000000 \
  algorithm.learning_rate=1e-4 \
  algorithm.buffer_size=500000