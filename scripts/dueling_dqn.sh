export SWANLAB_API_KEY=yyKpLHGppV78RFW0p1PNQ
python run_dqn.py \
  algorithm=dqn \
  train_action_deterministic=true \
  env=atari \
  env.name=PongNoFrameskip-v4 \
  env.num_training_envs=1 \
  save_interval=100000 \
  test_interval=10000 \
  train_log_interval=5000 \
  total_epoch=1000000 \
  start_train_step=50000 \
  algorithm.buffer_size=1000000 \
  algorithm.huber_loss=false \
  algorithm.end_epsilon=0.1 \
  algorithm.target_update_interval=1000 \
  algorithm.target_update_tau=1 \
  algorithm.doubledqn=true \
  algorithm.buffer_name="PrioritizedReplayBuffer" \
  algorithm.weight_batch_norm=true \
  algorithm.learning_rate=5e-5 \
  algorithm.dueling_network=true \
  algorithm.conv_gradient_rescale=true \
  algorithm.use_grad_clip=true \
  "$@"
