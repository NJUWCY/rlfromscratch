python run_ppo.py \
  algorithm=ppo \
  total_epoch=5000 \
  train_log_interval=10 \
  save_interval=1000 \
  test_interval=50 \
  train_action_deterministic=false \
  algorithm.rescale=true \
  algorithm.collect_traj=false \
  env.name=HalfCheetah-v5 \
  algorithm.buffer_name=ReplayBuffer \
  interact_per_epoch=2048 \
  algorithm.update_epochs=5 \
  algorithm.minibatch_size=64 \
  algorithm.use_grad_clip=false

