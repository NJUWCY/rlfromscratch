export SWANLAB_API_KEY=yyKpLHGppV78RFW0p1PNQ
python run_ppo.py \
  algorithm=ppo \
  total_epoch=5000 \
  train_log_interval=10 \
  save_interval=1000 \
  test_interval=50 \
  train_action_deterministic=false \
  algorithm.rescale=false \
  algorithm.collect_traj=false \
  env.name=Hopper-v5 \
  algorithm.buffer_name=ReplayBuffer \
  interact_per_epoch=256 \
  algorithm.update_epochs=5 \
  algorithm.minibatch_size=128 \
  algorithm.use_grad_clip=true \
  algorithm.advan_norm=false \
  env.obs_norm=true \
  algorithm.rescale=false \
  algorithm.return_scaling=true \
  "$@"


