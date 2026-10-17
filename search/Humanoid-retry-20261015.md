# Humanoid DAC 补跑

一个两卡实例运行以下命令，以前台方式等待所有任务结束，适用于平台“任务完成释放”。HyPESAC 的四个中断任务已到约 93% 的预算，用户查看效果后决定不补跑。

DAC：
```bash
bash /home/ubuntu/wangchenyang/rlzero/rlfromscratch/search/DACSAC/humanoid_retry.sh
```

| 脚本 | 待补齐参数 | 任务数 | 每 GPU 并发 | 每次完整预算 |
|---|---|---:|---:|---:|
| DAC | 自动温度 target_entropy {-0.5,-1.0} × actor_lr {1e-4,3e-4} × critic_lr {1e-4,3e-4} | 8 | 4 | 1,500,000 steps |

DAC 的 4 个 fixed-alpha 实验及 HyPESAC 的 8 个较小判别器学习率实验已经完成，不在补跑范围。DAC 此前有 4 组未启动的 target_entropy=-1.0 参数，本次一并补齐。

现有 checkpoint 只有 agent 的 state_dict 和配置，没有判别器、回放池、优化器、温度更新和训练计数等完整状态。`algorithm/basealgorithm.py` 的 save 注释也明确它用于推理。因此这次按原始参数和 seed=0 从头完整重跑，不加载旧模型并伪装成原训练的续训。

DAC 沿用同一份 `humanoid_wave1_fixed5.hdf5`，5 条专家轨迹、subsample_frequency=1。学习参数保持原来配置，额外启用 TensorBoard 日志备份。脚本使用 offline SwanLab，同时保留 stdout 和 TensorBoard。

输出位于新目录 `outputs/Humanoid-v5-DACSAC-wave1-retry/<timestamp>`。原实验文件保留。

采用已有 parallel_search.py 调度，每次训练绑定一张 GPU；前台等待任务，任一任务失败会返回非零状态。单任务最长 48 小时。固定专家文件和两张 GPU 的可用性在实际运行前检查。

脚本可以用 `--dry-run` 查看任务，不启动训练。此次在 CPU 实例上验证脚本语法、任务数量与全部 Hydra 配置；GPU 训练待新实例执行。

离线日志上传时使用 SwanLab 0.10.2 或包含时间窗口修正的新版本，并从云端读回指标确认。不要仅凭 Upload complete 判断成功。详见根目录 `SwanLab上传空曲线核查_2026-10-15.md`。
