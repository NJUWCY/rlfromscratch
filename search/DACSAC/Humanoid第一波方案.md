# Humanoid DAC 第一波候选、资源与离线上传

2026-10-13，当前代码2441a16，SwanLab 0.8.2。该文档和脚本是第一波候选方案；尚未在GPU上训练或测吞吐。新增脚本已在无GPU实例做配置解析、调度dry-run、CPU内存探测和离线日志写入验证。原有训练源代码未改，lr_decay在本波保持关闭。

## 已确定的专家数据

按照用户与学长确认：`trajectory_num=5`、`subsample_frequency=1`。`humanoid_wave1_prepare.py`从原100条Humanoid专家轨迹中用独立selection_seed=0选择[26,30,50,61,81]，保存为新文件`outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5`，原专家文件保持原样。五条均1000步，回报约5434.49、5431.21、5420.86、5430.06、5431.26。旁边JSON记录来源、所选索引和文件SHA256；后续运行验证哈希，避免不同配置/seed悄悄换专家数据。

## 第一波：完整的2×2×3网格，共12组

| 变量 | 候选 |
|---|---|
| actor_lr | 1e-4、3e-4 |
| critic_lr | 1e-4、3e-4 |
| 温度策略 | 固定alpha=0.1；自动alpha且每维target_entropy=-0.5；自动alpha且每维target_entropy=-1.0 |

三种温度策略各有四组actor/critic学习率组合。自动温度初始值0.1，temp_lr=3e-4，update_log_temp=true；固定温度时target_entropy/temp_lr没有作用，不将其重复组合。Humanoid是17维动作，所以两档自动温度的实际总目标是-8.5、-17。

每组seed=0、150万环境交互步（num_training_envs=1、interact_per_epoch=1），12组总计1800万步；test_interval=10000、10个确定性评估回合、8个评估子环境。保持每步一个SAC更新、一个判别器更新，专家完整保留5000个转移。主排名用最后10次测试均值，辅以后半程回报曲线和崩溃幅度。初筛仅单seed，不能称为最优或稳定性结论；优胜候选后续补seed 1、2。旧记录约114万步才出现高分，因此不因20万/30万步仍低回报而提前淘汰。

判别器先显式固定当前默认：Adam、disc_lr=1e-4、[256,256] tanh、spectral_norm=true、weight_decay=0、GP系数10、目标梯度范数1、AIRL/logit奖励。SAC网络[256,256]、gamma=0.99、tau=0.01、obs_norm=false、absorbing=true、nstep=1。它们是本波控制变量，未证明最优。若本波全失败，不能据此否定DAC或该温度/学习率范围，应优先检查训练诊断并进行判别器学习率/GP阶段。

本波选择明确、完整的小网格，而不在12个名额里稀疏搜索所有变量。判别器lr、GP、optimizer和衰减留到第二波；若换optimizer要允许局部重新调lr，不能把第一波Adam参数永久锁为全局最优。当前代码没有lr_decay，本波不需要为其改训练核心代码。

## 内存实测与CPU申请依据

无GPU实例实测4vCPU/16GiB（cgroup cpu.max为400000 100000；memory.max为17179869184），宿主机显示的192线程不等于可用CPU配额。

`humanoid_resource_probe.py`在CPU上创建真实Humanoid环境、百万容量AbsorbingReplayBuffer、5条专家数据，触碰全部buffer页后启动8个评估子环境：

| 项目 | 数值 |
|---|---:|
| replay的实际数组容量 | 2,880,000,000字节 = 2.682 GiB |
| 主进程，填满replay并加载专家后的PSS | 3.23 GiB |
| 加8个评估子环境后的进程树PSS | 6.19 GiB |
| 同一进程树RSS简单相加 | 8.03 GiB，含重复共享页，不能直接当成唯一物理占用 |

探测不含CUDA、完整网络/优化器、训练批次和SwanLab的全部开销，不是完整训练峰值。历史训练主进程RSS最大约4.4～4.6GiB；GPU记录约1.1～1.2GiB、平均利用率约11%，但部分旧网络较小，而且GPU指标是整卡统计。申请建议每个并发任务8vCPU、16GiB内存，约12GiB/任务可作为紧凑下限而非首选。8个评估子进程只在评估阶段出现；CPU线程库已固定每进程1线程。

## GPU与耗时：先给保守方案，再校准同卡并发

历史约150万步用6.0～6.2h，当前判别器网络/实现不同，成功策略会延长评估回合；申请规划先用每组6～10h。以下是假设GPU与先前5090相近、CPU/内存足够的一任务一卡估算，不是本轮GPU实测：

| GPU数量 | 并发训练任务 | CPU | 内存 | 12组预计墙钟时间 |
|---|---:|---:|---:|---:|
| 1×RTX5090 | 1 | 8vCPU | 16GiB | 72～120h |
| 2×RTX5090 | 2 | 16vCPU | 32GiB | 36～60h |
| 4×RTX5090 | 4 | 32vCPU | 64GiB | 18～30h |

推荐申请**2×RTX5090、32vCPU、64GiB**，留足4个并发任务的资源。先测试每卡1任务与每卡2任务：如果同卡两任务的单任务变慢系数s约1.3～1.8，则12组/4并发/3轮的估算为3×(6～10h)×s≈24～54h。该s是假设，尚未实测；若没有吞吐收益，退回每卡1任务，按36～60h完成。因此可按约60h预订，短时校准后收紧预测。两张卡60h为120物理GPU小时，符合之前暂定168 GPU小时总预算，但会留较少余量给后续验证。

若优先最短完工，选择4张5090、32vCPU、64GiB，每卡1任务，约18～30h。当前launcher按三种温度分组，各4个候选，因此最多4个任务同时有效；4GPU每卡2任务不会提高这一版首波调度的并发数，无需为8个任务申请资源。

吞吐校准用同一150万步方案的小步数副本（例如--epochs 10000），测包含评估/日志的总步数÷墙钟时间；短副本只用于速度/异常检查，不参与学习效果排名。用同一温度组运行四候选，分别比较--jobs-per-gpu 1/2，记录整波耗时、PSS和显存峰值。小步数副本不代表训练成功后长评估的开销，完工预测还应在首批长跑后刷新。脚本没有自动断点续训，失败重跑应使用新输出目录，不能覆盖原日志；24h/trial硬超时是挂死保护，不是早期回报剪枝。

使用持久化磁盘保存代码、专家数据、每个trial的Hydra配置、manifest、stdout.log、SwanLab run目录和模型。日志本身通常较小，模型大小随网络增加；建议为本波预留10～20GiB可用磁盘，保留原专家文件所需空间另计。实例过期前确认输出所在卷会保留；只复制.swanlab单文件不够，应保留完整run目录。

## SwanLab：本波默认offline

理由是之前已经出现DNS解析/在线上传重试，本波是多任务长跑，不依赖实时网页排名；离线让训练阶段不依赖云端连通性。当前0.8.2实际测试`SWANLAB_MODE=offline`后，成功写出2条scalar和finish事件，run_id也记录在本地文件中。无需改logger，环境变量能生效。offline有本地记录，disabled没有，两者不能互换。

训练完成后，联网环境用`swanlab sync <完整run目录>`上传即可，不需要重新训练。[官方sync文档](https://docs.swanlab.cn/api/cli-swanlab-sync.html)支持批量路径以及`--id`对原实验补传差异数据。官方默认sync可能生成新实验；脚本从本地start记录取稳定run_id并显式传--id，方便同一记录重试时保持身份。这个辅助脚本已验证能生成命令，但没有实际向云端上传；首次上传/补传仍需用已有账号登录并检查云端结果。若日志版本变化，使用匹配的SwanLab版本同步，本波固定0.8.2。

在线模式也能记录本地日志，已有源码会重试上传；但不保证长期断网后网页自动恢复到完整状态。选择online时不要启用跳过本地存储的选项，记录原实验ID；必要时再以`sync <run-dir> --id <原实验ID>`补传。offline的主要代价是网页看不到实时曲线，可用本地stdout日志判断进度，或定期同步**已经结束的**trial。首次上传失败后只重试上传，不重跑训练。[模式文档](https://docs.swanlab.cn/api/py-init.html)

## 使用命令

在GPU实例上，下面第一行bash命令直接前台运行完整第一波：单GPU0挂4个任务，共12组，每组150万步，offline默认、每组48小时挂死保护。入口用exec等待训练结束，支持平台任务完成释放。当前无GPU实例无法启动完整训练。

```bash
cd /home/ubuntu/wangchenyang/rlzero/rlfromscratch
bash search/DACSAC/humanoid_wave1_run.sh
bash search/DACSAC/humanoid_wave1_run.sh --dry-run --gpus 0,1 --jobs-per-gpu 2
```

新GPU实例上，先做吞吐校准，确认双并发值得启用后启动完整第一波：

```bash
# 小步数速度检查，可按1/2并发分别测试；不是收敛验证
bash search/DACSAC/humanoid_wave1_run.sh --run --gpus 0,1 --jobs-per-gpu 1 --epochs 10000
bash search/DACSAC/humanoid_wave1_run.sh --run --gpus 0,1 --jobs-per-gpu 2 --epochs 10000

# 完整候选波，每组150万步；短时测试输出与正式输出分别保存
bash search/DACSAC/humanoid_wave1_run.sh --run --gpus 0,1 --jobs-per-gpu 2

# 四GPU，每卡一任务
bash search/DACSAC/humanoid_wave1_run.sh --run --gpus 0,1,2,3 --jobs-per-gpu 1
```

完整运行前会自动验证/准备固定五条专家数据；不提供CPU训练fallback。当前默认offline，想显式用online可传`--mode online`。输出每次有独立时间戳目录，三组温度各有四个独立trial目录。

补传辅助脚本默认只打印命令，--upload才执行：

```bash
# 按实际波次目录替换<时间戳>；以下python应来自rlzero环境
python search/DACSAC/humanoid_sync_logs.py outputs/Humanoid-v5-DACSAC-wave1/<时间戳>
python search/DACSAC/humanoid_sync_logs.py outputs/Humanoid-v5-DACSAC-wave1/<时间戳> --upload
```

默认跳过没有finish事件的run，避免同步仍在写入的实验。若训练进程已停止/崩溃，可加--include-incomplete上传已有记录。上传命令使用现有SwanLab登录态或API key环境变量，脚本不硬编码密钥。
