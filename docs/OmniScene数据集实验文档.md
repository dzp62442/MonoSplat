# OmniScene 数据集实验规划

## 配置概览
- **数据集配置文件（对齐 depthsplat）**：在 `config/dataset/omniscene.yaml` 里复用 depthsplat 的结构，保持 `defaults: view_sampler=all`、`image_shape`/`background_color`/`cameras_are_circular`、`make_baseline_1=false` 等字段，`name` 指向 `omniscene`、`roots` 定位 `datasets/omniscene`。MonoSplat 的 `DatasetCfgCommon` 只含基础字段，因此 `DatasetOmniSceneCfg` 需额外声明 `near/far/skip_bad_shape/train_times_per_scene/highres` 等，以便 1:1 复刻 depthsplat 的抽样/近远平面设定。
- **实验配置文件（覆盖 datasets & 节奏字段）**：README 中的训练/测试命令都会追加 `+experiment=re10k`，因此我们也会以该 experiment 为模板创建 `config/experiment/omniscene_112x200.yaml`、`config/experiment/omniscene_224x400.yaml`：继续引用 re10k 中对 `/model/encoder: costvolume`、`/model/decoder: splatting_cuda`、`/loss: [mse, lpips]` 的覆写，同时将 depthsplat 要求的节奏配置（`data_loader.{train,val,test}.batch_size=1`、`trainer.max_steps=100000`、`trainer.val_check_interval=0.01` 等）写入这些新 experiment 中，使训练/验证/测试频率与 depthsplat 完全一致。
- **模型/训练参数（沿用本项目 re10k 设定）**：除节奏字段外，其余模型与训练配置继续继承 re10k experiment：包含 costvolume 编码器、splatting CUDA 解码器、MSE+LPIPS 损失组合、`optimizer.lr=2e-4`、`warm_up_steps=2000`、`train.depth_mode=null` 等。这样可以保证 OmniScene 的实现完全遵循 MonoSplat 既有的网络结构与优化策略，仅在数据与节奏部分做差异化。
- **运行节奏（按 depthsplat 抽样与频率）**：Lightning 层面的 `trainer.val_check_interval`、`trainer.max_steps`、`data_loader` 中的 batch size 等配置由上述 OmniScene experiment 设置为 depthsplat 的取值；同时 `DatasetOmniScene` 会继承 depthsplat 的 bin 抽样策略（train 全量、val 采样 10 个、test 每 14 个取 1 个）以及 `test.eval_time_skip_steps=5` 等细化开关。需要切换 mini-test/完整 test 时，通过 dataset 配置参数控制，无需修改模型或优化器逻辑。
- **train/test 额外开关**：为接入 OmniScene 专属特性，需要给 `config/main.yaml` 的 `train` 区域新增 `use_dynamic_mask`、`l1_loss`、`train_ignore_large_loss` 等字段（默认 false/0，保持对 re10k 的兼容），并在 `test` 区域新增 `save_video_omniscene`。这些字段在 re10k 实验中保持默认值，只有 OmniScene 覆盖时才会开启。

## 数据加载流程
1. **入口注册**：在 `src/dataset/__init__.py` 把 `"omniscene" -> DatasetOmniScene` 加入 `DATASETS`。Hydra 一旦读到 `dataset.name=omniscene` 就会实例化新类，`DataModule` 逻辑无需改动。
2. **数据类实现**：在 `src/dataset/dataset_omniscene.py` 复刻 depthsplat 的 `DatasetOmniScene`，保留 `bins_train_3.2m.json / bins_val_3.2m.json` / `bin_infos_3.2m/*.pkl` 的读取、按 stage 抽样（train 全量，val 采样 10 个，test 默认为 mini-test）。MonoSplat 目前只有 `Stage = [train,val,test]`，因此 demo/super-mini 流程会并入 test，或通过额外 flag 控制。类本身可以继承 `torch.utils.data.Dataset`，`DataModule` 会检测 `IterableDataset` 与否来自行设置 `shuffle`。
3. **视图组织**：DepthSplat 直接把 6 个环视 key-frame 当作 context，再把 index `[1,2]` 的非关键帧拼到 target。MonoSplat 沿用相同策略，因此 `view_sampler` 在此数据集不会参与采样（保持 `defaults.view_sampler=all` 即可）。最终返回的 `context/target` 字段必须与 `src/dataset/types.BatchedViews` 对齐，并额外附加 `target["masks"]` 供掩码损失使用。
4. **图像/掩码/参数加载**：可直接复用 depthsplat 的 `utils_omniscene.load_info`、`load_conditions`，包括：按 `samples_small`/`sweeps_small` 读取图像、重标定内参、加载 `*_mask_small` 掩码。需要注意在 MonoSplat 中也要把这些工具放在 `src/dataset/utils_omniscene.py`，并在 `__init__` 时根据配置设置 `self.near/self.far`，以匹配 decoder 的射线范围。
5. **差异处理**：MonoSplat 现有的 `DatasetRE10k` 使用 chunk+view_sampler 迭代，与 OmniScene 的逐 bin 结构完全不同，因此无法直接套用当前数据加载方式，但 depthsplat 的实现可以整体搬迁，只需把依赖（如 `train_times_per_scene`、`highres`）整理进新 cfg 即可。

## 主程序调用与模型改动
- **Hydra 入口**：运行命令保持 `python -m src.main +experiment=omniscene_224x400 mode=train ...`。MonoSplat 的 `src/main.py` 不需要大改，只需在构建 `ModelWrapper` 时向 `TrainCfg`/`TestCfg` 传入新增字段，并确保 logger 名称/输出目录允许与 re10k 共存。
- **Mask 逻辑**：当前 MonoSplat 的 `ModelWrapper.training_step` 与 loss 函数都没有掩码概念。需要像 depthsplat 那样扩展 `TrainCfg`（新增 `use_dynamic_mask`, `l1_loss`, `train_ignore_large_loss`），并在 `training_step` 中根据 `batch["target"]["masks"]` 生成 `valid_depth_mask`，交给 `LossMse`、`LossLpips` 等损失函数。Loss 类的 `forward` 签名也要同步扩展，默认情况下可传 `None` 以兼容旧数据集。
- **测试视频**：若 `test.save_video_omniscene=true`，在 `ModelWrapper.test_step` 末尾参考 depthsplat 的实现，读取 batch 中最后 6 个姿态，生成自定义轨迹、调用 decoder 渲染，并将 mp4 保存到 `outputs/test/<run>/videos_omniscene/scene.mp4`。该功能只在 OmniScene 任务启用，其他数据集默认保持当前逻辑。
- **评估节奏**：MonoSplat 不会自动插入额外的 evaluation dataloader，因此如需训练过程中执行 mini-test，可通过 `trainer.val_check_interval` 调整验证频率，或在 README / 脚本中提供“训练完再跑一次 `mode=test`”的命令。

## PCC 指标补充方案
1. **相对深度加载（仅 test）**：
   - 在 `src/dataset/utils_omniscene.py::load_conditions` 增加 `load_rel_depth` 开关；为 `True` 时读取 DepthAnything-v2 的 disparity（`samples_dpt_small`/`sweeps_dpt_small` 下 `.npy`），如遇 resize 同步做双线性缩放；随后按 depthsplat 的做法将 disparity 转为相对深度（限制最远/最近比例为 50，再做 min-max 归一化到 `[0, 1]`），返回 `rel_depth` 张量。
   - `DatasetOmniScene` 在 `__init__` 中维护 `self.load_rel_depth`，默认 `stage == "test"` 时启用；其它阶段强制关闭以降低 IO。`__getitem__` 在 `target` 中追加 `rel_depth`（输入/输出拼接后对齐），未启用时显式为 `None`。
   - 若启用了 patch shim（`src/dataset/shims/patch_shim.py`），需像 `masks` 一样对 `rel_depth` 做中心裁剪，保证与图像/内参对齐。
2. **渲染深度结果**：
   - 本项目的解码器支持深度渲染：`Decoder.forward` 在 `depth_mode` 非空时返回 `DecoderOutput.depth`（形状 `[b, v, h, w]`），当前 `ModelWrapper.test_step` 传入 `depth_mode=None`，因此默认没有深度输出。
   - 计算 PCC 时需在 test 阶段为 `decoder.forward` 指定 `depth_mode="depth"`（或与相对深度一致的模式），确保 `output.depth` 可用；建议仅在 `test.compute_scores=true` 且 `target["rel_depth"]` 存在时开启，以减少额外渲染开销。
3. **PCC 指标计算位置**：
   - 参照 depthsplat，将 `compute_pcc` 与 `compute_psnr/compute_ssim/compute_lpips` 放在同一文件 `src/evaluation/metrics.py`，使用 `torchmetrics.PearsonCorrCoef`，输入为 `(rel_depth, pred_depth)` 的 `[b, h, w]` 张量并在像素维度上计算相关系数。
4. **PCC 统计与汇总**：
   - 复用 `ModelWrapper.test_step` 中的 `test_step_outputs` 聚合逻辑：当 `output.depth` 与 `target["rel_depth"]` 均存在时追加 `pcc`；`on_test_end` 与 PSNR/SSIM/LPIPS 共用汇总/落盘流程（写入 `scores_pcc_all.json`，并在 `scores_all_avg.json` 中加入 `pcc` 均值）。

## 与 depthsplat 的差异与复用策略
1. **可直接迁移的部分**：`config/dataset/omniscene.yaml`、`config/experiment/omniscene_*.yaml`、`src/dataset/dataset_omniscene.py`、`src/dataset/utils_omniscene.py` 的主体逻辑都可直接复用，只需要把少量依赖（如 `train_times_per_scene`）封装到新 cfg 中。
2. **需要适配的部分**：MonoSplat 的损失定义、`TrainCfg/TestCfg` 以及 `ModelWrapper` 缺少 depthsplat 的掩码/可视化字段，必须添加新参数后再在 `training_step/test_step` 中显式处理；另外 Stage 仅有 train/val/test，因此 demo 模式等附加分支要通过配置开关模拟。
3. **无法直接复用的部分**：depthsplat 有 `eval_model_every_n_val`、`save_video_omniscene` 等高级调度逻辑，MonoSplat 中不存在，需要结合 Lightning 的默认行为重新设计触发点，或留待实验脚本手动控制。

## 实现路线
1. **配置层**：新增 dataset/experiment yaml，并在 `config/main.yaml` 补充 train/test 新字段，保持默认值与现有任务兼容。
2. **数据层**：移植 `DatasetOmniScene` 与辅助函数，注册到 `DATASETS`，并在 README 中补充 OmniScene 的运行命令。
3. **模型层**：扩展 `TrainCfg`/`TestCfg` dataclass、`ModelWrapper`、`LossMse`/`LossLpips` 等，使其支持动态掩码和 OmniScene 专属的视频导出。
4. **验证**：使用 `+experiment=omniscene_224x400` 在少量 bin（val/test 采样）上跑通 `mode=train`（仅跑若干 step 验证数据流）与 `mode=test`（验证渲染+视频生成），待文档审核后再逐步完善实现。
