# VadCLIP selected-neuron Adapter（XD）

这是对 global-768 的下一步实验：**不训练 VadCLIP baseline，也不训练 CLIP 原参数**；只在 CLIP ViT-B/16 的 12 个冻结 block 之间插入小 Adapter。每个 Adapter 只读取、修改该层被 global-768 选中的 CLS 神经元，并把修改后的 CLS token 送进下一层冻结 CLIP。

它不是原来的“768D 拼接后预测一个 512D 残差”：旧方法只能读取缓存，不能让改动经过 CLIP 的后续层。这里每个 Adapter 的输出为零初始化，最终输入使用 **特征锚定**：`原 512D 缓存 + (在线 Adapter-CLIP − 在线冻结 CLIP)`。因此第 0 步严格等价于原 VadCLIP；训练的弱监督梯度仍可从原 VadCLIP loss 反传到各层 Adapter。

## 可复用的数据

只读已有产物，不导入其他项目代码：

- `../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json`：global-768 的层号和神经元下标；
- `../vad_data/work_xd/clip_hidden_stride16_*_8gpu/manifest.csv` 及它指向的 hidden `.npz`：复用 `video_path` 和精确 `frame_indices`，保证重新跑原视频时抽到与选择阶段相同的 snippet；
- `../vad_data/work_xd/xd_*_local.csv`：复用原 512D 特征路径和弱标签。它是冻结的 VadCLIP **特征锚点**；按既有拼接脚本的规则裁去 hidden 末尾多出的 snippet，在线 Adapter 只向它添加真实 CLIP 内部改动产生的差分。

因为 `.npz` 中的 768D hidden 是已经计算完的结果，不能反向传播到后续 CLIP 层，所以训练必须从 `video_path` 在线重跑 CLIP。XD 原始特征文件名末尾的 `__0` 到 `__9` 是官方十裁剪编号；新代码按编号独立复刻相同空间裁剪后再送入 CLIP。已验证当前原视频重算结果与官方发布 512D 缓存不完全一致，所以不替换原缓存，而是只把 Adapter 相对同一在线冻结 CLIP 的差分加回原缓存。新代码只读取同级数据目录，没有跨项目 Python import。

## 先做零起点一致性检查

从 `vadclip_code` 运行。必须先通过：它检查 Adapter 仍为零时，特征锚定后的每个 512D snippet 是否严格等于原 VadCLIP 缓存；报告还会记录未锚定在线特征与旧缓存的差异，供数据溯源诊断。

```bash
python experiments/vadclip_selected_neuron_adapter/verify_zero_adapter_xd.py \
  --dataset xd \
  --source-list ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --out-dir ../vadclip_data/work_xd_residual/selected_neuron_adapter_global768/zero_check \
  --vadclip-root VadCLIP \
  --samples 3 \
  --frame-batch-size 128 \
  --atol 3e-4 \
  --rtol 3e-4 \
  --device cuda
```

产物：`../vadclip_data/work_xd_residual/selected_neuron_adapter_global768/zero_check/zero_adapter_alignment.json`。

## 正式训练

下面保持 VadCLIP XD 的 `batch-size=96`、`lr=1e-5`、10 epoch、每 epoch 测试一次和 AP2 选最优模型。原始视频长度不同，在线 CLIP 必须逐视频跑；代码会把 96 个视频的梯度累积后再做一次 AdamW 更新，数学上对应官方 batch loss 的平均，不改变 snippet 和时间处理。`frame-batch-size` 只控制一次送入 CLIP 的帧数量，不改变视频内容或顺序。

当前 XD 训练 CSV 有 4 个原视频、共 40 个切分特征行，在已有 raw-video/hidden manifest 中不存在；不能凭空在线重跑它们。正式命令显式加入 `--skip-missing-train-manifest`，并把名单写入 `training/skipped_train_missing_manifest.csv`。测试 manifest 已完整匹配 800 个测试行，测试侧绝不允许跳过。

```bash
python experiments/vadclip_selected_neuron_adapter/train_selected_neuron_adapter_xd.py \
  --dataset xd \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/selected_neuron_adapter_global768/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/selected_neuron_adapter_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/selected_neuron_adapter_global768/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --frame-batch-size 128 \
  --adapter-rank 8 \
  --skip-missing-train-manifest \
  --seed 234 \
  --device cuda
```

中断后加 `--resume` 重跑同一条训练命令；需要重新开始时，在**确认输出目录无保留需要后**加 `--clean`，不能与 `--resume` 同时使用。

## 独立测试

训练里的测试用于按 baseline 规则挑选最佳 epoch；下列命令才是保存最终逐视频分数和最终指标的独立测试。中断后默认复用已完成的 `per_video/*.npz`；完全重测加 `--clean`。

```bash
python experiments/vadclip_selected_neuron_adapter/test_selected_neuron_adapter_xd.py \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/selected_neuron_adapter_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/selected_neuron_adapter_global768/evaluation \
  --vadclip-root VadCLIP \
  --frame-batch-size 128 \
  --adapter-rank 8 \
  --device cuda
```

结果在 `../vadclip_data/work_xd_residual/selected_neuron_adapter_global768/evaluation/metrics.json`，逐视频可恢复结果在同目录 `per_video/`。训练记录为 `training/history.csv`，最后可恢复训练状态为 `training/checkpoint_last.pth`。
