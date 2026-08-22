# M5：高精度种子扩展 + 可靠性门控（XD）

这个实验针对当前 global-768 的主要问题：冻结 baseline 的 top 10% 异常片段很准，但覆盖率低；异常视频最低分片段又不是真正可靠的负样本。

方法只用训练数据建立可靠性图 `q`：

- 分数必须高于纯正常训练视频的 95% 分位；
- hidden state 必须与该视频 top-10% 高分种子原型相似；
- 两者相乘得到 `q`，取 `q` 最高的 30% 作为软正样本候选；
- 负参考只来自纯正常训练视频，不使用异常视频低分片段；
- 训练损失仍是原始 residual-injection 损失，baseline 权重完全冻结；
- 验证/测试使用 `base_logits + q * (residual_logits - base_logits)`。`q=0` 时严格回到 baseline。

所有路径均相对 `vadclip_code`。`../vad_data/.../manifest.csv` 只是已完成的共享 CLIP hidden **数据缓存**；本目录不导入、也不依赖 `vad_code` 的任何代码。

## 0. 先做标签隔离的质量诊断

这一条读取 XD 测试帧标注，只用于判断种子扩展是否真的提高了“正样本覆盖率且仍保持精度”。它不会生成训练数据、更新参数或参与选模。

```bash
python experiments/vadclip_seed_expand_reliability_gate/diagnose_seed_expansion_quality.py \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_xd_residual/compare_bottom10/pseudo_scores/group_scores.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --model-path ../vadclip_data/model/vadclip_xd.pth \
  --gt-path VadCLIP/list/gt.npy \
  --out-dir ../vadclip_data/diagnostics/seed_expand_reliability/xd \
  --vadclip-root VadCLIP \
  --seed-top-p 0.10 \
  --expand-top-p 0.30 \
  --normal-score-quantile 0.95 \
  --score-temperature 0.05 \
  --normal-score-snippets-per-video 256 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6 \
  --device cuda
```

看 `../vadclip_data/diagnostics/seed_expand_reliability/xd/summary.json`：`reliability_expanded_top.micro_positive_recall` 应明显高于 `seed_top`，且 precision 不能明显坍塌；同时 `normal_reliability` 应较低。诊断不通过就不要进入正式训练。

## 1. 正式实验

第一次运行下面命令即可；中断后直接重复相同命令会复用已完成的逐视频产物。只有需要从头清理本实验输出时才给对应命令加 `--clean`。

### 1.1 选全局 768 个神经元

```bash
python experiments/vadclip_seed_expand_reliability_gate/select_neurons_seed_expand.py \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_xd_residual/compare_bottom10/pseudo_scores/group_scores.csv \
  --out-dir ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/neurons \
  --seed-top-p 0.10 \
  --expand-top-p 0.30 \
  --topk-global 768 \
  --normal-score-quantile 0.95 \
  --score-temperature 0.05 \
  --normal-score-snippets-per-video 256 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6
```

输出：`../vadclip_data/work_xd_residual/seed_expand_reliability_global768/neurons/selected_neurons.json`。其中记录了 normal calibration、可靠性公式和全局 768 个坐标。

### 1.2 写 concat 合约并构建特征

```bash
python experiments/vadclip_neuron_injection/make_concat_neuron_json.py \
  --source-neuron-json ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/concat_neurons \
  --neuron-width 768 \
  --clip-dim 512

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/features/train \
  --out-csv ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/xd_concat_train.csv \
  --keep-missing

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/features/test \
  --out-csv ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/xd_concat_test.csv
```

### 1.3 原始损失训练；以门控后的 AP2 选最佳模型

```bash
python experiments/vadclip_seed_expand_reliability_gate/train_seed_expand_reliability_xd.py \
  --dataset xd \
  --train-list ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/xd_concat_test.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

训练损失、学习率、batch size、每 epoch 验证和“最高 language AP2 选模”均保持正式 VadCLIP residual 协议；只有验证分数采用训练前固定的 `q` 门控。中断续训加 `--resume`。

### 1.4 最终测试

```bash
python experiments/vadclip_seed_expand_reliability_gate/test_seed_expand_reliability_xd.py \
  --dataset xd \
  --test-list ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/xd_concat_test.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/seed_expand_reliability_global768/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

最终指标在 `../vadclip_data/work_xd_residual/seed_expand_reliability_global768/evaluation/metrics.json`；逐视频可恢复结果在 `evaluation/per_video/`；门控实际使用强度在 `evaluation/reliability_summary.json`。
