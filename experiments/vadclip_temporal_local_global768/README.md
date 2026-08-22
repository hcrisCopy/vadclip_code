# VadCLIP temporal-local global-768

M3 只改变 768 个神经元的选择方式，不改变 VadCLIP、冻结策略、原始三项训练 loss、AP2 选模或测试流程。

每个异常训练视频中，baseline 分数 top10% 的 snippet 是伪正样本。它们根据相邻 snippet 分数得到连续性权重；同一视频中远离这些正样本邻域的低分 snippet 是背景。神经元必须同时稳定地区分：

```text
伪正样本 vs 同视频背景
伪正样本 vs 纯正常训练视频参考
```

不会读取测试数据或帧级标注。`../vad_data` 只复用已经完成的 CLIP hidden cache 和训练 CSV；不导入 `vad_code`，所有新产物写入 `../vadclip_data`。

从 `vadclip_code` 运行。选择阶段可中断重跑：`normal_stats.npz` 和 `per_video_contributions/*.npz` 会被复用。若要从头重选，在第 1 条命令加 `--clean`；训练按原脚本的 `--resume` 续跑；测试按原脚本逐视频复用。

## XD-Violence 正式命令

训练伪分数与 top-vs-normal 实验完全相同，直接复用已有结果：

```bash
python experiments/vadclip_temporal_local_global768/select_neurons_temporal_local.py \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_xd_residual/top_vs_normal_global768/pseudo_scores/group_scores.csv \
  --out-dir ../vadclip_data/work_xd_residual/temporal_local_global768/neurons \
  --top-p 0.10 \
  --background-p 0.50 \
  --context-radius 1 \
  --topk-global 768 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6

python experiments/vadclip_neuron_injection/make_concat_neuron_json.py \
  --source-neuron-json ../vadclip_data/work_xd_residual/temporal_local_global768/neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/temporal_local_global768/concat_neurons \
  --neuron-width 768 \
  --clip-dim 512

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_local_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/temporal_local_global768/features/train \
  --out-csv ../vadclip_data/work_xd_residual/temporal_local_global768/xd_concat_train.csv \
  --keep-missing

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_local_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/temporal_local_global768/features/test \
  --out-csv ../vadclip_data/work_xd_residual/temporal_local_global768/xd_concat_test.csv \
  --target-feature-csv ../vad_data/work_xd/xd_test_local.csv

python experiments/vadclip_neuron_injection/train_single_vadclip_style_xd.py \
  --dataset xd \
  --train-list ../vadclip_data/work_xd_residual/temporal_local_global768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/temporal_local_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_local_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/temporal_local_global768/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/temporal_local_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/temporal_local_global768/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda

python experiments/vadclip_neuron_injection/test_single_vadclip_style_xd.py \
  --dataset xd \
  --test-list ../vadclip_data/work_xd_residual/temporal_local_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_local_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/temporal_local_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/temporal_local_global768/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## 产物

```text
../vadclip_data/work_xd_residual/temporal_local_global768/
  neurons/normal_mean.npy / normal_std.npy
  neurons/per_video_contributions/*.npz
  neurons/per_video_contributions.csv
  neurons/selected_neurons.json
  concat_neurons/selected_neurons.json
  features/train/ / features/test/
  xd_concat_train.csv / xd_concat_test.csv
  training/checkpoint_last.pth / model_best.pth / history.csv
  evaluation/metrics.json
```

`selected_neurons.json` 会记录 `frame_labels_used=false`、`test_data_used=false`、`same_video_background_used_for_localisation=true`，用于核验训练/测试隔离和 M3 选择逻辑。它也会明确记录：低分异常视频 snippet 被用作“同视频背景”，而不是伪标为纯正常样本。
