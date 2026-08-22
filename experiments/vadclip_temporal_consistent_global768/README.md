# VadCLIP temporal-consistent global-768

这个实验只改变 global-768 神经元选择，不改 `VadCLIP/`、1280D 拼接契约、冻结残差、原始 loss、学习率、验证频率、最佳模型规则或测试代码。

每个候选神经元必须同时满足：

```text
异常语义：异常训练视频 baseline Top 10% 片段相对纯正常参考有稳定偏移
时间定位：同一视频内，该神经元轨迹与冻结 baseline 分数有同方向稳定 Spearman 相关
稳定性：视频级 bootstrap 中稳定进入 global Top 768
```

异常视频 bottom 10% 不作为负样本；不读取测试视频、测试预测或帧级标注。纯正常训练视频只用于 normal z-score 参考。它与 M1 排序 loss 不同：M1 改训练目标，本实验只改输入残差支路的 768 个神经元。

所有命令从 `vadclip_code` 运行。首次运行不要加 `--clean`；选择器会复用 `normal_stats.npz` 与 `per_video_contributions/*.npz`，中断后直接重复同一命令即可。需要从头重做当前选择目录时才加 `--clean`。`--no-resume` 会重算已有产物而不删除目录。

## XD：先做带标注的保留集诊断

这一步仅判断新的神经元集合是否值得进入正式实验。选择器和训练都不读取帧级标注；标注仅由既有的 video-disjoint diagnostic 脚本生成 validation/held-out 对齐评测文件。validation 按 VadCLIP 官方 XD 规则的 AP2 选模型，held-out 最后才评测一次。

```bash
cd vadclip_code
export OMP_NUM_THREADS=1

python experiments/vadclip_neuron_injection/score_vadclip_pseudo.py \
  --dataset xd \
  --vadclip-root VadCLIP \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --model-path ../vadclip_data/model/vadclip_xd.pth \
  --out-dir ../vadclip_data/work_xd_residual/temporal_consistent_global768/pseudo_scores \
  --device cuda

python experiments/vadclip_temporal_consistent_global768/select_neurons_temporal_consistent.py \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_xd_residual/temporal_consistent_global768/pseudo_scores/group_scores.csv \
  --out-dir ../vadclip_data/work_xd_residual/temporal_consistent_global768/neurons \
  --top-p 0.10 \
  --topk-global 768 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6 \
  --bootstrap-rounds 20 \
  --bootstrap-seed 234

python experiments/vadclip_neuron_injection/make_concat_neuron_json.py \
  --source-neuron-json ../vadclip_data/work_xd_residual/temporal_consistent_global768/neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/temporal_consistent_global768/concat_neurons \
  --neuron-width 768 \
  --clip-dim 512

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/temporal_consistent_global768/features/train \
  --out-csv ../vadclip_data/work_xd_residual/temporal_consistent_global768/xd_concat_train.csv \
  --keep-missing

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/temporal_consistent_global768/features/test \
  --out-csv ../vadclip_data/work_xd_residual/temporal_consistent_global768/xd_concat_test.csv \
  --target-feature-csv ../vad_data/work_xd/xd_test_local.csv

python experiments/vadclip_original_split_diagnostic/split_video_level_test.py \
  --dataset xd \
  --full-test-list ../vadclip_data/work_xd_residual/temporal_consistent_global768/xd_concat_test.csv \
  --full-gt-path VadCLIP/list/gt.npy \
  --full-segment-path VadCLIP/list/gt_segment.npy \
  --full-label-path VadCLIP/list/gt_label.npy \
  --out-dir ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --seed 234

python experiments/vadclip_original_split_diagnostic/train_original_split.py \
  --dataset xd \
  --train-list ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/train.csv \
  --validation-list ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/validation.csv \
  --validation-gt-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/annotations/validation_gt.npy \
  --validation-segment-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/annotations/validation_segment.npy \
  --validation-label-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/annotations/validation_label.npy \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/training/checkpoint_last.pth \
  --model-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --eval-interval-samples 1280 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda

python experiments/vadclip_original_split_diagnostic/test_original_split.py \
  --dataset xd \
  --test-list ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/heldout.csv \
  --gt-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/annotations/heldout_gt.npy \
  --segment-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/annotations/heldout_segment.npy \
  --label-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/split/annotations/heldout_label.npy \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/temporal_consistent_global768/xd_seed234/heldout_evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

重点比较 `heldout_evaluation/metrics.json` 与已有 original global-768 的**同一个诊断切分**。held-out 从未参与训练或选模；其 mAP 只能用于同一切分的相对比较。

## XD：诊断通过后进行正式实验

```bash
python experiments/vadclip_neuron_injection/train_single_vadclip_style_xd.py \
  --dataset xd \
  --train-list ../vadclip_data/work_xd_residual/temporal_consistent_global768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/temporal_consistent_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/temporal_consistent_global768/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/temporal_consistent_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/temporal_consistent_global768/training \
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
  --test-list ../vadclip_data/work_xd_residual/temporal_consistent_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/temporal_consistent_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/temporal_consistent_global768/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

正式训练和测试仍完全按当前 VadCLIP baseline 对齐协议执行：训练中每个 epoch 在 `--test-list` 上验证，按官方 XD 语言分支 AP2 保存 `model_best.pth`。

## UCF-Crime：正式实验

```bash
python experiments/vadclip_neuron_injection/score_vadclip_pseudo.py \
  --dataset ucf \
  --vadclip-root VadCLIP \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --model-path ../vadclip_data/model/vadclip_ucf.pth \
  --out-dir ../vadclip_data/work_ucf_residual/temporal_consistent_global768/pseudo_scores \
  --device cuda

python experiments/vadclip_temporal_consistent_global768/select_neurons_temporal_consistent.py \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_ucf_residual/temporal_consistent_global768/pseudo_scores/group_scores.csv \
  --out-dir ../vadclip_data/work_ucf_residual/temporal_consistent_global768/neurons \
  --top-p 0.10 \
  --topk-global 768 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6 \
  --bootstrap-rounds 20 \
  --bootstrap-seed 234

python experiments/vadclip_neuron_injection/make_concat_neuron_json.py \
  --source-neuron-json ../vadclip_data/work_ucf_residual/temporal_consistent_global768/neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/temporal_consistent_global768/concat_neurons \
  --neuron-width 768 \
  --clip-dim 512

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/temporal_consistent_global768/features/train \
  --out-csv ../vadclip_data/work_ucf_residual/temporal_consistent_global768/ucf_concat_train.csv

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_ucf/ucf_test_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/temporal_consistent_global768/features/test \
  --out-csv ../vadclip_data/work_ucf_residual/temporal_consistent_global768/ucf_concat_test.csv \
  --target-feature-csv ../vad_data/work_ucf/ucf_test_local.csv

python experiments/vadclip_neuron_injection/train_single_vadclip_style.py \
  --dataset ucf \
  --train-list ../vadclip_data/work_ucf_residual/temporal_consistent_global768/ucf_concat_train.csv \
  --test-list ../vadclip_data/work_ucf_residual/temporal_consistent_global768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --checkpoint-path ../vadclip_data/work_ucf_residual/temporal_consistent_global768/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_ucf_residual/temporal_consistent_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/temporal_consistent_global768/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 2e-5 \
  --num-workers 0 \
  --eval-interval-samples 1280 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda

python experiments/vadclip_neuron_injection/test_single_vadclip_style.py \
  --dataset ucf \
  --test-list ../vadclip_data/work_ucf_residual/temporal_consistent_global768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/temporal_consistent_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_ucf_residual/temporal_consistent_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/temporal_consistent_global768/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## 产物

```text
../vadclip_data/work_{xd,ucf}_residual/temporal_consistent_global768/
  pseudo_scores/group_scores.csv
  neurons/normal_mean.npy / normal_std.npy
  neurons/per_video_contributions/*.npz
  neurons/semantic_* / temporal_* / bootstrap_frequency.npy
  neurons/selection_scores.npy
  neurons/selected_neurons.json
  concat_neurons/selected_neurons.json
  features/train/ / features/test/
  {xd,ucf}_concat_train.csv / {xd,ucf}_concat_test.csv
  training/checkpoint_last.pth / model_best.pth / history.csv
  evaluation/metrics.json
```

`selected_neurons.json` 会记录 `frame_labels_used=false`、`test_data_used=false`、`abnormal_bottom_snippets_used_as_negatives=false`、每个选中神经元的语义效应、时间一致性和 bootstrap 频率，便于复核方法边界。
