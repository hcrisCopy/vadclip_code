# 原始 global-768 的视频级切分诊断

这是诊断实验，不是正式结果：把原测试集按**视频**分成 train / validation / held-out 三部分。训练仍是最初的 global-768 残差注入：冻结 VadCLIP、原始三项 loss、原始 1280D 特征；不启用 M1、伪分数排序、帧级监督或任何其他 trick。

帧级标注只用于生成 validation 和 held-out 的对齐评测文件。它们不会被训练脚本读取。validation 按 baseline 原规则选模型：UCF 为 AUC1，XD 为 AP2；held-out 仅在最后测试一次。所有命令从 `vadclip_code` 运行，首次不加 `--clean`；中断训练后添加 `--resume`。

```bash
cd vadclip_code
```

## UCF

```bash
python experiments/vadclip_original_split_diagnostic/split_video_level_test.py \
  --dataset ucf \
  --full-test-list ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/ucf_concat_test.csv \
  --full-gt-path VadCLIP/list/gt_ucf.npy \
  --full-segment-path VadCLIP/list/gt_segment_ucf.npy \
  --full-label-path VadCLIP/list/gt_label_ucf.npy \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --seed 234

python experiments/vadclip_original_split_diagnostic/train_original_split.py \
  --dataset ucf \
  --train-list ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/train.csv \
  --validation-list ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/validation.csv \
  --validation-gt-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/annotations/validation_gt.npy \
  --validation-segment-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/annotations/validation_segment.npy \
  --validation-label-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/annotations/validation_label.npy \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --checkpoint-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/training/checkpoint_last.pth \
  --model-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/training \
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

python experiments/vadclip_original_split_diagnostic/test_original_split.py \
  --dataset ucf \
  --test-list ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/heldout.csv \
  --gt-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/annotations/heldout_gt.npy \
  --segment-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/annotations/heldout_segment.npy \
  --label-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/split/annotations/heldout_label.npy \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/ucf_seed234/heldout_evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

UCF 的小训练子集如果没有达到一次原始 1280-sample 验证间隔，会在 epoch 末尾补一次 validation；这是为了产生可恢复 checkpoint，held-out 仍不参与训练或选模。

## XD-Violence

```bash
python experiments/vadclip_original_split_diagnostic/split_video_level_test.py \
  --dataset xd \
  --full-test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --full-gt-path VadCLIP/list/gt.npy \
  --full-segment-path VadCLIP/list/gt_segment.npy \
  --full-label-path VadCLIP/list/gt_label.npy \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --seed 234

python experiments/vadclip_original_split_diagnostic/train_original_split.py \
  --dataset xd \
  --train-list ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/train.csv \
  --validation-list ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/validation.csv \
  --validation-gt-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/validation_gt.npy \
  --validation-segment-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/validation_segment.npy \
  --validation-label-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/validation_label.npy \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training/checkpoint_last.pth \
  --model-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training \
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
  --test-list ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/heldout.csv \
  --gt-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/heldout_gt.npy \
  --segment-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/heldout_segment.npy \
  --label-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/heldout_label.npy \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/heldout_evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## 产物与解读

```text
../vadclip_data/diagnostics/original_global768_no_trick/{ucf,xd}_seed234/
  split/train.csv / validation.csv / heldout.csv
  split/annotations/{validation,heldout}_{gt,segment,label}.npy
  split/split_manifest.json                 # 无视频泄漏、标签覆盖、帧数对齐记录
  training/checkpoint_last.pth / model_best.pth / history.csv
  heldout_evaluation/metrics.json           # 唯一用于诊断结论的保留集指标
```

`detection mAP` 在子集上只用于同一 split 内的模型比较；因官方实现始终平均全部类别，子集缺少某类别会降低绝对 mAP，不能与完整官方测试集的 mAP 数值直接横比。
