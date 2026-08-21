# XD split 的等更新次数诊断

之前的 XD split 训练集只有 478 条，而正式 global-768 训练集有 39,500 条。两者都训练 10 epoch 时，前者仅有 50 次 optimizer 更新，正式训练有 4,120 次；因此不能用前者判断方法或样本质量。

本诊断只对 split 的训练集做有放回随机采样，让每个 epoch 都抽取 39,500 条。它保留原始 global-768 的 10 epoch、batch size、学习率、scheduler、loss、冻结策略、validation AP2 选模规则；不读取训练帧级标注，也不使用 validation 或 held-out 视频训练。

从 `vadclip_code` 运行：

```bash
export OMP_NUM_THREADS=1

python experiments/vadclip_original_split_diagnostic/train_original_split.py \
  --dataset xd \
  --train-list ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/train.csv \
  --validation-list ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/validation.csv \
  --validation-gt-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/validation_gt.npy \
  --validation-segment-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/validation_segment.npy \
  --validation-label-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/annotations/validation_label.npy \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training_matched_steps/checkpoint_last.pth \
  --model-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training_matched_steps/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training_matched_steps \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --samples-per-epoch 39500 \
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
  --model-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training_matched_steps/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/heldout_evaluation_matched_steps \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

`39500 / 96` 会得到每 epoch 412 个 batch；10 epoch 共 4,120 次更新，正好与正式 XD global-768 对齐。首次运行不加 `--clean`；如中断，在训练命令末尾加 `--resume`。若需要从头重做，改用新输出目录，或明确加入 `--clean`。

产物写入：

```text
../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/
  training_matched_steps/model_best.pth
  training_matched_steps/checkpoint_last.pth
  training_matched_steps/history.csv
  heldout_evaluation_matched_steps/metrics.json
```

这仍是诊断，不是正式结果。每个 split 训练样本平均会重复抽取约 83 次；它用于控制优化次数不足，不能增加训练视频的多样性。
