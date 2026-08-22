# global-768 有监督上限诊断

这是诊断，不是正式实验。它将原测试集按**完整视频**切成 60% 诊断训练、20% validation、20% held-out；训练只读取 train 的帧级标注，validation 只按 VadCLIP 原规则选最佳模型（XD 用 AP2，UCF 用 AUC1），held-out 从不参与训练和选模。

目的：测量冻结 VadCLIP 编码器下，global-768 特征是否有能力同时改善帧级 AP/AUC 和 detection mAP。若 held-out 仍无明显提升，直接解冻编码器更有依据；若提升明显，瓶颈更可能是原先的伪标签而不是特征容量。

模型只从 global-768 预测每个 snippet 的一个标量修正：它同时改变二分类异常分数和“normal 对全部异常类”的间隔，但保持各异常类别之间的相对排序不变。模型初始输出与给定 VadCLIP checkpoint 完全一致，官方 VadCLIP 及其评测代码均未修改。

以下命令从 `vadclip_code` 运行。首次运行使用 `--clean`；中断后训练命令去掉 `--clean` 并加 `--resume`，测试命令不加 `--clean` 即可复用已完成视频。

```bash
cd vadclip_code
export OMP_NUM_THREADS=1

python experiments/vadclip_supervised_upper_bound_diagnostic/prepare_supervised_split.py \
  --dataset xd \
  --full-test-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_test.csv \
  --full-gt-path VadCLIP/list/gt.npy \
  --full-segment-path VadCLIP/list/gt_segment.npy \
  --full-label-path VadCLIP/list/gt_label.npy \
  --out-dir ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --seed 234 \
  --clean

python experiments/vadclip_supervised_upper_bound_diagnostic/train_supervised_upper_bound.py \
  --dataset xd \
  --train-list ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/train.csv \
  --train-gt-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/annotations/train_gt.npy \
  --validation-list ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/validation.csv \
  --validation-gt-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/annotations/validation_gt.npy \
  --validation-segment-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/annotations/validation_segment.npy \
  --validation-label-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/annotations/validation_label.npy \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/training/checkpoint_last.pth \
  --model-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --seed 234 \
  --adapter-hidden-dim 256 \
  --adapter-kernel-size 5 \
  --delta-logit-cap 4.0 \
  --logits1-weight 1.0 \
  --logits2-weight 1.0 \
  --delta-smooth-weight 0.0 \
  --device cuda \
  --clean

python experiments/vadclip_supervised_upper_bound_diagnostic/test_supervised_upper_bound.py \
  --dataset xd \
  --test-list ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/heldout.csv \
  --gt-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/annotations/heldout_gt.npy \
  --segment-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/annotations/heldout_segment.npy \
  --label-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/split/annotations/heldout_label.npy \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/heldout_evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --adapter-hidden-dim 256 \
  --adapter-kernel-size 5 \
  --delta-logit-cap 4.0 \
  --device cuda \
  --clean
```

产物在 `../vadclip_data/diagnostics/supervised_upper_bound_global768/xd_seed234/`：`split/` 保存无视频泄漏的划分和每个划分对齐标注；`training/history.csv` 保存每轮 validation；`heldout_evaluation/metrics.json` 是唯一用于诊断结论的指标。子集 mAP 缺少部分类别时绝对值会变化，因此只能与同一 held-out split 内的冻结基线比较，不能和完整测试集的 mAP 直接横比。
