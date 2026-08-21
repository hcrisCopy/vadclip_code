# global-768 残差强度校准

这是正式 global-768 模型的推理期校准，不训练、不改 checkpoint、不改 `VadCLIP` baseline。它对同一个模型测试：

```text
原始 512D CLIP + alpha × 已训练残差
```

`alpha=0` 是冻结 baseline 路径，`alpha=1` 是当前正式 global-768。每个 alpha 都使用原测试流程、帧级指标和 detection mAP；选取规则与 baseline 一致：UCF 用 AUC1，XD 用 AP2。

所有输入都应使用 baseline 当前用于模型选择的同一套 CSV 和标注。这个实验不使用 held-out 诊断数据，不读取训练标签，不改变正式训练逻辑。

从 `vadclip_code` 运行：

## XD-Violence

```bash
export OMP_NUM_THREADS=1

python experiments/vadclip_residual_alpha_calibration/sweep_residual_alpha.py \
  --dataset xd \
  --selection-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --selection-gt-path VadCLIP/list/gt.npy \
  --selection-segment-path VadCLIP/list/gt_segment.npy \
  --selection-label-path VadCLIP/list/gt_label.npy \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/alpha_calibration \
  --alphas 0 0.25 0.5 0.75 1.0 1.25 \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## UCF-Crime

```bash
export OMP_NUM_THREADS=1

python experiments/vadclip_residual_alpha_calibration/sweep_residual_alpha.py \
  --dataset ucf \
  --selection-list ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/ucf_concat_test.csv \
  --selection-gt-path VadCLIP/list/gt_ucf.npy \
  --selection-segment-path VadCLIP/list/gt_segment_ucf.npy \
  --selection-label-path VadCLIP/list/gt_label_ucf.npy \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/alpha_calibration \
  --alphas 0 0.25 0.5 0.75 1.0 1.25 \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

如中断，重复同一命令会复用 `alpha_*/per_video/*.npz`；如需重做，显式加入 `--clean`。产物位于：

```text
../vadclip_data/work_{xd,ucf}_residual/compare_bottom10/global_top768/alpha_calibration/
  alpha_*/metrics.json            # 每个 alpha 的完整 AP/AUC/mAP
  alpha_*/per_video/*.npz         # 可中断复用的逐视频预测
  alpha_metrics.csv               # 所有 alpha 的指标对比表
  selected_alpha.json             # 按 baseline 规则选出的 alpha
  run_config.json
```

结果解释：若 alpha 小于 1 更好，正式残差偏强；若 alpha 大于 1 更好，正式残差偏弱；若 alpha 等于 0 最好，当前残差没有净收益；若 alpha 等于 1 最好，应优先检查神经元选择或训练监督而不是残差大小。
