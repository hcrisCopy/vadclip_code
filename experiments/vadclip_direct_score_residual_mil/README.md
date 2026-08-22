# Direct score residual MIL (global-768)

该实验保留冻结的 VadCLIP、global-768 特征、官方 MIL loss、验证节奏、模型选择和测试指标。新模块不再把 768D 神经元先投影回 CLIP 特征，而是直接输出每个 snippet 的二分类和类别 logit 修正：

```text
frozen VadCLIP logits + bounded temporal adapter(neuron-768) logits
```

初始时两个修正 head 都为零，所以预测严格等于传入的 baseline。训练新增两项、且均不使用伪标签或帧级标签：

- 标注为正常的训练视频：所有有效 snippet 都是可靠 normal target；
- 仅对新增 logit 修正做“相邻神经元相似度”加权的时间一致约束。

异常训练视频仍只使用官方类别 top-k MIL，不把低分 snippet 伪造为负样本。`VadCLIP/` 不会被修改。

以下命令从 `vadclip_code` 运行。首次训练不加 `--clean` 或 `--resume`；训练中断后加 `--resume`；测试中断后直接重跑以复用 `per_video/*.npz`。要从头清理本实验的输出，可在相应命令末尾加 `--clean`。

## XD-Violence

```bash
python experiments/vadclip_direct_score_residual_mil/train_score_residual_mil.py \
  --dataset xd \
  --train-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/direct_score_residual_mil/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/direct_score_residual_mil/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/direct_score_residual_mil/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --eval-interval-samples 1280 \
  --seed 234 \
  --adapter-hidden-dim 256 \
  --adapter-kernel-size 5 \
  --delta-logit-cap 2.0 \
  --normal-snippet-weight 0.20 \
  --delta-temporal-weight 0.02 \
  --device cuda

python experiments/vadclip_direct_score_residual_mil/test_score_residual_mil.py \
  --dataset xd \
  --test-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/direct_score_residual_mil/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/direct_score_residual_mil/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --adapter-hidden-dim 256 \
  --adapter-kernel-size 5 \
  --delta-logit-cap 2.0 \
  --device cuda
```

XD 按官方规则：每个 epoch 验证一次，以语言分支 `AP2` 选择最佳模型。

## UCF-Crime

```bash
python experiments/vadclip_direct_score_residual_mil/train_score_residual_mil.py \
  --dataset ucf \
  --train-list ../vadclip_data/work_ucf_residual/top_vs_normal_global768/ucf_concat_train.csv \
  --test-list ../vadclip_data/work_ucf_residual/top_vs_normal_global768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --checkpoint-path ../vadclip_data/work_ucf_residual/top_vs_normal_global768/direct_score_residual_mil/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_ucf_residual/top_vs_normal_global768/direct_score_residual_mil/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/direct_score_residual_mil/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 2e-5 \
  --num-workers 0 \
  --eval-interval-samples 1280 \
  --seed 234 \
  --adapter-hidden-dim 256 \
  --adapter-kernel-size 5 \
  --delta-logit-cap 2.0 \
  --normal-snippet-weight 0.20 \
  --delta-temporal-weight 0.02 \
  --device cuda

python experiments/vadclip_direct_score_residual_mil/test_score_residual_mil.py \
  --dataset ucf \
  --test-list ../vadclip_data/work_ucf_residual/top_vs_normal_global768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_ucf_residual/top_vs_normal_global768/direct_score_residual_mil/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/direct_score_residual_mil/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --adapter-hidden-dim 256 \
  --adapter-kernel-size 5 \
  --delta-logit-cap 2.0 \
  --device cuda
```

UCF 保持官方验证频率，并以 `logits1 ROC-AUC` 选择最佳模型。

## 产物

```text
../vadclip_data/work_{xd,ucf}_residual/top_vs_normal_global768/direct_score_residual_mil/
  training/run_config.json
  training/history.csv
  training/checkpoint_last.pth
  training/model_best.pth
  evaluation/metrics.json
  evaluation/evaluation_config.json
  evaluation/per_video/*.npz
```

训练进度的 `delta1`、`delta2`、`prob2_shift` 用于确认新增分数支路确实在产生修正；它们不是模型选择指标。
