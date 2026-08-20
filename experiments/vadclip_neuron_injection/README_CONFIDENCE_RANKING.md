# M1：置信加权时序排序损失

本实验复用已有的 `global_top768` 1280D 特征和**同一训练 CSV**产生的冻结 VadCLIP 伪分数，不覆盖原始 global-768 训练结果。baseline 的模型、原始三项 loss、学习率、验证频率和最佳模型选择规则都不变；只额外训练残差分支上的 M1 loss。

对异常训练视频，伪分数 top-10% 为 P、bottom-10% 为 N；对同 batch 正常视频，当前模型最高的 10% 为 H。M1 在 `logits1` 和语言分支的 `1-P(normal)` 上同时加入：

```text
softplus(m_intra - score(P) + score(N))
softplus(m_cross - score(P) + score(H))
```

每个异常视频的权重是其伪分数 P--N 均值差除以该视频的伪分数范围。伪分数仅决定相对排序和置信度，不作为绝对片段标签；不读取测试集标签或测试预测。

以下命令都从 `vadclip_code` 运行。`--ranking-pseudo-csv` 必须来自构建 `*_concat_train.csv` 时所用的同一原始训练 CSV。首次正式运行不加 `--clean`；中断后在训练命令末尾加 `--resume`。

```bash
cd vadclip_code
```

## UCF

```bash
python experiments/vadclip_neuron_injection/train_single_vadclip_style.py \
  --dataset ucf \
  --train-list ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/ucf_concat_train.csv \
  --test-list ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --ranking-pseudo-csv ../vadclip_data/work_ucf_residual/compare_bottom10/pseudo_scores/group_scores.csv \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --checkpoint-path ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768_rank_m1/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768_rank_m1/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768_rank_m1/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 2e-5 \
  --num-workers 0 \
  --eval-interval-samples 1280 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --rank-loss-weight 0.10 \
  --rank-top-p 0.10 \
  --rank-intra-margin 0.10 \
  --rank-cross-margin 0.10 \
  --device cuda

python experiments/vadclip_neuron_injection/test_single_vadclip_style.py \
  --dataset ucf \
  --test-list ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768_rank_m1/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768_rank_m1/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

UCF 仍按官方 cadence 验证，并以 `logits1 ROC-AUC` 保存 `model_best.pth`。

## XD-Violence

```bash
python experiments/vadclip_neuron_injection/train_single_vadclip_style_xd.py \
  --dataset xd \
  --train-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --ranking-pseudo-csv ../vadclip_data/work_xd_residual/compare_bottom10/pseudo_scores/group_scores.csv \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768_rank_m1/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768_rank_m1/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768_rank_m1/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --rank-loss-weight 0.10 \
  --rank-top-p 0.10 \
  --rank-intra-margin 0.10 \
  --rank-cross-margin 0.10 \
  --device cuda

python experiments/vadclip_neuron_injection/test_single_vadclip_style_xd.py \
  --dataset xd \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768_rank_m1/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768_rank_m1/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

XD 仍每个 epoch 验证一次，并以官方语言分支 `AP2` 保存 `model_best.pth`。

## 新产物

```text
../vadclip_data/work_{ucf,xd}_residual/compare_bottom10/global_top768_rank_m1/
  training/run_config.json       # 完整 M1 参数与伪分数来源
  training/history.csv           # 原始 loss + rank_intra/rank_cross/rank
  training/checkpoint_last.pth   # 可恢复训练状态
  training/model_best.pth        # 仍按 baseline 原规则选取
  evaluation/metrics.json        # AUC、AP、detection mAP
```
