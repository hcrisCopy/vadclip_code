# M2：逐 snippet 自适应残差门控

这是基于 global-768 正式实验分析的新模块。原方法为所有 snippet 使用同一个残差强度；M2 让每个 snippet 根据自己的 768D 神经元表示产生一个 gate，再注入残差：

```text
enhanced CLIP = original CLIP + snippet_gate × neuron_residual
```

VadCLIP baseline 始终冻结，原 MIL loss、batch size、学习率、scheduler、模型选择和指标不变。训练开始时残差为零，因此初始前向等于 baseline。M2 不使用帧级标注、测试监督、伪分数排序或 M1。

新增正则仅作用于真实 snippet，padding 的 0 特征由 `length` 排除：

- 幅度约束：当 `||delta|| / ||CLIP||` 超过 `residual-ratio-cap` 才惩罚，避免少量视频训练时出现过强修正。
- 时间平滑：惩罚相邻真实 snippet 残差的剧烈变化，目标是减少局部假阳性并改善时间段边界。

推荐的首个正式配置使用 `ratio-cap=0.01`、`amp-weight=0.10`、`tv-weight=0.01`。这些是可控的首个 M2 配置，不保证提升；模型仍按 baseline 规则选取：XD 为 AP2，UCF 为 AUC1。

从 `vadclip_code` 运行。

## XD-Violence

```bash
export OMP_NUM_THREADS=1

python experiments/vadclip_adaptive_residual_gate/train_adaptive_vadclip.py \
  --dataset xd \
  --train-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/adaptive_gate/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/adaptive_gate/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/adaptive_gate \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --eval-interval-samples 1280 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --residual-ratio-cap 0.01 \
  --residual-amp-weight 0.10 \
  --residual-tv-weight 0.01 \
  --device cuda

python experiments/vadclip_adaptive_residual_gate/test_adaptive_vadclip.py \
  --dataset xd \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/adaptive_gate/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/adaptive_gate/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## UCF-Crime

```bash
export OMP_NUM_THREADS=1

python experiments/vadclip_adaptive_residual_gate/train_adaptive_vadclip.py \
  --dataset ucf \
  --train-list ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/ucf_concat_train.csv \
  --test-list ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --checkpoint-path ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/adaptive_gate/checkpoint_last.pth \
  --model-path ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/adaptive_gate/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/adaptive_gate \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 2e-5 \
  --num-workers 0 \
  --eval-interval-samples 1280 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --residual-ratio-cap 0.01 \
  --residual-amp-weight 0.10 \
  --residual-tv-weight 0.01 \
  --device cuda

python experiments/vadclip_adaptive_residual_gate/test_adaptive_vadclip.py \
  --dataset ucf \
  --test-list ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/adaptive_gate/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/global_top768/adaptive_gate/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

首次运行不加 `--clean` 或 `--resume`。训练中断后在同一训练命令后加 `--resume`；评测中断后重复同一测试命令即可复用逐视频预测。产物位于：

```text
../vadclip_data/work_{xd,ucf}_residual/compare_bottom10/global_top768/adaptive_gate/
  checkpoint_last.pth / model_best.pth / history.csv / run_config.json
  evaluation/metrics.json
  evaluation/per_video/*.npz
```
