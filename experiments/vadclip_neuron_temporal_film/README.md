# M1 / M2：global-768 神经元条件特征增强（XD）

本目录不修改 `VadCLIP/`。它直接复用已经完成的 global-768 产物：每个输入片段仍是 `[768D selected-neuron | 512D original-CLIP]`。因此不需要重新生成伪分数、重新选神经元或重新构建特征。

- **M1**：每个 snippet 的 768D 神经元生成 512 个逐通道调制量（FiLM），再基于“调制后的原始 512D 特征”生成残差。
- **M2**：先对神经元条件做 dilation=1、2 的轻量深度时序卷积，再走与 M1 相同的 FiLM 残差。它只增加很小的缓存特征计算，不读取视频、不在线运行 CLIP。

两者均满足：VadCLIP 全部冻结；训练使用官方 XD 三项弱监督损失；每 epoch 在完整测试集验证；按官方语言分支 `AP2` 保存 `model_best.pth`；不使用测试标注参与反向传播。

所有命令从 `vadclip_code` 目录运行。首次正式运行不要加 `--clean`。训练中断后，在同一训练命令末尾加 `--resume`；测试会自动复用已完成的逐视频预测，若要强制重算则添加 `--clean`。

```bash
cd vadclip_code
```

## M1：神经元条件 FiLM

先做一次零残差核验。它只读一个测试视频，确认未经训练的 M1 与冻结 VadCLIP 输出完全一致。

```bash
python experiments/vadclip_neuron_temporal_film/verify_zero_identity_xd.py \
  --dataset xd \
  --module m1 \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --vadclip-root VadCLIP \
  --condition-hidden-dim 256 \
  --residual-hidden-dim 512 \
  --temporal-dilations 1 2 \
  --device cuda
```

```bash
python experiments/vadclip_neuron_temporal_film/train_m1_m2_xd.py \
  --dataset xd \
  --module m1 \
  --train-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m1/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m1/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m1/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --seed 234 \
  --condition-hidden-dim 256 \
  --residual-hidden-dim 512 \
  --temporal-dilations 1 2 \
  --device cuda
```

```bash
python experiments/vadclip_neuron_temporal_film/test_m1_m2_xd.py \
  --dataset xd \
  --module m1 \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m1/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m1/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --condition-hidden-dim 256 \
  --residual-hidden-dim 512 \
  --temporal-dilations 1 2 \
  --device cuda
```

## M2：M1 + 多尺度神经元时序上下文

M2 与 M1 的特征、训练协议和超参数完全一致，只有 `--module m2` 额外启用神经元条件上的双尺度深度时序卷积。因此 M1/M2 结果可以直接比较。

```bash
python experiments/vadclip_neuron_temporal_film/verify_zero_identity_xd.py \
  --dataset xd \
  --module m2 \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --vadclip-root VadCLIP \
  --condition-hidden-dim 256 \
  --residual-hidden-dim 512 \
  --temporal-dilations 1 2 \
  --device cuda
```

```bash
python experiments/vadclip_neuron_temporal_film/train_m1_m2_xd.py \
  --dataset xd \
  --module m2 \
  --train-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m2/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m2/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m2/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --seed 234 \
  --condition-hidden-dim 256 \
  --residual-hidden-dim 512 \
  --temporal-dilations 1 2 \
  --device cuda
```

```bash
python experiments/vadclip_neuron_temporal_film/test_m1_m2_xd.py \
  --dataset xd \
  --module m2 \
  --test-list ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m2/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/m2/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --condition-hidden-dim 256 \
  --residual-hidden-dim 512 \
  --temporal-dilations 1 2 \
  --device cuda
```

每个实验目录均会产生：

```text
../vadclip_data/work_xd_residual/compare_bottom10/global_top768/temporal_film/{m1,m2}/
  training/checkpoint_last.pth  # 可恢复训练状态
  training/model_best.pth       # 官方 AP2 最优模型
  training/history.csv          # 每 epoch 的 loss、残差/原特征比、AUC、AP、mAP
  training/run_config.json      # 完整参数和协议
  evaluation/per_video/*.npz    # 可中断复用的逐视频预测
  evaluation/metrics.json       # AUC1/AP1、AUC2/AP2、detection mAP
```

进度条的 `ratio` 是残差 L2 范数 / 原始 512D 特征 L2 范数；第一个 batch 接近 0 是预期的，因为模块从严格 baseline 身份映射开始。它在后续 batch 是否稳定离开 0，比单看 loss 更能说明增强模块是否真的参与了训练。
