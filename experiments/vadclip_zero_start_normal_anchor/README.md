# M2：零起点残差尺度 + 正常视频锚定（XD）

此实验复用已完成的 top-vs-normal global-768 特征。VadCLIP 权重始终冻结；训练、验证、AP2 选模和 XD 测试均保持原协议。

与原残差分支不同：MLP 最后一层不置零，注入尺度 `alpha` 初始化为 0。故初始预测严格等于 baseline，但 `alpha` 首步即可学习。仅对训练集正常视频的有效 snippet 加入残差特征锚定，抑制正常帧被残差错误抬高。

不使用 M1 排序损失、测试集标签或帧级标注。

## 训练

```bash
python experiments/vadclip_zero_start_normal_anchor/train_zero_start_normal_anchor_xd.py \
  --dataset xd \
  --train-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/zero_start_normal_anchor/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/zero_start_normal_anchor/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/zero_start_normal_anchor/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --residual-scale-init 0.0 \
  --normal-anchor-weight 0.10 \
  --device cuda
```

训练会按 XD 官方语言分支 `AP2` 保存最优模型。中断后在相同指令末尾添加 `--resume`；需要重新开始时添加 `--clean`，不要同时使用二者。

## 测试

```bash
python experiments/vadclip_zero_start_normal_anchor/test_zero_start_normal_anchor_xd.py \
  --dataset xd \
  --test-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/zero_start_normal_anchor/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/zero_start_normal_anchor/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## 产物

```text
../vadclip_data/work_xd_residual/top_vs_normal_global768/zero_start_normal_anchor/
  training/run_config.json       # 固定配置与无测试标签声明
  training/history.csv           # 每轮 loss、anchor、alpha 与指标
  training/checkpoint_last.pth   # 可恢复训练状态
  training/model_best.pth        # 按 AP2 选择的模型
  evaluation/metrics.json        # AUC1/AP1、AUC2/AP2、detection mAP
  evaluation/per_video/*.npz     # 可恢复的逐视频预测
```
