# VadCLIP 伪正负 snippet 质量诊断

这是一个只读诊断：在测试集上用**冻结的原始 VadCLIP baseline**重建每个 snippet 的两类异常分数，再用官方帧级 GT 衡量 top/bottom 10% 伪样本的质量。

GT 只用于离线报告，不参与神经元选择、训练、模型保存或正式测试。因此该目录的结果不能作为正式实验指标，也不能把测试 GT 移入训练流程。

诊断复现当前 global-768 的真实选择信号：`classifier prob1 = sigmoid(logits1)`。同时报告语言分支 `prob2 = 1-softmax(logits2)[normal]`，用于检查它与 XD 的 AP2/detection-mAP 分支是否存在候选片段错位。

从 `vadclip_code` 运行。首次运行不要加 `--clean`；单个视频的 frozen score 会保存在 `per_video_scores/`，中断后默认续跑。需要强制重新打分时加 `--no-resume`；需要清除本诊断旧产物时加 `--clean`。

## XD-Violence

```bash
python experiments/vadclip_pseudo_label_diagnostic/analyze_pseudo_label_quality.py \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --model-path ../vadclip_data/model/vadclip_xd.pth \
  --gt-path VadCLIP/list/gt.npy \
  --out-dir ../vadclip_data/diagnostics/pseudo_label_quality/xd_baseline_top10 \
  --vadclip-root VadCLIP \
  --top-p 0.10 \
  --device cuda
```

## UCF-Crime

```bash
python experiments/vadclip_pseudo_label_diagnostic/analyze_pseudo_label_quality.py \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --model-path ../vadclip_data/model/vadclip_ucf.pth \
  --gt-path VadCLIP/list/gt_ucf.npy \
  --out-dir ../vadclip_data/diagnostics/pseudo_label_quality/ucf_baseline_top10 \
  --vadclip-root VadCLIP \
  --top-p 0.10 \
  --device cuda
```

输出均在同级 `../vadclip_data/diagnostics/pseudo_label_quality/.../`：

```text
run_config.json                 # 明确声明：GT 只用于诊断，模型没有训练
per_video_scores/*.npz          # 可恢复的 frozen logits1 / logits2 分数
per_video_quality.csv           # 每个视频的 top/bottom 伪样本质量
summary.json                    # 异常视频上的 micro / macro 汇总与两分支重合度
```

重点查看 `summary.json`：

- `classifier_prob1.top.micro_positive_rate`：当前 global-768 伪正样本的帧级纯度，越高越好。
- `classifier_prob1.bottom.micro_positive_rate`：伪负样本中真实异常帧占比，越低越好；高说明真实异常被错放进 bottom 10%。
- `classifier_prob1.top.micro_positive_recall`：top 10% 覆盖的所有真实异常帧比例。
- `head_agreement`：`logits1` 选出的 top/bottom 与 `logits2` 选出的集合重合程度。若低，说明当前基于 `logits1` 的神经元选择可能与 XD 最终 AP2/mAP 分支存在信号错位。
