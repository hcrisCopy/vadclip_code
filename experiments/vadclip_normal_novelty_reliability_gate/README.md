# 正常参考新颖度门控：先做诊断

上一版 q 使用“与本视频高分种子相似”。正常视频的高分种子仍然是正常内容，因而该项不能抑制正常高分尾部。

本版本改为：

```text
q = baseline 高分置信度 × 偏离纯正常 hidden 分布的置信度
```

normal hidden 的均值、方差、RMS 新颖度阈值都只由训练正常视频估计。帧级 GT 仅在最后评价 q 的质量，绝不进入校准、训练或选模。

从 `vadclip_code` 运行：

```bash
python experiments/vadclip_normal_novelty_reliability_gate/diagnose_normal_novelty.py \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_xd_residual/compare_bottom10/pseudo_scores/group_scores.csv \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --model-path ../vadclip_data/model/vadclip_xd.pth \
  --gt-path VadCLIP/list/gt.npy \
  --out-dir ../vadclip_data/diagnostics/normal_novelty_reliability/xd \
  --vadclip-root VadCLIP \
  --seed-top-p 0.10 \
  --expand-top-p 0.30 \
  --normal-score-quantile 0.95 \
  --score-temperature 0.05 \
  --normal-novelty-quantile 0.95 \
  --novelty-temperature-scale 1.0 \
  --normal-score-snippets-per-video 256 \
  --normal-hidden-snippets-per-video 256 \
  --sigma-min 1e-6 \
  --device cuda
```

结果在 `../vadclip_data/diagnostics/normal_novelty_reliability/xd/summary.json`。

只有同时满足以下条件，才值得把此 q 接入新的 global-768 选择、训练验证和最终测试：扩展 precision 接近原始 77.3%，recall 显著超过 16.0%，且正常视频 q 的 p95 最大值显著低于上一版的 0.871。
