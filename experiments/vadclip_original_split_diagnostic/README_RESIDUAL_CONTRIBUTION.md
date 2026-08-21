# global-768 残差贡献诊断

这个脚本不训练、不评测 AUC/AP/mAP，也不读取帧级标注。它只回答一个问题：训练后的残差支路究竟在多大程度上改变了原始 512D CLIP 特征，以及这种改变是否传递到了最终异常分数。

它复用普通测试的特征切分、padding、prompt 和分数定义。在同一输入上分别运行冻结的原 VadCLIP 路径与残差注入路径，输出两者的分数差。不会改动 `VadCLIP` baseline 或模型 checkpoint。

从 `vadclip_code` 运行：

```bash
export OMP_NUM_THREADS=1

python experiments/vadclip_original_split_diagnostic/analyze_residual_contribution.py \
  --dataset xd \
  --test-list ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/split/heldout.csv \
  --neuron-json ../vadclip_data/work_xd_residual/compare_bottom10/global_top768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/training/model_best.pth \
  --out-dir ../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/residual_contribution \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --ratio-epsilon 1e-8 \
  --device cuda
```

若中断，重复原命令会复用 `per_video/*.npz`；如需重新生成，加入 `--clean`。

产物位于：

```text
../vadclip_data/diagnostics/original_global768_no_trick/xd_seed234/residual_contribution/
  summary.json       # gate、全体 snippet 的范数比和分数变化分布
  per_video.csv      # 每条测试视频的均值、p50、p95，便于定位异常视频
  per_video/*.npz    # 可中断复用的逐 snippet 原始统计
```

重点看 `summary.json`：

- `gate_sigmoid`：残差总开关。接近 0 代表模型主动压低残差。
- `residual_to_clip_ratio`：`||gate × residual|| / ||原始 CLIP||`。若中位数和 p95 都很小，说明注入特征几乎没有改变 baseline 的输入。
- `absolute_score_change.prob2`：注入前后最终二分类异常分数的绝对差。若它也接近 0，则指标基本相同是预期结果。

这些是定位问题的只读证据，不是新的训练 trick，也不能单独说明神经元选择或正负样本一定有问题。
