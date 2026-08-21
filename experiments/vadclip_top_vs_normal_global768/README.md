# VadCLIP top-vs-pure-normal global-768

这是一条新的神经元选择消融。它保留异常训练视频中 baseline top10% snippet 作为伪正样本，但**不再把异常视频 bottom10% 当作负样本**。负参考来自纯正常训练视频；已有 normal mean/std 将 hidden 转成 z-score 后，纯正常参考均值就是 0。

因此只有神经元选择改变：

```text
旧：异常视频 top10% − 异常视频 bottom10%
新：异常视频 top10% − 纯正常视频参考分布
```

拼接、冻结残差训练、验证、最佳模型选择和测试全部复用已验证的 VadCLIP global-768 脚本。没有 M1 排序损失，不读取测试 GT，也不修改 `VadCLIP/`。

从 `vadclip_code` 运行。首次运行不加 `--clean`；选择阶段会复用 `normal_stats.npz` 和每个异常视频的 `per_video_deltas/*.npz`。若中断，直接重跑同一命令即可。需要只重算选择阶段时加 `--clean`；训练中断时按原训练脚本加 `--resume`；测试默认复用已完成的逐视频结果。

## XD-Violence 正式命令

```bash
python experiments/vadclip_neuron_injection/score_vadclip_pseudo.py \
  --dataset xd \
  --vadclip-root VadCLIP \
  --train-list ../vad_data/work_xd/xd_train_local.csv \
  --model-path ../vadclip_data/model/vadclip_xd.pth \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/pseudo_scores \
  --device cuda

python experiments/vadclip_top_vs_normal_global768/select_neurons_top_vs_normal.py \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_xd_residual/top_vs_normal_global768/pseudo_scores/group_scores.csv \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/neurons \
  --top-p 0.10 \
  --topk-global 768 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6

python experiments/vadclip_neuron_injection/make_concat_neuron_json.py \
  --source-neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons \
  --neuron-width 768 \
  --clip-dim 512

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/features/train \
  --out-csv ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_train.csv

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/features/test \
  --out-csv ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_test.csv \
  --target-feature-csv ../vad_data/work_xd/xd_test_local.csv

python experiments/vadclip_neuron_injection/train_single_vadclip_style_xd.py \
  --dataset xd \
  --train-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_train.csv \
  --test-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --checkpoint-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --num-workers 0 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda

python experiments/vadclip_neuron_injection/test_single_vadclip_style_xd.py \
  --dataset xd \
  --test-list ../vadclip_data/work_xd_residual/top_vs_normal_global768/xd_concat_test.csv \
  --neuron-json ../vadclip_data/work_xd_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_xd_residual/top_vs_normal_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_xd_residual/top_vs_normal_global768/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## UCF-Crime

```bash
python experiments/vadclip_neuron_injection/score_vadclip_pseudo.py \
  --dataset ucf \
  --vadclip-root VadCLIP \
  --train-list ../vad_data/work_ucf/ucf_train_local.csv \
  --model-path ../vadclip_data/model/vadclip_ucf.pth \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/pseudo_scores \
  --device cuda

python experiments/vadclip_top_vs_normal_global768/select_neurons_top_vs_normal.py \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_ucf_residual/top_vs_normal_global768/pseudo_scores/group_scores.csv \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/neurons \
  --top-p 0.10 \
  --topk-global 768 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6

python experiments/vadclip_neuron_injection/make_concat_neuron_json.py \
  --source-neuron-json ../vadclip_data/work_ucf_residual/top_vs_normal_global768/neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/concat_neurons \
  --neuron-width 768 \
  --clip-dim 512

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/features/train \
  --out-csv ../vadclip_data/work_ucf_residual/top_vs_normal_global768/ucf_concat_train.csv

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vad_data/work_ucf/ucf_test_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/features/test \
  --out-csv ../vadclip_data/work_ucf_residual/top_vs_normal_global768/ucf_concat_test.csv \
  --target-feature-csv ../vad_data/work_ucf/ucf_test_local.csv

python experiments/vadclip_neuron_injection/train_single_vadclip_style.py \
  --dataset ucf \
  --train-list ../vadclip_data/work_ucf_residual/top_vs_normal_global768/ucf_concat_train.csv \
  --test-list ../vadclip_data/work_ucf_residual/top_vs_normal_global768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --checkpoint-path ../vadclip_data/work_ucf_residual/top_vs_normal_global768/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_ucf_residual/top_vs_normal_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 2e-5 \
  --num-workers 0 \
  --eval-interval-samples 1280 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda

python experiments/vadclip_neuron_injection/test_single_vadclip_style.py \
  --dataset ucf \
  --test-list ../vadclip_data/work_ucf_residual/top_vs_normal_global768/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/top_vs_normal_global768/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_ucf_residual/top_vs_normal_global768/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/top_vs_normal_global768/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## 产物

```text
../vadclip_data/work_xd_residual/top_vs_normal_global768/
  pseudo_scores/group_scores.csv              # 冻结 baseline 的训练伪分数
  neurons/normal_mean.npy / normal_std.npy    # 纯正常训练视频参考分布
  neurons/per_video_deltas/*.npz              # 可恢复的每个异常视频 top-vs-normal delta
  neurons/selected_neurons.json               # 新的 global-768 选择结果
  concat_neurons/selected_neurons.json        # 768+512=1280D 输入契约
  features/train/ / features/test/
  xd_concat_train.csv / xd_concat_test.csv
  training/checkpoint_last.pth / model_best.pth / history.csv
  evaluation/metrics.json
```

`selected_neurons.json` 内的 `abnormal_bottom_snippets_used_as_negatives=false` 与 `frame_labels_used=false` 用于核验本方法确实没有把异常视频低分 snippet 或测试标注混入负参考。
