# VadCLIP 神经元残差注入（UCF）

这个目录把给定的 DSANet 神经元实验迁到 VadCLIP，不修改 `VadCLIP/` 内任何 baseline 文件。所有新代码只读写当前仓库及其同级目录 `../vadclip_data/`；不会在代码或运行 CSV 中引用另一个项目。

方法保持如下边界：VadCLIP 的冻结 `logits1` 给异常训练视频生成伪分数；每个视频内部的 top/bottom `p` 片段形成配对 ShiftScore；每层各取 64 个 CLIP ViT-B/16 CLS 神经元，得到 768D z-score 特征；其与同时间步的 512D 最终 CLIP 特征拼成 1280D。训练时只学习一个带小门控的 `768 -> 1024 -> 1024 -> 512` 残差分支，原始 512D VadCLIP 始终冻结。

## 数据独立与可复用输入

先把已完成的 CLIP hidden 文件和相应 manifest **复制或同步**到下列目标目录；复制后不保留任何外部项目路径。

```text
../vadclip_data/
  UCFClipFeatures/                                  # 512D .npy，VadCLIP/DSANet 共用的 CLIP 输出
  staging/
    ucf_train_hidden_manifest.csv                   # 仅作为一次性导入清单
    ucf_test_hidden_manifest.csv
  work_ucf/
    clip_hidden_stride16_train_8gpu/features/*.npz  # 已完成的 [T,12,768] CLS hidden
    clip_hidden_stride16_test_8gpu/features/*.npz
```

hidden 的 stride、token pool、层数必须保持原实验的 `stride=16`、`token_pool=cls`、12 层。`stage_reused_hidden_manifest.py` 会忽略导入 manifest 的旧 `hidden_path`，改写成上述目录中的路径，从而使新仓库独立。旧 DSANet 的伪分数、神经元选择结果、768D/1280D 派生特征、checkpoint 和模型不能复用；共享的原始 512D CLIP 特征与 hidden 可以复用。新的伪分数必须由 VadCLIP checkpoint 产生。

以下命令全部从 `vadclip_code` 运行。首次正式运行不加 `--clean`；数据构建、伪分数和测试默认会复用已完成的单项输出。需要故意从头做某阶段时，在该阶段添加 `--clean`。

```bash
cd vadclip_code
```

先为 VadCLIP 原始 list 创建只指向 `../vadclip_data` 的 CSV，并把复制后的 manifest 本地化：

```bash
python experiments/vadclip_neuron_injection/prepare_local_feature_csv.py \
  --input-csv VadCLIP/list/ucf_CLIP_rgb.csv \
  --feature-root ../vadclip_data/UCFClipFeatures \
  --output-csv ../vadclip_data/work_ucf/ucf_train_local.csv \
  --strict

python experiments/vadclip_neuron_injection/prepare_local_feature_csv.py \
  --input-csv VadCLIP/list/ucf_CLIP_rgbtest.csv \
  --feature-root ../vadclip_data/UCFClipFeatures \
  --output-csv ../vadclip_data/work_ucf/ucf_test_local.csv \
  --strict

python experiments/vadclip_neuron_injection/stage_reused_hidden_manifest.py \
  --input-manifest ../vadclip_data/staging/ucf_train_hidden_manifest.csv \
  --hidden-root ../vadclip_data/work_ucf/clip_hidden_stride16_train_8gpu \
  --out-manifest ../vadclip_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv

python experiments/vadclip_neuron_injection/stage_reused_hidden_manifest.py \
  --input-manifest ../vadclip_data/staging/ucf_test_hidden_manifest.csv \
  --hidden-root ../vadclip_data/work_ucf/clip_hidden_stride16_test_8gpu \
  --out-manifest ../vadclip_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv
```

## 正式流程

`../vadclip_data/model/vadclip_ucf.pth` 是 VadCLIP 的 512D UCF baseline 权重，不是 DSANet 权重。

```bash
python experiments/vadclip_neuron_injection/score_vadclip_pseudo.py \
  --dataset ucf \
  --vadclip-root VadCLIP \
  --train-list ../vadclip_data/work_ucf/ucf_train_local.csv \
  --model-path ../vadclip_data/model/vadclip_ucf.pth \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/pseudo_scores \
  --device cuda

python experiments/vadclip_neuron_injection/select_neurons_intravideo_paired.py \
  --dataset ucf \
  --source-train-csv ../vadclip_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vadclip_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclip_data/work_ucf_residual/compare_bottom10/pseudo_scores/group_scores.csv \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/neurons \
  --top-p 0.10 \
  --topk-per-layer 64 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6

python experiments/vadclip_neuron_injection/make_concat_neuron_json.py \
  --source-neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/concat_neurons \
  --neuron-width 768 \
  --clip-dim 512

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vadclip_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vadclip_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/features/train \
  --out-csv ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/ucf_concat_train.csv

python experiments/vadclip_neuron_injection/build_concat_features.py \
  --source-csv ../vadclip_data/work_ucf/ucf_test_local.csv \
  --hidden-manifest ../vadclip_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/concat_neurons/selected_neurons.json \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/features/test \
  --out-csv ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/ucf_concat_test.csv

python experiments/vadclip_neuron_injection/train_single_vadclip_style.py \
  --dataset ucf \
  --train-list ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/ucf_concat_train.csv \
  --test-list ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/concat_neurons/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --checkpoint-path ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/training/checkpoint_last.pth \
  --model-path ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/training \
  --vadclip-root VadCLIP \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 7e-5 \
  --num-workers 4 \
  --seed 234 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda

python experiments/vadclip_neuron_injection/test_single_vadclip_style.py \
  --dataset ucf \
  --test-list ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/ucf_concat_test.csv \
  --neuron-json ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/concat_neurons/selected_neurons.json \
  --model-path ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/training/model_best.pth \
  --out-dir ../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/evaluation \
  --vadclip-root VadCLIP \
  --num-workers 4 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

## 产物与中断恢复

```text
../vadclip_data/work_ucf_residual/compare_bottom10/perlayer_top64/
  neurons/selected_neurons.json              # 配对 ShiftScore、normal mean/std、每层 top-64
  concat_neurons/selected_neurons.json       # 768+512=1280D 拼接契约
  features/train/*.npy                       # 保留 UCF __0..__9 的每行训练变体
  features/test/*.npy                        # 测试 1280D 特征
  ucf_concat_train.csv / ucf_concat_test.csv
  training/checkpoint_last.pth               # 每个 epoch 的可恢复状态
  training/model_best.pth                    # 最大 logits1 ROC-AUC 的模型
  training/history.csv / run_config.json
  evaluation/per_video/*.npz                 # 可逐视频复用的预测
  evaluation/metrics.json
```

训练意外中断后，在原训练命令末尾加 `--resume`；它会恢复模型、optimizer、scheduler、epoch 和最佳指标。伪分数、拼接特征和测试会默认跳过已完成且契约正确的单项文件。数据变更或要完全重算时，对对应命令加 `--clean`，不要对其他阶段盲目清理。
