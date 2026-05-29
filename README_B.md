# B 同学任务执行说明

## 你负责的交付物

- `attacks/mi_fgsm.py`: MI-FGSM 动量迭代攻击。
- `attacks/ensemble.py`: 多代理模型集成攻击与集成 logits 推理。
- `eval_transfer_attacks.py`: 迁移攻击评估脚本，支持 `fgsm`、`pgd`、`mi_fgsm`、`ensemble`。
- `search_transfer_params.py`: 参数搜索脚本，用于比较不同超参数和模型组合。
- `b_search_records.csv`: 第一轮迁移搜索明细。
- `b_search_summary.csv`: 第一轮集成攻击搜索汇总。
- `b_search_mi_records.csv`: 单模型 MI-FGSM 源模型消融与搜索明细。
- `b_search_mi_summary.csv`: 单模型 MI-FGSM 源模型消融汇总。
- `b_final_results.csv`: 全量 500 张图上的最终验证结果。
- `b_experiment_records.csv`: 早期方法对比记录。

## 我做了什么

- 在 A 同学提供的 4 个代理模型基础上，实现了更强的迁移攻击方法：
  - `MI-FGSM`
  - `Leave-One-Out Ensemble Attack`
- 在攻击迭代中加入了可选输入变换：
  - `resize + padding`
- 新建了迁移评估脚本，不再只看白盒结果，而是直接统计：
  - `source_clean_acc`
  - `target_clean_acc`
  - `transfer_asr_target_clean`
  - `transfer_asr_joint_clean`
  - `mean_ssim_target_clean`
  - `transfer_score_target_clean`
- 做了两轮调参：
  - 第一轮：比较 `FGSM / PGD / MI-FGSM / Ensemble`
  - 第二轮：系统搜索 `epsilon / alpha / steps / momentum / diversity_prob / resize_rate`
- 做了源模型消融，确认单模型迁移攻击里最强的源模型不是默认的 `resnet18`，而是 `densenet121`。
- 把最终最优配置在全量 500 张图上重新验证，形成最终实验结果。

## 关键结论

### 1. 单模型最优方案

最优单模型攻击来自 `densenet121 + MI-FGSM`。

推荐参数：

```text
epsilon = 10/255
alpha = 2/255
steps = 12
momentum = 0.75
diversity_prob = 0.3
resize_rate = 0.85
```

来源：

- `b_search_mi_summary.csv`
- `b_final_results.csv`

全量 500 张验证结果：

- 对 `resnet18` 目标：`transfer_score_target_clean = 95.80`
- 对 `resnet34` 目标：`transfer_score_target_clean = 97.68`
- 对 `vgg16` 目标：`transfer_score_target_clean = 93.09`

### 2. 最终推荐方案

最终最推荐给 C 同学接入生成管线的是 `Leave-One-Out Ensemble Attack`。

意思是：

- 如果目标模型是假想的 `resnet18`，就用 `resnet34 + vgg16 + densenet121` 做集成攻击
- 如果目标模型是假想的 `resnet34`，就用 `resnet18 + vgg16 + densenet121`
- 如果目标模型是假想的 `vgg16`，就用 `resnet18 + resnet34 + densenet121`
- 如果目标模型是假想的 `densenet121`，就用 `resnet18 + resnet34 + vgg16`

推荐参数：

```text
epsilon = 10/255
alpha = 2/255
steps = 12
momentum = 1.0
diversity_prob = 0.3
resize_rate = 0.85
```

来源：

- `b_search_summary.csv`
- `b_final_results.csv`

全量 500 张验证结果：

- 对 `resnet18` 目标：`transfer_score_target_clean = 97.27`
- 对 `resnet34` 目标：`transfer_score_target_clean = 97.89`
- 对 `vgg16` 目标：`transfer_score_target_clean = 96.43`
- 对 `densenet121` 目标：`transfer_score_target_clean = 97.73`

## 结论怎么理解

- `FGSM` 能快速验证管线，但迁移性明显不如后两者。
- `PGD` 比 `FGSM` 强，但在迁移场景下通常不如 `MI-FGSM` 稳定。
- `MI-FGSM` 明显优于基础攻击。
- `Ensemble Attack` 整体上优于单模型攻击，是当前最值得交给 C 同学用于最终出图的方案。
- 少量输入变换在当前任务上是有效的，但不宜把 `diversity_prob` 调得太高。

## 推荐给 C 同学怎么接

如果 C 同学要做最终 500 张出图，优先采用：

```text
attack = ensemble
epsilon = 10/255
alpha = 2/255
steps = 12
momentum = 1.0
diversity_prob = 0.3
resize_rate = 0.85
```

如果工程上只能先接单模型攻击，使用：

```text
source model = densenet121
attack = mi_fgsm
epsilon = 10/255
alpha = 2/255
steps = 12
momentum = 0.75
diversity_prob = 0.3
resize_rate = 0.85
```

## 常用命令

单模型 MI-FGSM 迁移评估：

```powershell
python eval_transfer_attacks.py `
  --dataset dataset `
  --source-checkpoints models_public\densenet121_best.pt `
  --target-checkpoints models_public\resnet18_best.pt models_public\resnet34_best.pt models_public\vgg16_best.pt `
  --attack mi_fgsm `
  --epsilon 0.0392156862745098 `
  --alpha 0.00784313725490196 `
  --steps 12 `
  --momentum 0.75 `
  --diversity-prob 0.3 `
  --resize-rate 0.85 `
  --batch-size 16 `
  --output b_final_results.csv
```

集成攻击迁移评估：

```powershell
python eval_transfer_attacks.py `
  --dataset dataset `
  --source-checkpoints models_public\resnet18_best.pt models_public\vgg16_best.pt models_public\densenet121_best.pt `
  --target-checkpoints models_public\resnet34_best.pt `
  --attack ensemble `
  --epsilon 0.0392156862745098 `
  --alpha 0.00784313725490196 `
  --steps 12 `
  --momentum 1.0 `
  --diversity-prob 0.3 `
  --resize-rate 0.85 `
  --batch-size 32 `
  --output b_final_results.csv
```

参数搜索：

```powershell
python search_transfer_params.py --dataset dataset --models-root models_public
```

## 备注

- 当前环境是 CPU，完整搜索和全量验证比较慢。
- B 部分已经把“强迁移攻击研究”需要的代码、参数搜索和最终推荐配置补齐。
- C 同学后续主要工作应是：接入批量生成、质量筛选、提交打包和根据平台反馈继续微调。
