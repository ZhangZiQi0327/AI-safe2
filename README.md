# CIFAR-10 无限制对抗攻击竞赛

三人协作项目：在尽量不改变图片肉眼观感的情况下，骗过多个未知 AI 图像分类模型。

## 项目结构

```text
.
|-- attacks/              # 攻击算法
|   |-- fgsm.py           # FGSM 单步攻击
|   |-- pgd.py            # PGD 迭代攻击
|   |-- mi_fgsm.py        # MI-FGSM 动量迭代攻击
|   |-- ensemble.py       # 集成攻击
|   `-- common.py         # 共享工具（归一化、epsilon转换等）
|-- data/                 # 数据加载
|   `-- loader.py
|-- models/               # 代理模型架构
|   `-- architectures.py
|-- pipeline/             # 批量生成管线（C同学）
|   |-- generate.py       # 批量生成对抗样本
|   |-- grid_search.py    # 参数网格搜索
|   `-- ssim_analysis.py  # SSIM质量分析
|-- models_public/        # 训练好的模型权重
|-- dataset/              # 500张竞赛图片
|-- train_proxy.py        # 模型训练脚本
|-- eval_attack_baseline.py   # 基线评估
|-- eval_transfer_attacks.py  # 迁移攻击评估
`-- search_transfer_params.py # 参数搜索
```

## 快速开始

### 环境

```bat
pip install torch torchvision numpy pillow tqdm
```

### 生成对抗样本（推荐）

```bat
python pipeline/generate.py ^
  --attack ensemble ^
  --checkpoints models_public/resnet18_best.pt models_public/resnet34_best.pt models_public/vgg16_best.pt models_public/densenet121_best.pt ^
  --epsilon 0.0510 --alpha 0.0098 --steps 20 --decay 1.0 ^
  --diversity-prob 0.3 --resize-rate 0.85 ^
  --output output_eps13
```

输出 `output_eps13/adversarial_images.zip`，可直接提交。

### 提交zip格式

```
adversarial_images.zip
├── images/
│   ├── 0.png ~ 499.png
└── label.txt
```

## 三人分工

| 角色 | 负责内容 | 状态 |
|------|---------|------|
| A | 代理模型训练 + 基础攻击 | 已完成 |
| B | 强迁移攻击研究（MI-FGSM、Ensemble） | 已完成 |
| C | 批量生成管线 + 调参 + 提交 | 已完成 |

## 当前最优配置

- 攻击方法：Ensemble（4模型集成）
- 参数：eps=0.0510, alpha=0.0098, steps=20, decay=1.0, diversity_prob=0.3, resize_rate=0.85
- 平台得分：13.61（2026-05-29）

详细说明见 [README_A.md](README_A.md)（A同学）和 [README_B.md](README_B.md)（B同学）。
