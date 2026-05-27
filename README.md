# CIFAR-10 无限制对抗攻击竞赛 - A 同学部分

本项目对应第二次大作业中 A 同学负责的内容：基础工程、数据加载、代理模型训练、FGSM/PGD 基础攻击和 baseline 记录。

## 已实现功能

- 统一数据加载：读取 `dataset/images` 与 `dataset/label.txt`，不修改 `label.txt`。
- 数据增强：随机裁剪、水平翻转、颜色抖动、CIFAR-10 标准归一化。
- 代理模型：ResNet-18、ResNet-34、VGG-16、DenseNet-121。
- 基础攻击：FGSM、PGD。
- baseline 评估：记录 clean accuracy、ASR、SSIM、本地 proxy score。

## 目录结构

```text
.
|-- attacks/
|   |-- fgsm.py
|   |-- pgd.py
|   `-- common.py
|-- data/
|   `-- loader.py
|-- models/
|   `-- architectures.py
|-- train_proxy.py
|-- eval_attack_baseline.py
|-- run_all_a.cmd
|-- README_A.md
|-- requirements.txt
`-- dataset/
    |-- images/
    `-- label.txt
```

## 环境

推荐使用 Python 3.10 或 3.11。当前代码已在 `ai-learn` 环境中跑通。

CPU 版依赖安装：

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow tqdm
```

GPU 版请根据本机 CUDA 版本安装对应 PyTorch。

检查 PyTorch 是否可用：

```bat
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

## 训练

第一次使用公开 CIFAR-10 时需要下载：

```bat
python train_proxy.py --model resnet18 --epochs 80 --batch-size 128 --include-public-cifar10 --download-cifar10 --output-dir models_public
```

之后训练其他模型不需要再加 `--download-cifar10`：

```bat
python train_proxy.py --model resnet34 --epochs 80 --batch-size 128 --include-public-cifar10 --output-dir models_public
python train_proxy.py --model vgg16 --epochs 100 --batch-size 128 --include-public-cifar10 --output-dir models_public
python train_proxy.py --model densenet121 --epochs 100 --batch-size 64 --include-public-cifar10 --output-dir models_public
```

也可以直接运行自动脚本：

```bat
run_all_a.cmd
```

该脚本会顺序训练四个模型并跑 FGSM/PGD baseline，中途失败会停止。

## 攻击评估

FGSM：

```bat
python eval_attack_baseline.py --checkpoint models_public\resnet18_best.pt --attack fgsm --epsilon 0.031372549 --output baseline_results_public.csv
```

PGD：

```bat
python eval_attack_baseline.py --checkpoint models_public\resnet18_best.pt --attack pgd --epsilon 0.031372549 --alpha 0.007843137 --steps 10 --output baseline_results_public.csv
```

其中：

- `epsilon=0.031372549` 等于 `8/255`
- `alpha=0.007843137` 等于 `2/255`
- `steps=10` 是 PGD 迭代步数

## 当前结果

四个代理模型均已训练完成，验证集 clean accuracy 如下：

| Model | Epoch | Val Acc |
| --- | ---: | ---: |
| ResNet-18 | 80 | 0.93 |
| ResNet-34 | 80 | 0.94 |
| VGG-16 | 100 | 0.94 |
| DenseNet-121 | 100 | 0.96 |

本地白盒 PGD baseline：

| Model | Clean Acc | ASR | Mean SSIM |
| --- | ---: | ---: | ---: |
| ResNet-18 | 0.962 | 1.000 | 0.9917 |
| ResNet-34 | 0.990 | 1.000 | 0.9924 |
| VGG-16 | 0.958 | 1.000 | 0.9918 |
| DenseNet-121 | 0.974 | 1.000 | 0.9922 |

这些结果说明本地代理模型质量较高，PGD 白盒攻击已充分跑通。但平台是黑盒隐藏模型，最终迁移效果仍需 B/C 同学用 MI-FGSM、集成攻击、输入变换和提交反馈继续验证。

## A 同学交付物

建议交付给组员：

```text
models_public/resnet18_best.pt
models_public/resnet34_best.pt
models_public/vgg16_best.pt
models_public/densenet121_best.pt
baseline_results_public.csv
attacks/fgsm.py
attacks/pgd.py
data/loader.py
models/architectures.py
train_proxy.py
eval_attack_baseline.py
README.md
README_A.md
```
