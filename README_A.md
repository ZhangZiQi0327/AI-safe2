# A 同学任务执行说明

## 你负责的交付物

- `requirements.txt`: 基础环境依赖。
- `data/loader.py`: 读取 `dataset/images` 和 `dataset/label.txt`，做分层训练/验证划分。
- `models/architectures.py`: ResNet-18、ResNet-34、VGG-16、DenseNet-121 四类代理模型。
- `train_proxy.py`: 代理模型训练脚本，包含随机裁剪、水平翻转、颜色抖动。
- `attacks/fgsm.py`: FGSM 单步攻击。
- `attacks/pgd.py`: PGD 迭代攻击。
- `eval_attack_baseline.py`: 记录各模型在 FGSM/PGD 下的 clean accuracy、ASR、SSIM 和代理分数。

`dataset/label.txt` 只会被读取，不会被任何脚本改写。

## 推荐运行顺序

先确认加载器正常：

```powershell
python -c "from data.loader import CifarAttackDataset; d=CifarAttackDataset('dataset'); print(len(d), d[0][1], d[0][2])"
```

快速冒烟训练：

```powershell
python train_proxy.py --model resnet18 --epochs 1 --batch-size 64
```

更推荐的正式训练方式：如果你已经有公开 CIFAR-10 数据，或网络允许下载，用 50k 公开训练集一起训练代理模型。只用比赛给的 500 张从零训练会明显过拟合，迁移攻击价值有限。

```powershell
python train_proxy.py --model resnet18 --epochs 80 --batch-size 128 --include-public-cifar10 --download-cifar10
python train_proxy.py --model resnet34 --epochs 80 --batch-size 128 --include-public-cifar10 --download-cifar10
python train_proxy.py --model vgg16 --epochs 100 --batch-size 128 --include-public-cifar10 --download-cifar10
python train_proxy.py --model densenet121 --epochs 100 --batch-size 64 --include-public-cifar10 --download-cifar10
```

如果网络不能下载，但你已经手动放好了 CIFAR-10：

```powershell
python train_proxy.py --model resnet18 --epochs 80 --include-public-cifar10 --cifar10-root data_cache
```

只能使用当前 500 张图时，按模型分别跑：

```powershell
python train_proxy.py --model resnet18 --epochs 80 --batch-size 64
python train_proxy.py --model resnet34 --epochs 80 --batch-size 64
python train_proxy.py --model vgg16 --epochs 100 --batch-size 64
python train_proxy.py --model densenet121 --epochs 100 --batch-size 32
```

CPU 会比较慢。如果有 NVIDIA GPU，PyTorch 会自动使用 CUDA。

## baseline 记录

训练完成后对每个模型跑 FGSM 和 PGD：

```powershell
python eval_attack_baseline.py --checkpoint models/resnet18_best.pt --attack fgsm --epsilon 0.031372549
python eval_attack_baseline.py --checkpoint models/resnet18_best.pt --attack pgd --epsilon 0.031372549 --alpha 0.007843137 --steps 10
```

结果会追加写入 `baseline_results.csv`。建议至少记录：

- ResNet-18: FGSM, PGD
- ResNet-34: FGSM, PGD
- VGG-16: FGSM, PGD
- DenseNet-121: FGSM, PGD

## 调参建议

- `epsilon=8/255` 是保守起点，图像质量通常较稳。
- 如果 ASR 太低，可试 `12/255`、`16/255`，但要关注 SSIM 和肉眼质量。
- PGD 可从 `steps=10, alpha=2/255` 开始，再试 `steps=20`。
- 代理模型训练集只有 500 张，容易过拟合；保留验证集结果，给 B/C 同学说明模型可信度。
- 提交平台是黑盒未知模型，单个代理模型上的 ASR 只代表本地参考，真正有价值的是多架构集成后的迁移性。
