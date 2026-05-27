@echo off
setlocal

echo [1/12] Train ResNet-18
python train_proxy.py --model resnet18 --epochs 80 --batch-size 128 --include-public-cifar10 --output-dir models_public
if errorlevel 1 exit /b 1

echo [2/12] Evaluate ResNet-18 FGSM
python eval_attack_baseline.py --checkpoint models_public\resnet18_best.pt --attack fgsm --epsilon 0.031372549 --output baseline_results_public.csv
if errorlevel 1 exit /b 1

echo [3/12] Evaluate ResNet-18 PGD
python eval_attack_baseline.py --checkpoint models_public\resnet18_best.pt --attack pgd --epsilon 0.031372549 --alpha 0.007843137 --steps 10 --output baseline_results_public.csv
if errorlevel 1 exit /b 1

echo [4/12] Train ResNet-34
python train_proxy.py --model resnet34 --epochs 80 --batch-size 128 --include-public-cifar10 --output-dir models_public
if errorlevel 1 exit /b 1

echo [5/12] Evaluate ResNet-34 FGSM
python eval_attack_baseline.py --checkpoint models_public\resnet34_best.pt --attack fgsm --epsilon 0.031372549 --output baseline_results_public.csv
if errorlevel 1 exit /b 1

echo [6/12] Evaluate ResNet-34 PGD
python eval_attack_baseline.py --checkpoint models_public\resnet34_best.pt --attack pgd --epsilon 0.031372549 --alpha 0.007843137 --steps 10 --output baseline_results_public.csv
if errorlevel 1 exit /b 1

echo [7/12] Train VGG-16
python train_proxy.py --model vgg16 --epochs 100 --batch-size 128 --include-public-cifar10 --output-dir models_public
if errorlevel 1 exit /b 1

echo [8/12] Evaluate VGG-16 FGSM
python eval_attack_baseline.py --checkpoint models_public\vgg16_best.pt --attack fgsm --epsilon 0.031372549 --output baseline_results_public.csv
if errorlevel 1 exit /b 1

echo [9/12] Evaluate VGG-16 PGD
python eval_attack_baseline.py --checkpoint models_public\vgg16_best.pt --attack pgd --epsilon 0.031372549 --alpha 0.007843137 --steps 10 --output baseline_results_public.csv
if errorlevel 1 exit /b 1

echo [10/12] Train DenseNet-121
python train_proxy.py --model densenet121 --epochs 100 --batch-size 64 --include-public-cifar10 --output-dir models_public
if errorlevel 1 exit /b 1

echo [11/12] Evaluate DenseNet-121 FGSM
python eval_attack_baseline.py --checkpoint models_public\densenet121_best.pt --attack fgsm --epsilon 0.031372549 --output baseline_results_public.csv
if errorlevel 1 exit /b 1

echo [12/12] Evaluate DenseNet-121 PGD
python eval_attack_baseline.py --checkpoint models_public\densenet121_best.pt --attack pgd --epsilon 0.031372549 --alpha 0.007843137 --steps 10 --output baseline_results_public.csv
if errorlevel 1 exit /b 1

echo.
echo All A tasks finished. Baseline results:
type baseline_results_public.csv
