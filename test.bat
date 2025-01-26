@echo off
python empty_cache.py
python train.py --teachermodel resnet101 --studentmodel resnet101 --dataset CIFAR-100 --batchsize 128 --epochs 15 --hyperparameters 0.5 0.5 0 6 0 3 --student_dir student --teacher_dir teacher --distill_dir distill --learningrate 0.001 --experiment "teacher"
python empty_cache.py
pause