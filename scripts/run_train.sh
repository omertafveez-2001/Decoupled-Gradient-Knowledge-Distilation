#!/bin/bash
cd "$(dirname "$0")/.."   # always run from the project root

# ── Model Architectures ────────────────────────────────────────────────────────
# resnet8 | resnet14 | resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | resnet8x4 | resnet32x4
# vgg8 | vgg8_bn | vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn
# ShuffleV2
TEACHER_ARCH="resnet56"
STUDENT_ARCH="resnet20"

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET="CIFAR-100"    # MNIST | CIFAR-10 | CIFAR-100 | SVHN | Food101 | TinyImageNet
AUGMENT="False"

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS=100
BATCH_SIZE=64
SEED=1

# ── Distillation ───────────────────────────────────────────────────────────────
DISTILLATION_METHOD="decoupled"    # logit_matching | decoupled | decoupled_gradient_distillation
LOSS_WEIGHTS="1.0 8.0 6.0"         # alpha beta epsilon
TEMPERATURE=4.0
WARMUP=20

# ── Checkpoints to load (leave empty to train from scratch) ───────────────────
TEACHER_CHECKPOINT="checkpoints/CIFAR100/resnet56_CIFAR-100.pth"
STUDENT_CHECKPOINT=""

# ── Save paths (leave empty to skip saving) ───────────────────────────────────
TEACHER_SAVE_PATH="teacher_${TEACHER_ARCH}_${DATASET}.pth"
STUDENT_SAVE_PATH="student_${STUDENT_ARCH}_${DATASET}.pth"
DISTILLATION_SAVE_PATH="distill_${DISTILLATION_METHOD}_${STUDENT_ARCH}_${DATASET}.pth"

# ── Robustness Evaluation (leave empty to skip) ───────────────────────────────
ROBUSTNESS_EVAL=""     # stylized | noised | scrambled
CUSTOM_DATA_PATH=""    # required only if ROBUSTNESS_EVAL=stylized

# ── Build and run ──────────────────────────────────────────────────────────────
CMD="python -m scripts.train \
    --teacher_arch $TEACHER_ARCH \
    --student_arch $STUDENT_ARCH \
    --dataset $DATASET \
    --augment $AUGMENT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --distillation_method $DISTILLATION_METHOD \
    --loss_weights $LOSS_WEIGHTS \
    --temperature $TEMPERATURE \
    --warmup $WARMUP \
    --seed $SEED"

[ -n "$TEACHER_CHECKPOINT" ]     && CMD="$CMD --teacher_checkpoint $TEACHER_CHECKPOINT"
[ -n "$STUDENT_CHECKPOINT" ]     && CMD="$CMD --student_checkpoint $STUDENT_CHECKPOINT"
[ -n "$TEACHER_SAVE_PATH" ]      && CMD="$CMD --teacher_save_path $TEACHER_SAVE_PATH"
[ -n "$STUDENT_SAVE_PATH" ]      && CMD="$CMD --student_save_path $STUDENT_SAVE_PATH"
[ -n "$DISTILLATION_SAVE_PATH" ] && CMD="$CMD --distillation_save_path $DISTILLATION_SAVE_PATH"
[ -n "$ROBUSTNESS_EVAL" ]        && CMD="$CMD --robustness_eval $ROBUSTNESS_EVAL"
[ -n "$CUSTOM_DATA_PATH" ]       && CMD="$CMD --custom_data_path $CUSTOM_DATA_PATH"

echo "Running: $CMD"
eval $CMD
