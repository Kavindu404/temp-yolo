#!/bin/bash

# Set default values
BACKBONE="resnet50"
DATA_PATH="/path/to/coco"
BATCH_SIZE=16
EPOCHS=100
LR=0.001
CHECKPOINT_FREQ=5
VAL_IMG_COUNT=8
OUTPUT_DIR="runs/exp"
CONFIG="configs/default.yaml"
DEVICE="cuda"
SEED=42

# Help function
show_help() {
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --backbone STR        ResNet backbone to use (default: resnet50)"
    echo "                         Available options: resnet18, resnet34, resnet50, resnet101, resnet152"
    echo "  --data_path PATH      Path to COCO dataset (required)"
    echo "  --batch_size INT      Batch size (default: 16)"
    echo "  --epochs INT          Number of epochs to train (default: 100)"
    echo "  --lr FLOAT           Learning rate (default: 0.001)"
    echo "  --checkpoint_freq INT Checkpoint save frequency in epochs (default: 5)"
    echo "  --val_img_count INT   Number of validation images to save (default: 8)"
    echo "  --output_dir PATH     Output directory (default: runs/exp)"
    echo "  --config PATH         Path to config file (default: configs/default.yaml)"
    echo "  --device STR          Device to use (default: cuda)"
    echo "  --seed INT            Random seed (default: 42)"
    echo "  --resume PATH         Path to checkpoint to resume from"
    echo "  --help                Show this help message and exit"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --backbone)
            BACKBONE="$2"
            shift
            shift
            ;;
        --data_path)
            DATA_PATH="$2"
            shift
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --lr)
            LR="$2"
            shift
            shift
            ;;
        --checkpoint_freq)
            CHECKPOINT_FREQ="$2"
            shift
            shift
            ;;
        --val_img_count)
            VAL_IMG_COUNT="$2"
            shift
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --config)
            CONFIG="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --seed)
            SEED="$2"
            shift
            shift
            ;;
        --resume)
            RESUME="$2"
            shift
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if data_path is provided
if [[ -z "$DATA_PATH" || "$DATA_PATH" == "/path/to/coco" ]]; then
    echo "Error: --data_path is required"
    show_help
    exit 1
fi

# Build command
CMD="python main.py \
    --backbone $BACKBONE \
    --data_path $DATA_PATH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --checkpoint_freq $CHECKPOINT_FREQ \
    --val_img_count $VAL_IMG_COUNT \
    --output_dir $OUTPUT_DIR \
    --config $CONFIG \
    --device $DEVICE \
    --seed $SEED"

# Add resume if provided
if [[ -n "$RESUME" ]]; then
    CMD="$CMD --resume $RESUME"
fi

# Print command
echo "Running: $CMD"

# Execute command
eval $CMD
