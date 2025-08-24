#!/bin/bash
# Simple Behavior Retrieval Runner Script

echo "================================"
echo "Running Behavior Retrieval"
echo "================================"

# Default parameters
NUM_TEST=5
VAE_EPOCHS=20
BC_EPOCHS=10
MAX_SAMPLES=5000
TRAIN=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_test)
            NUM_TEST="$2"
            shift 2
            ;;
        --vae_epochs)
            VAE_EPOCHS="$2"
            shift 2
            ;;
        --bc_epochs)
            BC_EPOCHS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --no_train)
            TRAIN=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --num_test N      Number of test predictions (default: 5)"
            echo "  --vae_epochs N    VAE training epochs (default: 20)"
            echo "  --bc_epochs N     BC training epochs (default: 10)"
            echo "  --max_samples N   Max training samples (default: 5000)"
            echo "  --no_train        Skip training (load pretrained models)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 run_behavior_retrieval.py --num_test $NUM_TEST --vae_epochs $VAE_EPOCHS --bc_epochs $BC_EPOCHS --max_samples $MAX_SAMPLES"

if [ "$TRAIN" = false ]; then
    CMD="$CMD --no_train"
fi

# Run Behavior Retrieval
echo "Command: $CMD"
echo ""
$CMD

echo "================================"
echo "Behavior Retrieval completed!"
echo "================================"