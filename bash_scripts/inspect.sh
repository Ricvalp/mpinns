#!/bin/bash

# Default values
SEED=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 -s|--seed <seed> -d|--dataset <dataset>"
            exit 1
            ;;
    esac
done

# Check if required parameters are provided
if [ -z "$SEED" ]; then
    echo "Missing required parameters"
    echo "Usage: $0 -s|--seed <seed> -d|--dataset <dataset>"
    exit 1
fi


PYTHONPATH=. NETWORKX_AUTOMATIC_BACKENDS="networkx" JAX_DISABLE_JIT="False" python fit/inspec.py $SEED $DATASET