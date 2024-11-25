#!/bin/bash
#SBATCH --account=""
#SBATCH --partition=""
#SBATCH --job-name=codealpaca-personified
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive

source .env/bin/activate

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo hostname=`hostname`

# Args
# --batch_size: The batch size
# --seed_dataset: The path to the seed dataset
# --db_path: The path to the database
# Default Args
batch_size=128
db_path=".cache/codealpaca_personified.db"
seed_dataset="seeds.jsonl"
num_samples=500_000
num_prompt_instructions=3
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch_size)
            batch_size="$2"
            shift
            shift
            ;;
        --db_path)
            db_path="$2"
            shift
            shift
            ;;
        --seed_dataset)
            seed_dataset="$2"
            shift
            shift
            ;;
        --num_samples)
            num_samples="$2"
            shift
            shift
            ;;
        --num_prompt_instructions)
            num_prompt_instructions="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$seed_dataset" ] || [ -z "$batch_size" ]; then
    echo "Error: Both --seed_dataset and --batch_size must be provided."
    exit 1
fi

# Launch the server in the background
echo "Launching the LM server..."
python -m sglang.launch_server \
    --model-path "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --port 8002 \
    --dp 2 \
    --tp 4  \
    --mem-fraction-static 0.8 &
# python -m sglang.launch_server \
#    --model-path "meta-llama/Llama-3.1-405B-Instruct-FP8" \
#    --port 8002 \
#    --dp 1 \
#    --tp 8 \
#    --mem-fraction-static 0.9 \
#    --kv-cache-dtype fp8_e5m2 \
#    --disable-cuda-graph &

# Wait for the server to be ready - usually takes 3min
echo "Waiting for the server to be ready..."
sleep 400  # Usually enough time...
echo "Server is ready"

echo "🏃 Starting the CodeAlpaca-Personified generation..."

python codealpaca_personified.py \
    --db_path ${db_path} \
    --seed_dataset ${seed_dataset} \
    --batch_size ${batch_size} \
    --num_samples ${num_samples} \
    --num_prompt_instructions ${num_prompt_instructions}

echo "🚀 CodeAlpaca-Personified generation complete!"
