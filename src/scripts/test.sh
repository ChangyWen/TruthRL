set -x

source_dir=/mnt/blob_output/v-dachengwen/TruthRL/logs_math-grpo-1760692940
for i in 40 80 120; do
# for i in 160 200; do
    if [ -f $source_dir/global_step_${i}/actor/huggingface/model-00002-of-00002.safetensors ]; then
        echo "model has been merged already"
    else
        echo "model not merged, merging model ..."
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir $source_dir/global_step_${i}/actor \
            --target_dir $source_dir/global_step_${i}/actor/huggingface
    fi

    python src/scripts/test.py \
        --model_name $source_dir/global_step_${i}/actor/huggingface \
        --n_samples 16
done
