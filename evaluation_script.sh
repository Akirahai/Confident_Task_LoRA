models="Qwen/Qwen2.5-1.5B-Instruct"
peft_model="Qwen2.5-1.5B-Instruct_Confident_Task_LoRA_6750_final"

exp_dir="exp_14_11"

python run_evaluation.py \
    --gpus 1 2 \
    --model "$models" \
    --peft_model_path "$peft_model" \
    --task "maths" \
    --n_samples 100 \
    --experiment "$exp_dir" \
    --tensor_parallel_size 2