CUDA_VISIBLE_DEVICES=3 python src/eval/test_value.py \
    --json-file Benchmark.jsonl \
    --model-base Qwen3-VL-2B-Instruct \
    --model-path output/qwen3vl_2b_dpo_largebatch_4epoch_balance \
    --device cuda \
    --disable_flash_attention \
    --output-file eval_output/dpo_lora_3_2b_4epoch_Lbatch_balance.jsonl
    
