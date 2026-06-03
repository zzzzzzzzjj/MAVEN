CUDA_VISIBLE_DEVICES=3 python src/eval/test_value.py \
    --json-file Benchmark.jsonl \
    --model-base Qwen3-VL-2B-Instruct \
    --model-path output/qwen3vl_2b_samdpo \
    --device cuda \
    --disable_flash_attention \
    --output-file eval_output/qwen3vl_2b_samdpo.jsonl
    
