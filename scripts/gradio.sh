CUDA_VISIBLE_DEVICES=7 python src/serve/app.py \
    --model-base Qwen2.5-VL-3B-Instruct \
    --model-path output/test_mdpo/checkpoint-10 \
    --device cuda:0 \
    --disable_flash_attention