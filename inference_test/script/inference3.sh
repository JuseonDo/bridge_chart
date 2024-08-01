CUDA_VISIBLE_DEVICES=4,5 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs=bridge \
    --template_number='auto-cot' \
    --batch_size=1 \
    --model_name="llava-hf/llava-1.5-7b-hf" \

CUDA_VISIBLE_DEVICES=4,5 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs=bridge \
    --template_number='auto-cot' \
    --batch_size=1 \
    --model_name="llava-hf/llava-1.5-7b-hf" \