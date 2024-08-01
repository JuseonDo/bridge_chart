CUDA_VISIBLE_DEVICES=2,3 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs='image' \
    --template_number='6' \
    --batch_size=5 \
    --table_type='dict' \
    --model_name="llava-hf/llava-1.5-7b-hf" \

CUDA_VISIBLE_DEVICES=2,3 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs='image' \
    --template_number='6' \
    --batch_size=5 \
    --table_type='dict' \
    --model_name="llava-hf/llava-1.5-7b-hf" \

CUDA_VISIBLE_DEVICES=2,3 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs='image' \
    --template_number='6' \
    --batch_size=5 \
    --table_type='csv' \
    --model_name="llava-hf/llava-1.5-7b-hf" \

CUDA_VISIBLE_DEVICES=2,3 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs='image' \
    --template_number='6' \
    --batch_size=5 \
    --table_type='csv' \
    --model_name="llava-hf/llava-1.5-7b-hf" \

# CUDA_VISIBLE_DEVICES=2,3 python /data1/juseondo/bridge_chart/inference_test/inference.py \
#     --task=chart2text \
#     --inputs=bridge \
#     --template_number='base' \
#     --draft_name='base' \
#     --batch_size=1 \
#     --model_name="llava-hf/llava-1.5-7b-hf" \
