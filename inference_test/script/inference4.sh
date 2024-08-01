CUDA_VISIBLE_DEVICES=6,7 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs=text \
    --template_number='6' \
    --batch_size=1 \
    --table_type='dict' \
    --model_name="llava-hf/llava-1.5-7b-hf" \

CUDA_VISIBLE_DEVICES=6,7 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs=text \
    --template_number='6' \
    --batch_size=1 \
    --table_type='dict' \
    --model_name="llava-hf/llava-1.5-7b-hf" \

CUDA_VISIBLE_DEVICES=6,7 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs=text \
    --template_number='6' \
    --batch_size=1 \
    --table_type='csv' \
    --model_name="llava-hf/llava-1.5-7b-hf" \

CUDA_VISIBLE_DEVICES=6,7 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs=text \
    --template_number='6' \
    --batch_size=1 \
    --table_type='csv' \
    --model_name="llava-hf/llava-1.5-7b-hf" \