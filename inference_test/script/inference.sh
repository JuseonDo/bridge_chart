CUDA_VISIBLE_DEVICES=0,1,2,3 python /data1/juseondo/bridge_chart/inference_test/inference.py \
    --task=chart2text \
    --inputs=image_text \
    --template_number='5' \
    --batch_size=8 \
    --model_name="llava-hf/llava-1.5-7b-hf" \