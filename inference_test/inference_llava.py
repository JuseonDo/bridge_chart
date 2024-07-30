from utils import output_save
from utils import Chart2Text,ChartQA
from utils import get_template
from utils import post_processing, extract
import evaluater
from nltk import word_tokenize
import os
import fire
from tqdm import tqdm
import gc
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from accelerate import Accelerator
from transformers import BitsAndBytesConfig


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def tokenize(sample):
    return ' '.join(word_tokenize(sample))


def make_cell_for_llava(instruction):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text":instruction},
                {"type": "image"},
                ],
        },
    ]
    return conversation

summary_path = '/data1/juseondo/bridge_inputs/Chart-to-text/baseline_models/Chart2Text/data/test/testOriginalSummary.txt'
data_path = '/data1/juseondo/bridge_inputs/evaluater/evaluater/data_for_eval/chart2text_statista/testData.txt'
title_path = '/data1/juseondo/bridge_inputs/evaluater/evaluater/data_for_eval/chart2text_statista/testTitle.txt'

with open(summary_path) as f:
    references = [line.strip() for line in f]
references = list(map(tokenize, references))
references = references[:100] + references[-100:]

with open(title_path) as f:
    title_list = [line.strip() for line in f]
title_list = title_list[:100] + title_list[-100:]

with open(data_path) as f:
    data_list = [line.strip() for line in f]
data_list = data_list[:100] + data_list[-100:]




accelerator = Accelerator()
# model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
model_name = "llava-hf/llava-v1.6-34b-hf"
nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

processor = LlavaNextProcessor.from_pretrained(model_name)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=nf4_config,
)

model = accelerator.prepare(model)


def main(
        task:str, # chart2text, chartqa
        inputs:str, # image, text, iamge_text, bridge
        template_number:str,
        bridge_template_number:any = None,
):
    global references, title_list, data_list
    template_number = str(template_number)
    if bridge_template_number is not None:
        bridge_template_number = str(bridge_template_number)
        draft_path = f'/data1/juseondo/bridge_inputs/drafts/{task}/task:{task},inputs:bridge,temp:{bridge_template_number}.txt'
        print('-'*40)
        print('*** Using Draft Mode ***')
        print('draft_path:',draft_path)
        print('-'*40)
        with open(draft_path) as f:
            drafts = [line.replace('[[SEP]]','\n').strip() for line in f]
        
        drafts = drafts[:100] + drafts[-100:]

        assert len(data_list) == len(drafts)
    else: 
        drafts = [None] * len(data_list)
        print('Not Using Draft')

    save_path = '/data1/juseondo/bridge_inputs/llava_result/chart2text/'
    file_name = f'task:{task},inputs:{inputs},temp:{template_number},draft:{bridge_template_number}.txt'
    print(file_name)
    output_save_path = os.path.join(save_path, 'outputs/', file_name)
    result_save_path = os.path.join(save_path, 'results/', file_name)

    if os.path.exists(output_save_path):
        with open(output_save_path) as f:
            predictions = [extract(line).strip() for line in f]
        start_cnt = len(predictions)
        print(f"Already saved outputs {start_cnt} line exits.")
        if len(predictions) > 0: print(predictions[0])
    else:
        predictions = []
        start_cnt = len(predictions)

    chart2text = Chart2Text(split_list=['test']) if task == 'chart2text' else ChartQA(split_list=['test'])
    dataset = chart2text.data_dict
    
    template = get_template(task, inputs, template_number)

    dataset = dataset[:100] + dataset[-100:]

    print(len(references), len(title_list), len(data_list), len(dataset), len(drafts))

    cnt = 0
    for line, draft in zip(dataset[start_cnt:], tqdm(drafts[start_cnt:])):
        image_path = line['image_path'] if inputs != 'text' else None
        
        title = line['title']
        data = line['data']
        columns = data.split('\n')[0].strip()
        data = data.replace(columns.strip(),'').strip()

        instruction = template.format(table=data, columns=columns, title=title, draft=draft)
        
        image = Image.open(image_path)
        messages = make_cell_for_llava(instruction)
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(prompt, image, return_tensors="pt").to(accelerator.device)


        output = model.generate(**inputs, max_new_tokens=100, do_sample=True)
        print(processor.decode(output[0], skip_special_tokens=True))
        output_save(output_save_path, output)
        predictions.append(output)
        cnt += 1

    assert len(references) == len(predictions) == len(title_list) == len(data_list)

    predictions, references = post_processing(predictions,references)
    tokenized_predictions = list(map(tokenize, predictions))

    bleu = evaluater.bleu_eval(tokenized_predictions,references)
    print("bleu:", bleu)
    
    meteor = evaluater.meteor_eval(tokenized_predictions,references)
    print("meteor:", meteor)
    
    rouge = evaluater.rouge_eval(tokenized_predictions,references)
    print("rouge", rouge)
    
    cs = evaluater.cs_eval(tokenized_predictions, references, title_list, data_list)
    print("cs:", cs)
    
    bertscore = evaluater.bertscore_eval(tokenized_predictions,references)
    print("bertscore:", bertscore)
    clear_gpu_memory()

    ppl = evaluater.ppl_eval(tokenized_predictions)
    print("ppl:", ppl)
    clear_gpu_memory()

    bartscore = evaluater.bartscore_eval(tokenized_predictions,references)
    print("bartscore:", bartscore)
    clear_gpu_memory()

    # bleurt = evaluater.bleurt_eval(tokenized_predictions,references)
    # print("bleurt:", bleurt)
    # clear_gpu_memory()

    results = f"""{file_name}
bleu: {bleu}
meteor: {meteor}
rouge: {rouge}
cs: {cs}
bertscore: {bertscore}
ppl: {ppl}
bartscore: {bartscore}
"""
    with open(result_save_path, 'w') as f:
        f.write(results)


if __name__ == '__main__':
    fire.Fire(main)