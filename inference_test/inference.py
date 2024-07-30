import os
ws_path = "/data1/juseondo/bridge_chart"
os.environ['WS_PATH'] = ws_path

from data_utils import (
    Chart2Text,
    ChartQA,
    get_data_for_eval,
    get_drafts
)
from utils import get_template
from utils import post_processing, extract
from utils import clear_gpu_memory
from utils import generated_output_check
from model_utils import load_model, batch_inference
import evaluater

from transformers import BitsAndBytesConfig
from accelerate import Accelerator
import torch
from nltk import word_tokenize
import fire
from tqdm import tqdm
from PIL import Image
import resource

def use_gpt():
    from gpt import (
        get_client,
        gpt_inference,
        make_cell,
        make_few_shots,
    )

def tokenize(sample):
    return ' '.join(word_tokenize(sample))

def multiple_metrics_evaluation(predictions,references,titles,data_list):
    bleu = evaluater.bleu_eval(predictions,references)
    meteor = evaluater.meteor_eval(predictions,references)
    rouge = evaluater.rouge_eval(predictions,references)
    cs = evaluater.cs_eval(predictions, references, titles, data_list)
    bertscore = evaluater.bertscore_eval(predictions,references)
    clear_gpu_memory()
    ppl = evaluater.ppl_eval(predictions)
    clear_gpu_memory()
    bartscore = evaluater.bartscore_eval(predictions,references)
    clear_gpu_memory()
    return bleu, meteor, rouge, cs, bertscore, ppl, bartscore

def main(
        task:str, # chart2text, chartqa
        inputs:str, # image, text, iamge_text, bridge
        template_number:str,
        draft_name:any = None,
        batch_size:int = 16,
        model_name:str = "llava-hf/llava-1.5-7b-hf", ## llava, openflamingo, paligemma, gpt-4o
        few_shot:bool = False,
        number_of_shots:int = 10,
):
    
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))

    if 'gpt' in model_name.lower(): use_gpt()
    if not isinstance(batch_size,int): batch_size = int(batch_size)
    if not isinstance(few_shot,bool): few_shot = bool(few_shot)
    if not isinstance(template_number,str): template_number = str(template_number)

    if few_shot: raise ValueError("Few-shot setting is not implemented")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model, processor = load_model(model_name, quantization_config)
    accelerator = Accelerator()
    model, processor = accelerator.prepare(model, processor)

    references, titles, data_list = get_data_for_eval(task)

    save_path = os.path.join(ws_path,'result/chart2text/')
    file_name = f'task:{task},inputs:{inputs},temp:{template_number},few_shot:{few_shot}-{number_of_shots},draft:{draft_name}.txt'
    print(file_name)
    output_save_path = os.path.join(save_path, 'outputs/', file_name)
    eval_save_path = os.path.join(save_path, 'eval/', file_name)

    chart2text = Chart2Text(split_list=['test']) if task == 'chart2text' else ChartQA(split_list=['test'])
    dataset = chart2text.data_dict

    predictions,start_idx = generated_output_check(output_save_path)
    draft_path = os.path.join(ws_path,f'drafts/{task}/{draft_name}_draft.txt')
    drafts = get_drafts(draft_path)
    if drafts is None: drafts = [None] * len(dataset)

    template = get_template(task, inputs, template_number)

    prompts,image_paths = [],[]
    for line, draft in zip(dataset[start_idx:], tqdm(drafts[start_idx:])):
        image_path = line['image_path'] if inputs != 'text' else None
        
    
        title = line['title']
        data = line['data']
        columns = data.split('\n')[0].strip()
        data = data.replace(columns.strip(),'').strip()

        prompt = template.format(table=data, columns=columns, title=title, draft=draft)
        prompt = "USER: <image>\n" + prompt.strip() + "\nASSISTANT:"
        
        prompts.append(prompt)
        image_paths.append(image_path)

    print(prompts[0])
    print(image_paths[0])

    model_outputs = batch_inference(
        model=model,
        processor=processor,
        prompts=prompts,
        image_paths=image_paths,
        batch_size=batch_size,
        output_save_path=output_save_path,
        accelerator=accelerator,
    )
    predictions += model_outputs

    assert len(references) == len(predictions) == len(titles) == len(data_list)

    predictions = [extract(prediction) for prediction in predictions]

    predictions, references = post_processing(predictions,references)
    tokenized_predictions = list(map(tokenize, predictions))

    bleu, meteor, rouge, cs, bertscore, ppl, bartscore = multiple_metrics_evaluation(tokenized_predictions,references,titles,data_list)
    results = f"bleu: {bleu}\nmeteor: {meteor}\nrouge: {rouge}\ncs: {cs}\nbertscore: {bertscore}\nppl: {ppl}\nbartscore: {bartscore}"

    with open(eval_save_path, 'w') as f:
        f.write(results)


if __name__ == '__main__':
    fire.Fire(main)