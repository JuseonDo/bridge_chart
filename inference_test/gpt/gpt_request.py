from openai import OpenAI
import base64
from typing import List
from inference_test.data_utils import Chart2Text,ChartQA
import random

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt_inference(client:OpenAI, messages:List[dict], max_tokens:int = 200, model_name:str = 'gpt-4o', seed:int = 42) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=.0000000000000000000001,
        top_p=.0000000000000000000001,
        seed=seed,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def make_few_shots(task:str, inputs:str, template:str, number_of_shots:str = 10, bridge_template_number:any = None):
    dc = Chart2Text(split_list=['train']) if task == 'chart2text' else ChartQA(split_list=['train'])
    dataset = dc.data_dict
    random.seed(42)
    random.shuffle(dataset)
    messages = []
    
    if bridge_template_number is not None:
        drafts_path = f"/data1/juseondo/bridge_inputs/few_shot_drafts/{task}/task:{task},inputs:bridge,temp:{bridge_template_number}.txt"
        with open(drafts_path) as f:
            drafts = [line.replace("[[SEP]]","\n").strip() for line in f]
    else:
        drafts = ['0']*number_of_shots

    for line, draft in zip(dataset[:number_of_shots],drafts[:number_of_shots]):
        image_path = line['image_path'] if inputs != 'text' else None
        
        title = line['title']
        data = line['data']
        columns = data.split('\n')[0].strip()
        data = data.replace(columns.strip(),'').strip()
        summary = line['summary']

        instruction = template.format(table=data, columns=columns, title=title, draft=draft) + "<caption>" + summary + "</caption>"
        
        cell = make_cell(instruction, image_path)
        messages.extend(cell)
    return messages


def make_cell(instruction:str, image_path:any = None):
    cell = {
        "role": "user",
        "content":[
            {"type":"text","text": instruction}
        ]
    }
    if image_path is not None:
        base64_image = encode_image(image_path)
        cell["content"].append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        )
    return [cell]