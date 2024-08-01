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
    bleurtscore = evaluater.bleurt_eval(predictions,references)
    print("bleurtscore:",bleurtscore)

def main():
    references, titles, data_list = get_data_for_eval('chart2text')

    output_save_path = "/data1/juseondo/bridge_chart/result/chart2text/outputs/task:chart2text,inputs:bridge,temp:auto-cot,few_shot:False-10,draft:None.txt"
    predictions,start_idx = generated_output_check(output_save_path)

    predictions = [extract(prediction) for prediction in predictions]

    predictions, references = post_processing(predictions,references)
    tokenized_predictions = list(map(tokenize, predictions))
    multiple_metrics_evaluation(tokenized_predictions,references,titles,data_list)

if __name__ == '__main__':
    fire.Fire(main)