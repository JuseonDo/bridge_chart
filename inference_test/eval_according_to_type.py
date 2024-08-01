import os
ws_path = "/data1/juseondo/bridge_chart"
os.environ['WS_PATH'] = ws_path

from data_utils import (
    Chart2Text,
    ChartQA,
    get_data_for_eval,
    get_drafts
)
from utils import post_processing, extract
from utils import clear_gpu_memory
import evaluater

import torch
from nltk import word_tokenize
import fire
from tqdm import tqdm
import json
import pandas as pd

def tokenize(sample):
    return ' '.join(word_tokenize(sample))

def multiple_metrics_evaluation(predictions,references,titles,data_list):
    if len(predictions) <= 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    ref_token_len = [len(ref.split()) for ref in references]
    avg_ref_token_len = sum(ref_token_len)/len(ref_token_len)
    print("avg_ref_token_len:",avg_ref_token_len)

    token_len = [len(pred.split()) for pred in predictions]
    avg_token_len = sum(token_len)/len(token_len)
    print("avg_token_len:",avg_token_len)

    bleu = evaluater.bleu_eval(predictions,references)
    print("bleu:",bleu)
    meteor = evaluater.meteor_eval(predictions,references)
    print("meteor:",meteor)
    rouge = evaluater.rouge_eval(predictions,references)
    print("rouge:",rouge)
    cs = evaluater.cs_eval(predictions, references, titles, data_list)
    print("cs:",cs)
    bertscore = evaluater.bertscore_eval(predictions,references)
    print("bertscore:",bertscore)
    clear_gpu_memory()
    ppl = evaluater.ppl_eval(predictions)
    print("ppl:",ppl)
    clear_gpu_memory()
    bartscore = evaluater.bartscore_eval(predictions,references)
    print("bartscore:",bartscore)
    clear_gpu_memory()
    return avg_ref_token_len,avg_token_len, bleu, meteor, rouge, cs, bertscore, ppl, bartscore


references, titles, data_list = get_data_for_eval('chart2text')
output_path = "/data1/juseondo/bridge_chart/result/chart2text/outputs/task:chart2text,inputs:image_text,temp:5,few_shot:False-10,draft:None.json"
with open(output_path) as f:
    outputs = [json.loads(line.strip()) for line in f]


df = pd.DataFrame({
    "id":[],
    "column_type":[],
    "chart_type":[],
    "output":[],
    "reference":[],
    "title":[],
    "data":[],
    "prompt":[]
})

for output,ref,title,data in zip(tqdm(outputs),references,titles,data_list):
    output["reference"] = ref
    output["title"] = title
    output["data"] = data

    df = df._append(output, ignore_index=True)

for chart_type in df["chart_type"].unique():
    filtered = df[df["chart_type"] == chart_type]
    predictions = filtered["output"].tolist()
    references = filtered["reference"].tolist()
    titles = filtered["title"].tolist()
    data_list = filtered["data"].tolist()

    predictions, references = post_processing(predictions,references)
    tokenized_predictions = list(map(tokenize, predictions))
    avg_ref_token_len,avg_token_len, bleu, meteor, rouge, cs, bertscore, ppl, bartscore = multiple_metrics_evaluation(tokenized_predictions,references,titles,data_list)
    results = f"avg_ref_token_len: {avg_ref_token_len}\navg_token_len: {avg_token_len}\nbleu: {bleu}\nmeteor: {meteor}\nrouge: {rouge}\ncs: {cs}\nbertscore: {bertscore}\nppl: {ppl}\nbartscore: {bartscore}"
    print(results)
    with open(output_path.rstrip('.json').replace('outputs','eval') + ',chart_type:' + chart_type + '.txt', 'w') as f:
        f.write(results)

for column_type in df["column_type"].unique():
    filtered = df[df["column_type"] == column_type]
    predictions = filtered["output"].tolist()
    references = filtered["reference"].tolist()
    titles = filtered["title"].tolist()
    data_list = filtered["data"].tolist()

    predictions, references = post_processing(predictions,references)
    tokenized_predictions = list(map(tokenize, predictions))
    avg_ref_token_len,avg_token_len, bleu, meteor, rouge, cs, bertscore, ppl, bartscore = multiple_metrics_evaluation(tokenized_predictions,references,titles,data_list)
    results = f"avg_ref_token_len: {avg_ref_token_len}\navg_token_len: {avg_token_len}\nbleu: {bleu}\nmeteor: {meteor}\nrouge: {rouge}\ncs: {cs}\nbertscore: {bertscore}\nppl: {ppl}\nbartscore: {bartscore}"
    print(results)
    with open(output_path.rstrip('.json').replace('outputs','eval') + ',column_type:' + column_type + '.txt', 'w') as f:
        f.write(results)

