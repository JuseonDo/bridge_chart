import os
import json
from typing import Tuple, List

ws_path = os.environ["WS_PATH"]


def read_file(path):
    with open(path) as f:
        return [line.strip() for line in f]

def read_json(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


def get_data_for_eval(task:str) -> Tuple[List[str],List[str],List[str]]:
    """
    In Chart-to-Text, we need references, titles, data to evaluate.
    return references, titles, data
    """
    if task.lower() == "chart2text":
        summary_path = os.path.join(ws_path,'Chart-to-text/baseline_models/Chart2Text/data/test/testOriginalSummary.txt')
        data_path = os.path.join(ws_path,'evaluater/evaluater/data_for_eval/chart2text_statista/testData.txt')
        title_path = os.path.join(ws_path,'evaluater/evaluater/data_for_eval/chart2text_statista/testTitle.txt')
        
        references = read_file(summary_path)
        titles = read_file(title_path)
        data = read_file(data_path)
        assert len(references) == len(titles) == len(data)
        return references, titles, data

    else: raise ValueError("ChartQA is not implemented")