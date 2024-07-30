from dataclasses import dataclass
import os
from typing import List

ws_path = os.environ['WS_PATH']

@dataclass
class Chart2Text:
    task:str = 'Chart2Text'
    data_name:str = 'statista'
    data_path:str = os.path.join(ws_path,"Chart-to-text/")
    idx_path:str = "baseline_models/Chart2Text/data/{split}/sorted_{split}_mapping.csv"
    summary_path:str = "baseline_models/Chart2Text/data/{split}/{split}OriginalSummary.txt"

    def __init__(self, split_list:List[str] = 'test'):
        self.two_column_path:str = os.path.join(self.data_path, "statista_dataset/dataset/")
        self.multi_column_path:str = os.path.join(self.two_column_path, "multiColumn/")
        self.data_dict = []

        if not isinstance(split_list, list): split_list = [split_list]

        # for split in ['test', 'valid', 'train']:
        for split in split_list:
            self.summary_path = os.path.join(self.data_path, self.summary_path.format(split=split))
            with open(self.summary_path) as f:
                summaries = [line.strip() for line in f]
            variable = f'{split}_idx_path'
            path = os.path.join(self.data_path, self.idx_path.format(split=split))
            setattr(self, variable, path)
            with open(path) as f:
                for line, summary in zip(f.readlines()[1:], summaries):
                    type, id = line.strip().rstrip('.txt').split('-')
                    data_folder_path = self.two_column_path if 'two' in type else self.multi_column_path

                    data_path = os.path.join(data_folder_path,f'data/{id}.csv')
                    image_path = os.path.join(data_folder_path,f'imgs/{id}.png')
                    title_path = os.path.join(data_folder_path,f'titles/{id}.txt')

                    with open(data_path) as f:
                        data = f.read().strip()

                    with open(title_path) as f:
                        title = f.read().strip()

                    self.data_dict.append({
                        'data_path':data_path,
                        'data':data,
                        'image_path':image_path,
                        'title_path':title_path,
                        'title':title,
                        'summary':summary,
                    })
        
            


        


@dataclass
class ChartQA:
    name:str = 'ChartQA'
    data_path:str = '/data1/juseondo/bridge_chart/ChartQA/'