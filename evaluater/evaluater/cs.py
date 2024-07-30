from typing import List
from statistics import mean, stdev

def cs_eval(predictions: List[str], references: List[str], title_list: List[str], data_list: List[str]) -> dict:
    fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-', 'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', ';']
    
    generated_scores = []
    count = 0
    
    for datas, titles, reference, prediction in zip(data_list, title_list, references, predictions):
        gold_arr = reference.split()
        
        record_list = []
        for gld in gold_arr:
            data_string = datas.replace("_", " ")
            if gld.lower() in " ".join([data_string, titles]).lower() and gld.lower() not in fillers and gld.lower() not in record_list:
                record_list.append(gld.lower())
        
        list1 = record_list[:]
        record_length = len(record_list)
        generated_list = []
        
        for token in prediction.split():
            if token.lower() in list1:
                list1.remove(token.lower())
                generated_list.append(token.lower())
        
        count += 1
        
        if record_length == 0:
            generated_ratio = 0
        else:
            generated_ratio = len(generated_list) / record_length
        
        generated_scores.append(generated_ratio)
    
    results = {
        'generated_CS_stdev': round(stdev(generated_scores) * 100, 2),
        'generated_CS_mean': round(mean(generated_scores) * 100, 2),
        'generated_CS_RSD': round((stdev(generated_scores) * 100) / abs(mean(generated_scores)), 2)
    }
    
    return results['generated_CS_mean']