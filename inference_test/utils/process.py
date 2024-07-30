from typing import List
import re

def post_processing(hyps:List[str], tgts:List[str]):
    processed_hyps, processed_tgts = [], []
    for hyp, tgt in zip(hyps, tgts):
        hyp = hyp.replace('%',' percent ')
        tgt = tgt.replace('%',' percent ')
        if "'" not in tgt: hyp = hyp.replace("'","")
        if '"' not in tgt: hyp = hyp.replace('"',"")
        if "`" not in tgt: hyp = hyp.replace("`","")
        hyp = hyp.strip()
        tgt = tgt.strip()
        processed_hyps.append(hyp)
        processed_tgts.append(tgt)
    return processed_hyps, processed_tgts

def extract(output):
    result = re.search(r'<caption>(.*?)</caption>', output)
    if result:
        output = result.group(1)
    return output

def output_save(save_path, output):
    output = output.replace('\n','[[SEP]]')
    with open(save_path, 'a') as f:
        f.write(output + '\n')