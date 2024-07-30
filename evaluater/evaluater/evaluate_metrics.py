import evaluate
from typing import List, Tuple
from nltk import word_tokenize
from evaluater.bart_score import BARTScorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def meteor_eval(predictions:List[str], references:List[List[str]]):
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=predictions, references=references)
    return results['meteor']

# def ppl_eval(predictions:List[str]):
#     perplexity = evaluate.load("perplexity", module_type="metric")
#     results = perplexity.compute(model_id='gpt2-medium',
#                                 add_start_token=False,
#                                 predictions=predictions,
#                                 device='cuda')
#     return results['mean_perplexity']

def bertscore_eval(predictions:List[str], references:List[List[str]]):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
        rescale_with_baseline=True,
        lang="en"
    )
    return sum(results['f1']) / len(results['f1'])

def bleu_eval(predictions:List[str], references:List[List[str]]):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    return results['bleu']*100


def google_bleu_eval(predictions:List[str], references:List[List[str]]):
    google_bleu = evaluate.load("google_bleu")
    results = google_bleu.compute(predictions=predictions, references=references)
    return results['google_bleu']


# def bleurt_eval(predictions:List[str], references:List[str], bleurt_20:bool = False):
#     from bleurt import score
#     checkpoint = '/data1/juseondo/bridge_inputs/BLEURT-20' if bleurt_20 else  'bleurt-base-128'
#     scorer = score.BleurtScorer(checkpoint)
#     scores = scorer.score(references=references, candidates=predictions)
#     assert isinstance(scores, list) and len(scores) == 1
#     return scores[0]


# def bleurt_eval(predictions:List[str], references:List[str]):
#     assert len(predictions) == len(references)
#     tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-128")
#     model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-128")
#     model.to('cuda')
#     model.eval()

#     with torch.no_grad():
#         scores = model(**tokenizer(references, predictions, return_tensors='pt', padding=True, truncation=True).to('cuda'))[0].squeeze().tolist()

#     return sum(scores)/len(scores)


def bartscore_eval(predictions:List[str], references:List[str]):
    assert len(predictions) == len(references)
    bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
    score = bart_scorer.score(predictions, references, batch_size=4)
    return sum(score)/len(score)

def rouge_eval(predictions:List[str], references:List[str]):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    return results

