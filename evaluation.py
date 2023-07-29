#auto evaluation

import os
import numpy as np
np.random.seed(17)
import random
random.seed(17)
import torch
torch.manual_seed(17)
torch.cuda.manual_seed(17)
torch.cuda.manual_seed_all(17)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification, RobertaTokenizerFast, RobertaForSequenceClassification, BertForMaskedLM, BartForConditionalGeneration, BartTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import bert_score
import math
from torch import cuda
import statistics
from nltk.tokenize import sent_tokenize
from torch.nn import CrossEntropyLoss
import json
import re

import torch
from tqdm import tqdm


bert_scorer = bert_score.BERTScorer(use_fast_tokenizer=True, lang='en')

device = 'cuda' if cuda.is_available() else 'cpu'

bert_tok_path = os.path.join("./mlm_model", "model_files_"+str(16)+"_"+str(5)+"_"+str(5e-5))
bert_tokenizer = BertTokenizer.from_pretrained(bert_tok_path)

bart_path = os.path.join("./eval_bart", "model_files_"+str(8)+"_"+str(3)+"_"+str(0.0001))
bart_tokenizer = BartTokenizer.from_pretrained(bart_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_path)
bart_model.to(device)
bart_model.eval()

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")


with open("./iwf_full.txt", 'r') as f_iwf:
  iwf_score = [float(line.strip()) for line in f_iwf.readlines()]


def lm_score(src_text, tgt_text, has_iwf=True):
  # compute the log probability of pre-trained models
  batch = bart_tokenizer(src_text, truncation=True, padding='longest',
                         return_tensors="pt").to(device)
  labels = bart_tokenizer(tgt_text, truncation=True, padding='longest', add_special_tokens=False,
                          return_tensors="pt").to(device)

  if has_iwf:
    tgt_score = [max([iwf_score[token_id] for token_id in
                      labels['input_ids'][label_id].cpu().numpy() if token_id<=50264]) for label_id in
                 range(labels['input_ids'].shape[0]) if label_id<=50264]
  else:
    tgt_score = []

  output = bart_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                      labels=labels['input_ids'])

  #print(labels['input_ids'].view(-1).shape)
  logits = output.logits.view(-1, bart_model.config.vocab_size)
  #print(logits.shape)
  loss_func = CrossEntropyLoss()
  loss = loss_func(logits, labels['input_ids'].view(-1))
  tgt_len = labels['attention_mask'].sum(dim=1)
  loss = loss.view(labels['input_ids'].shape[0], -1)
  loss = loss.sum(dim=1) / tgt_len

  #print(tgt_score)
  return loss, tgt_score


def bart_coh(mwps):
  cohs = []
  for mwp in mwps:
    mwp_split = [sent_tokenize(mwp)]
    #print(mwp_split)

    def get_mask_data(data_list):
      # mask each sentence respectively
      src_list, tgt_list, len_list = [], [], []
      for data_ele in data_list:
        src_list_ele, tgt_list_ele = [], []
        for idx in range(len(data_ele)):
          tgt_list_ele.append(data_ele[idx])
          src_list_ele.append(' '.join(data_ele[:idx]) + ' <mask_1> ' + ' '.join(data_ele[idx + 1:]))
        src_list.extend(src_list_ele)
        tgt_list.extend(tgt_list_ele)
        len_list.append(len(data_ele))
      return src_list, tgt_list, len_list

    src_data, tgt_data, data_len = get_mask_data(mwp_split)
    #print(src_data, tgt_data, data_len)

    # eval_score: score of each pattern evaluator
    # beta: (unnormalized) weight factor of each pattern evaluator
    eval_score, beta = [], []
    batch_size = 1
    for data_id in range(0, len(src_data), 1):
      src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
      bart_model.eval()
      with torch.no_grad():
        loss, tgt_score = lm_score(src_text, tgt_text)
        cur_score = [-loss_ele.detach().cpu().numpy() for loss_ele in loss]

      eval_score.extend(cur_score)
      beta.extend(tgt_score)

    data_st = 0
    res_score = []
    for len_ele in data_len:
      if sum(beta[data_st: data_st + len_ele]) > 0:
        res_score.append(np.dot(eval_score[data_st: data_st + len_ele], beta[data_st: data_st + len_ele]) /
                         sum(beta[data_st: data_st + len_ele]))
      else:
        res_score.append(np.mean(eval_score[data_st: data_st + len_ele]))
      data_st += len_ele
    #print(res_score)
    cohs.append(res_score[0])
  return cohs


def hamming(mwps, seeds):
  batch = torch.tensor(
    bert_tokenizer(seeds, padding="max_length", max_length=100, truncation=True, return_tensors="pt")[
      "input_ids"]).to(device)
  batch_new = bert_tokenizer(mwps, padding="max_length", max_length=100, truncation=True, return_tensors="pt")[
      "input_ids"].to(device)
  distance = np.sum(1 - np.array((batch_new == batch).detach().cpu()) * 1, axis=-1) / np.array([len(i) for i in seeds])
  print(np.sum(1 - np.array((batch_new == batch).detach().cpu()) * 1, axis=-1))
  print(np.array([len(i) for i in seeds]))
  return distance


def get_bert_score(mwps, seed_text):
    P, R, F1 = bert_scorer.score(mwps, seed_text, verbose=False, batch_size=len(mwps))
    return np.array(F1)


def bert_score(mwps, seeds):
  bert_score = get_bert_score(mwps, seeds)
  return bert_score


def read_mwps(f_dir):
  f = open(f_dir)
  csvreader = csv.reader(f)
  mwps = []
  seeds = []
  rows = []
  for row in csvreader:
    mwp = row[0]
    mwp = bert_tokenizer.encode(mwp)
    mwp = bert_tokenizer.decode(mwp, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    mwps.append(mwp)
    if len(row) > 5:
      seed = row[5]
      seed = bert_tokenizer.encode(seed)
      seed = bert_tokenizer.decode(seed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
      seeds.append(seed)
      rows.append(row)
    else:
      rows.append(row)
  return mwps, seeds, rows


def gpt2_ppl(mwps):
  res = []
  for mwp in mwps:
    encodings = gpt2_tokenizer(mwp, return_tensors="pt")

    max_length = gpt2_model.config.n_positions
    stride = 1
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc
      input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=target_ids)

        neg_log_likelihood = outputs.loss * trg_len

      nlls.append(neg_log_likelihood)

      prev_end_loc = end_loc
      if end_loc == seq_len:
        break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    res.append(ppl.item())
  return res

# replace with result dirs, result files in csv
# columns for the csv files: result mwp, difficulty level, task, original topic, target topic, seed mwp
in_dir = []
for dir in in_dir:
  if dir.endswith("csv"):
    mwps, seeds, rows = read_mwps(dir)
  print("read done")
  ppls2 = gpt2_ppl(mwps)
  print("ppl2 done")
  cohs = bart_coh(mwps)
  print("coh done")

  data = []
  out_dir = dir.split(".")[0] + "_eval_diff.csv"
  of = open(out_dir, "w")
  writer = csv.writer(of)

  if len(seeds) != 0:
    hams = hamming(mwps, seeds)
    berts = bert_score(mwps, seeds)

    for i in range(len(mwps)):
      data = rows[i] + [ppls2[i], cohs[i], hams[i], berts[i]]
      writer.writerow(data)

  else:
    for i in range(len(mwps)):
      data = rows[i] + [ppls2[i], cohs[i]]
      writer.writerow(data)
