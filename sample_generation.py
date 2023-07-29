
# adopted from "Mix and Match: Learning-free Controllable Text Generation using Energy Language Models" (https://github.com/mireshghallah/mixmatch)

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, BertForMaskedLM
import bert_score
from torch.distributions.categorical import Categorical
from datetime import datetime
import math
import time
from Levenshtein import distance as lev
import wandb
import argparse


################

topic_model_path = os.path.join("./topic_model", "model_files_"+str(32)+"_"+str(5)+"_"+str(2e-5))
#topic_model_path = os.path.join("./topic_model", "model_files_mod_"+str(32)+"_"+str(5)+"_"+str(2e-5))

eq_model_path = os.path.join("./eq_model", "model_files_"+"large_"+str(8)+"_"+str(8)+"_"+str(0.0001))

mlm_model_path = os.path.join("./mlm_model", "model_files_"+str(16)+"_"+str(5)+"_"+str(5e-5))
#mlm_model_path = os.path.join("./mlm_model", "model_files_large_"+str(16)+"_"+str(5)+"_"+str(5e-5))

parser = argparse.ArgumentParser(description="MWP generation")

parser.add_argument("--difficulty", type=str, help="difficulty level", default="basic")
parser.add_argument("--seed_setup", type=int, help="seed setup type", default=1)
parser.add_argument("--mlm_size", type=str, help="mlm size", default="bert-base")
parser.add_argument("--no_padding", action='store_true')
parser.add_argument("--max_len", type=int, help="max length of generation results", default=50)
parser.add_argument("--max_iter", type=int, help="number of changes to make in the gibbs chain", default=50)
parser.add_argument("--n_samples", type=int, help="number of changes to make in the gibbs chain", default=4)
parser.add_argument("--batch_size", type=int, help="number of changes to make in the gibbs chain", default=4)

parser.add_argument("--temperature", type=float, help="number of changes to make in the gibbs chain", default=1.0)
parser.add_argument("--degenerate", action='store_true')
parser.add_argument("--block", action='store_true')
parser.add_argument("--shuffle_positions", action='store_false')

parser.add_argument("--top_k", type=int, help="top_k sampler-so far only degenerate support", default=40)
parser.add_argument("--burnin", type=int, help="burn in for degenerate support", default=250)


parser.add_argument("--text_path_a", type=str, help="dir", default="./inputs/text_a.txt")
parser.add_argument("--topic_path_a", type=str, help="dir", default="./inputs/topic_a.txt")
parser.add_argument("--eq_path_a", type=str, help="dir", default="./inputs/eq_a.txt")
parser.add_argument("--text_path_b", type=str, help="dir", default="./inputs/text_b.txt")
parser.add_argument("--topic_path_b", type=str, help="dir", default="./inputs/topic_b.txt")
parser.add_argument("--eq_path_b", type=str, help="dir", default="./inputs/eq_b.txt")
parser.add_argument("--text_path_da", type=str, help="dir", default="./inputs/text_da.txt")
parser.add_argument("--topic_path_da", type=str, help="dir", default="./inputs/topic_da.txt")
parser.add_argument("--eq_path_da", type=str, help="dir", default="./inputs/eq_da.txt")
parser.add_argument("--text_path_db", type=str, help="dir", default="./inputs/text_db.txt")
parser.add_argument("--topic_path_db", type=str, help="dir", default="./inputs/topic_db.txt")
parser.add_argument("--eq_path_db", type=str, help="dir", default="./inputs/eq_db.txt")
parser.add_argument("--aoa_path", type=str, help="dir", default="./inputs/AoAKuperman.csv")
parser.add_argument("--out_path", type=str, help="dir", default="./outputs_gen")


parser.add_argument("--model_path", type=str, help="dir", default=mlm_model_path)
parser.add_argument("--tok_path", type=str, help="dir", default=mlm_model_path)
#disc
parser.add_argument("--topic_dir", type=str, help="topic disc dir", default=topic_model_path)
#eq
parser.add_argument("--eq_dir", type=str, help="equation generator dir", default=eq_model_path)


#hyper params
parser.add_argument("--alpha", type=float, help="topic weight", default=1000)  #topic disc
parser.add_argument("--beta", type=float, help="equation weight", default=100)   #equation
parser.add_argument("--gamma", type=float, help="mlm weight", default=1)   #mlm
parser.add_argument("--delta", type=float, help="hamming weight", default=100)  #hamming
parser.add_argument("--theta", type=float, help="bert score weight", default=2000) #bertscore
parser.add_argument("--phi", type=float, help="bleurt score weight", default=1000) #bleurt score: not used


args = parser.parse_args()

##################

#Age of acquisition
aoa_dict = {}
file = open(args.aoa_path)
csvreader = csv.reader(file)
header = []
header = next(csvreader)
for row in csvreader:
  aoa_dict[row[0]] = row[10]

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else 'cpu'

model_version = args.model_path
model = BertForMaskedLM.from_pretrained(model_version, return_dict = True)
model.eval()

topic_model = BertForSequenceClassification.from_pretrained(args.topic_dir)
topic_model.eval()

eq_model = T5ForConditionalGeneration.from_pretrained(args.eq_dir)
eq_model.eval()

bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
bleurt_model.eval()

if cuda:
    model = model.cuda()
    topic_model = topic_model.cuda()
    eq_model = eq_model.cuda()
    bleurt_model = bleurt_model.cuda()

if args.seed_setup==2:
    bert_scorer = bert_score.BERTScorer(use_fast_tokenizer=True, lang='en')

tokenizer = BertTokenizer.from_pretrained(args.tok_path)
topic_tokenizer = BertTokenizer.from_pretrained(args.topic_dir)
eq_tokenizer = T5Tokenizer.from_pretrained(args.eq_dir)
bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")


def get_opt_sent(sents, metadata):
    min_score = 10000
    ind = 0
    meta_array = np.array(metadata)

    ind = np.argmin(meta_array[:, 1, ...])
    #val = np.min(meta_array[:, 1, ...])
    sent_best = sents[ind].split()
    return " ".join(sent_best[1:-1]), meta_array[ind][-3], ind


def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(list(sent.to('cpu').numpy())) for sent in batch]


def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent


CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
PAD = "[PAD]"
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
pad_id = tokenizer.convert_tokens_to_ids([PAD])[0]


def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """Generate a word from out[gen_idx]
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [[CLS]+seed_text+[SEP] for _ in range(batch_size)]

    return tokenize_batch(batch)


def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


def to_file(sents, file):
    with open(file, "a") as f:
        f.write("\n".join(sents) + "\n")


def get_bert_score(batch, seed_text):
    sents = untokenize_batch(batch)

    sents = [(" ".join(item[1:-1])).strip() for item in sents]

    P, R, F1 = bert_scorer.score(sents, seed_text, verbose=False, batch_size=args.batch_size)

    return np.array(F1)


def energy_score_mlm(ids, gamma=1):
    seq_len = len(ids[1])-2
    posns = [i+1 for i in range(seq_len)]
    norm_score = [0.0] * ids.shape[0]
    raw_score = [0.0] * ids.shape[0]
    for posn in posns:
        old_wrd = ids[:,posn].clone()
        ids[:,posn] = mask_id
        output = model(ids)['logits'][:,posn,:]
        norm_output = output.log_softmax(dim=-1)
        for i in range(ids.shape[0]):
            raw_score[i] += output[i,old_wrd[i]].item()
            norm_score[i] += norm_output[i,old_wrd[i]].item()
        ids[:,posn] = old_wrd
    #normalize by length
    raw_score[:] = [raw_s/seq_len for raw_s in raw_score]
    norm_score[:] = [norm_s/seq_len for norm_s in norm_score]
    return [-1.0*raw_s*gamma for raw_s in raw_score], [-1.0*norm_s*gamma for norm_s in norm_score]


def energy_score_disc(input_ids, input_mask=None, topic=0, alpha=1):

    norm_score = np.array([0.0] * input_ids.shape[0])
    raw_score = np.array([0.0] * input_ids.shape[0])

    output = topic_model(input_ids, token_type_ids=None, attention_mask=input_mask)['logits']
    pred = np.argmax(np.array(output.log_softmax(dim=-1).cpu().detach()), axis=-1)

    classes = output.shape[-1]
    for i in range(classes):
        if i == topic:
            raw_score += np.array(output[:, i].cpu().detach())
            norm_output = output.log_softmax(dim=-1)
            norm_score += np.array(norm_output[:, i].cpu().detach())

    return [-1.0 * raw_s * alpha for raw_s in raw_score], [-1.0 * norm_s * alpha for norm_s in norm_score], pred


def energy_score_eq(ids, mask=None, eq_gt="", beta=1):
    raw_score = [0.0] * ids.shape[0]

    generated_ids = eq_model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    preds = [eq_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    # currently use Levenshtein distance
    for i in range(len(preds)):
        raw_score[i] += lev(preds[i], eq_gt)

    return [1.0 * raw_s * beta for raw_s in raw_score], preds


def parallel_sequential_generation(
        seed_text,
        topic=0,
        eq="",
        batch_size=10,
        max_len=200,
        top_k=0,
        temperature=1,
        max_iter=300,
        burnin=200,
        cuda=False,
        print_every=1,
        verbose=True,
        args=args
):
    """Generate for one random position at a timestep
    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    if args.no_padding:
        max_len = len(tokenizer.encode(seed_text))
    batch = torch.tensor(
        tokenizer(seed_text, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")[
            "input_ids"].tolist() * batch_size).to(device)


    batch_original = batch.detach().clone()

    seq_len = batch.shape[-1] - 2
    posns = [i + 1 for i in range(seq_len)]

    full_meta_data = [[] for i in range(batch_size)]
    hyper_scale = [[] for i in range(batch_size)]
    meta_data = []

    for ii in range(max_iter):
        print("current iter no:"+str(ii))
        st_iter = time.time()
        iter_no = ii

        if (args.shuffle_positions):
            random.shuffle(posns)
        if not (args.block):
            nmasks = 1
        else:
            nmasks = random.randint(max(1, math.ceil(seq_len / 2) - 3), min(seq_len - 1, math.ceil(seq_len / 2) + 3))

        groups = [posns[i:i + nmasks] for i in range(0, len(posns), nmasks)]
        if (args.shuffle_positions):
            random.shuffle(groups)
            # kk = mask_pos[np.random.randint(0, len(mask_pos))]
        #print(groups)

        for positions in groups:
            st_pos = time.time()
            print(positions)
            if args.degenerate:
                old_r, old_norm = np.array(energy_score_mlm(batch))
                old_wrd = batch[:, positions].detach().clone()

                batch[:, positions] = mask_id

                output = (model(batch)[:, positions, :] / temperature)
                output[:, :, mask_id] = -10000000000.0
                output = output.softmax(dim=-1)

                qxbx = np.array([1.0] * batch_size)
                qxxb = np.array([1.0] * batch_size)

                d = Categorical(output)
                new_wrd = d.sample()
                n_flag = np.array([0] * batch_size)
                msk_change = [False] * batch_size

                for ii in range(len(positions)):
                    for jj in range(batch_size):
                        qxxb[jj] *= output[jj, ii, old_wrd[jj, ii]].cpu()
                        qxbx[jj] *= output[jj, ii, new_wrd[jj, ii]].cpu()
                        if not (old_wrd[jj, ii].item() == new_wrd[jj, ii].item()):
                            n_flag[jj] = 1
                        if (old_wrd[jj, ii].item() == mask_id):
                            msk_change[jj] = True

                batch[:, positions] = new_wrd
                new_r, new_norm = np.array(energy_score_mlm(batch))

                # mask_id == np.array(old_wrd.cpu())

                # if the old word = [mask], always accept new; else, accept with a rate
                axbx = np.where(msk_change, 1.0, np.minimum(1.0, np.divide(
                    np.multiply(np.exp(old_r - new_r), np.array(qxxb)), np.array(qxbx))))


                acc = torch.ones(axbx.shape)  # torch.squeeze(torch.bernoulli(torch.Tensor([axbx])))

                batch[:, positions] = torch.where(acc.unsqueeze(1).repeat(1, len(positions)).to(device) > 0.0,
                                                  batch[:, positions], old_wrd)

                r_score = np.squeeze(np.where(acc > 0.0, new_r, old_r))
                norm_score = np.squeeze(np.where(acc > 0.0, new_norm, old_norm))

                acc = np.array(acc.cpu()) * np.array(n_flag)


            else:

                seed_old = [tokenizer.decode(b, skip_special_tokens=True, clean_up_tokenization_spaces=True) for b in
                            batch]

                batch_disc = topic_tokenizer(seed_old, padding="max_length", max_length=max_len, truncation=True,
                                            return_tensors="pt")
                input_ids = batch_disc["input_ids"].to(device)
                input_mask = batch_disc["attention_mask"].to(device)
                st_topic_o = time.time()
                disc_1, disc_2, disc_preds = energy_score_disc(input_ids, input_mask, topic=topic, alpha=args.alpha)
                et_topic_o = time.time()
                #print("Time for topic old:" + str(et_topic_o - st_topic_o))


                batch_eq = eq_tokenizer(seed_old, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
                ids = batch_eq["input_ids"].to(device, dtype=torch.long)
                mask = batch_eq["attention_mask"].to(device, dtype=torch.long)
                st_eq_o = time.time()
                eq_1, eq_preds = energy_score_eq(ids, mask, eq_gt=eq, beta=args.beta)
                et_eq_o = time.time()
                #print("Time for eq old:" + str(et_eq_o - st_eq_o))

                seed_1 = [tokenizer.decode(b, skip_special_tokens=False, clean_up_tokenization_spaces=True) for b in batch_original]
                seed_2 = [tokenizer.decode(b, skip_special_tokens=False, clean_up_tokenization_spaces=True) for b in batch]
                bleurt_score = np.array(bleurt_model(
                    **bleurt_tokenizer(seed_2, seed_2, padding="max_length", max_length=max_len, truncation=True,
                                       return_tensors='pt').to(device))[0].squeeze().detach().cpu())

                distance = 0
                bert_score = 0
                st_dist_o = time.time()
                if args.seed_setup==2:
                    distance = np.sum(1 - np.array((batch == batch_original).detach().cpu()) * 1, axis=-1)
                    distance = distance / (len(batch[1]) - 2)
                    seed_text_broad = [seed_text for i in range(batch.shape[0])]
                    bert_score = get_bert_score(batch, seed_text_broad)
                et_dist_o = time.time()
                #print("Time for dist old:" + str(et_dist_o - st_dist_o))

                st_mlm_o = time.time()
                mlm_1, mlm_2 = energy_score_mlm(batch, gamma=args.gamma)
                et_mlm_o = time.time()
                #print("Time for mlm old:" + str(et_mlm_o - st_mlm_o))

                old_r, old_norm = np.array([mlm_1, mlm_2]) + np.array([disc_1, disc_2])
                old_r += eq_1
                old_r -= args.phi * bleurt_score
                if args.seed_setup == 2:
                    old_r -= args.delta * distance
                    old_r += args.theta * bert_score
                for i in range(batch_size):
                    if args.seed_setup==2:
                        hyper_scale[i].append((np.array(mlm_1)[i], np.array(mlm_2)[i], np.array(disc_1)[i], np.array(disc_2)[i], eq_1[i], bleurt_score[i], distance[i], bert_score[i]))
                    else:
                        hyper_scale[i].append((np.array(mlm_1)[i], np.array(mlm_2)[i], np.array(disc_1)[i], np.array(disc_2)[i], eq_1[i], bleurt_score[i]))


                old_wrd = batch[:, positions].detach().clone()

                metrics = {"iter": iter_no,
                           "position": positions[0],
                }


                for i in range(batch_size):
                    metrics["mlm_r_old_"+str(i)] = np.array(mlm_1)[i]
                    metrics["topic_r_old_" + str(i)] = np.array(disc_1)[i]
                    metrics["eq_r_old_" + str(i)] = np.array(eq_1)[i]
                    metrics["bleurt_r_old_" + str(i)] = -args.phi * bleurt_score[i]
                    metrics["r_old" + str(i)]=old_r[i]
                    if args.seed_setup==2:
                        metrics["ham_r_old_" + str(i)] = -args.delta * distance[i]
                        metrics["bert_r_old_" + str(i)] = args.theta * bert_score[i]

                st_newtoken = time.time()
                batch[:, positions] = mask_id
                output = (model(batch)['logits'][:, positions, :] / temperature)
                output[:, :, mask_id] = -10000000000.0
                output = output.softmax(dim=-1)

                qxbx = np.array([1.0] * batch_size)
                qxxb = np.array([1.0] * batch_size)

                d = Categorical(output)
                new_wrd = d.sample()
                new_token = [tokenizer.convert_ids_to_tokens(id)[0] for id in new_wrd.tolist()]
                et_newtoken = time.time()
                #print("Time for get new token:" + str(et_newtoken - st_newtoken))
                #print(new_token)

                aoa_val = []
                for tok in new_token:
                    if tok in aoa_dict.keys():
                        if float(aoa_dict[tok]) > 12:
                            aoa_val.append(0)
                        else:
                            aoa_val.append(1)
                    else:
                        aoa_val.append(0)
                aoa_val = torch.tensor(aoa_val)

                n_flag = np.array([0] * batch_size)
                msk_change = [False] * batch_size

                for ii in range(len(positions)):
                    for jj in range(batch_size):
                        qxxb[jj] *= output[jj, ii, old_wrd[jj, ii]].cpu()
                        qxbx[jj] *= output[jj, ii, new_wrd[jj, ii]].cpu()
                        if not (old_wrd[jj, ii].item() == new_wrd[jj, ii].item()):
                            n_flag[jj] = 1
                        if (old_wrd[jj, ii].item() == mask_id):
                            msk_change[jj] = True

                batch[:, positions] = new_wrd

                seed_new = [tokenizer.decode(b, skip_special_tokens=True, clean_up_tokenization_spaces=True) for b in
                            batch]

                batch_disc = topic_tokenizer(seed_new, padding="max_length", max_length=max_len, truncation=True,
                                            return_tensors="pt")
                input_ids = batch_disc["input_ids"].to(device)
                input_mask = batch_disc["attention_mask"].to(device)
                st_topic_n = time.time()
                disc_1, disc_2, disc_preds_new = energy_score_disc(input_ids, input_mask, topic=topic, alpha=args.alpha)
                et_topic_n = time.time()
                #print("Time for topic new:" + str(et_topic_n - st_topic_n))


                batch_eq = eq_tokenizer(seed_new, padding="max_length", max_length=max_len, truncation=True,
                                        return_tensors="pt")
                ids = batch_eq["input_ids"].to(device, dtype=torch.long)
                mask = batch_eq["attention_mask"].to(device, dtype=torch.long)
                st_eq_n = time.time()
                eq_1, eq_preds_new = energy_score_eq(ids, mask, eq_gt=eq, beta=args.beta)
                et_eq_n = time.time()
                #print("Time for eq new:" + str(et_eq_n - st_eq_n))


                seed_2 = [tokenizer.decode(b, skip_special_tokens=False, clean_up_tokenization_spaces=True) for b in
                          batch]
                bleurt_new = np.array(
                    bleurt_model(
                        **bleurt_tokenizer(seed_2, seed_2, padding="max_length", max_length=max_len, truncation=True,
                                           return_tensors='pt').to(device))[0].squeeze().detach().cpu())


                distance_new=0
                bert_new = 0
                st_dist_n = time.time()
                if args.seed_setup==2:
                    distance_new = np.sum(1 - np.array((batch == batch_original).detach().cpu()) * 1, axis=-1)
                    distance_new = distance_new / (len(batch[1]) - 2)
                    seed_text_broad = [seed_text for i in range(batch.shape[0])]
                    bert_new = get_bert_score(batch, seed_text_broad)
                et_dist_n = time.time()
                #print("Time for dist new:" + str(et_dist_n - st_dist_n))

                st_mlm_n = time.time()
                mlm_1, mlm_2 = energy_score_mlm(batch, gamma=args.gamma)
                et_mlm_n = time.time()
                #print("Time for mlm new:" + str(et_mlm_n - st_mlm_n))

                new_r, new_norm = np.array([mlm_1, mlm_2]) + np.array([disc_1, disc_2])
                new_r += eq_1
                new_r -= args.phi * bleurt_new
                if args.seed_setup == 2:
                    new_r -= args.delta * distance_new
                    new_r += args.theta * bert_new
                for i in range(batch_size):
                    if args.seed_setup==2:
                        hyper_scale[i].append((np.array(mlm_1)[i], np.array(mlm_2)[i], np.array(disc_1)[i], np.array(disc_2)[i], eq_1[i], bleurt_new[i], distance_new[i], bert_new[i]))
                    else:
                        hyper_scale[i].append((np.array(mlm_1)[i], np.array(mlm_2)[i], np.array(disc_1)[i], np.array(disc_2)[i], eq_1[i], bleurt_new[i]))


                # mask_id == np.array(old_wrd.cpu())

                print("MH")
                print(old_r, new_r, np.exp(old_r-new_r))
                print(np.array(qxxb), np.array(qxbx), np.divide(np.array(qxxb), np.array(qxbx)))
                print(np.divide(np.multiply(np.exp(old_r - new_r), np.array(qxxb)), np.array(qxbx)))

                axbx = np.where(msk_change, 1.0, np.minimum(1.0, np.divide(
                    np.multiply(np.exp(old_r - new_r), np.array(qxxb)), np.array(qxbx))))

                print(axbx)

                acc = torch.squeeze(torch.bernoulli(torch.Tensor([axbx])))

                print(acc)

                if args.difficulty == "basic":
                    acc = acc.mul(aoa_val)
                    print(aoa_val)
                    print(acc)

                print("MH DONE")

                batch[:, positions] = torch.where(acc.unsqueeze(1).repeat(1, len(positions)).to(device) > 0.0,
                                                  batch[:, positions], old_wrd)

                r_score = np.squeeze(np.where(acc > 0.0, new_r, old_r))
                norm_score = np.squeeze(np.where(acc > 0.0, new_norm, old_norm))
                disc_preds = np.squeeze(np.where(acc > 0.0, disc_preds_new, disc_preds))
                eq_preds = np.squeeze(np.where(acc > 0.0, eq_preds_new, eq_preds))
                bleurt_score = np.squeeze(np.where(acc > 0.0, bleurt_new, bleurt_score))
                if args.seed_setup == 2:
                    distance = np.squeeze(np.where(acc > 0.0, distance_new, distance))
                    bert_score = np.squeeze(np.where(acc > 0.0, bert_new, bert_score))


                for i in range(batch_size):
                    metrics["bleurt_r_keep_" + str(i)] = -args.phi * bleurt_score[i]
                    metrics["r_keep" + str(i)] = r_score[i]
                    if args.seed_setup == 2:
                        metrics["ham_r_keep_" + str(i)] = -args.delta*distance[i]
                        metrics["bert_r_keep_" + str(i)] = args.theta*bert_score[i]


                acc = np.array(acc.cpu()) * np.array(n_flag)
                print(acc)



                for i in range(batch_size):
                    if args.seed_setup == 2:
                        full_meta_data[i].append((topic, eq, r_score[i], norm_score[i], qxxb[i], qxbx[i], axbx[i],
                        acc[i].item(), disc_preds[i], eq_preds[i], distance[i], bert_score[i]))
                    else:
                        full_meta_data[i].append((topic, eq, r_score[i], norm_score[i], qxxb[i], qxbx[i], axbx[i],
                                           acc[i].item(), disc_preds[i], eq_preds[i]))

                wandb.log(metrics)

                et_pos = time.time()
                print("Time for position" + str(positions[0]) + ":" + str(et_pos - st_pos))


        if np.mod(iter_no + 1, 10) == 0:
            topic_pred_table = wandb.Table(columns=["iter", "topic_pred"])
            eq_pred_table = wandb.Table(columns=["iter", "eq_pred"])
            cur_sen_table = wandb.Table(columns=["iter", "cur_sen"])
            cur_sen = [" ".join(b) for b in untokenize_batch(batch)]
            for i in range(batch_size):
                topic_pred_table.add_data(iter_no, disc_preds[i])
                eq_pred_table.add_data(iter_no, eq_preds[i])
                cur_sen_table.add_data(iter_no, cur_sen[i])
            wandb.log({"topic_pred":topic_pred_table})
            wandb.log({"eq_pred": eq_pred_table})
            wandb.log({"current_sen": cur_sen_table})


        if np.mod(iter_no + 1, print_every) == 0:
            print("iter", iter_no+1)
            for b in batch:
                print(" ".join(tokenizer.convert_ids_to_tokens(b)))
            print(disc_preds)
            print(eq_preds)

        et_iter = time.time()
        print("Time for iteration" + str(iter_no) + ":" + str(et_iter - st_iter))

    for i in range(batch_size):
        if args.seed_setup == 2:
            meta_data.append(
              (topic, eq, r_score[i], norm_score[i], qxxb[i], qxbx[i], axbx[i], acc[i].item(), disc_preds[i],
               eq_preds[i], distance[i], bert_score[i]))
        else:
            meta_data.append(
              (topic, eq, r_score[i], norm_score[i], qxxb[i], qxbx[i], axbx[i], acc[i].item(), disc_preds[i],
               eq_preds[i]))

    return untokenize_batch(batch), meta_data, full_meta_data, hyper_scale


def generate(
    n_samples,
    topic,
    template,
    seed_text,
    batch_size,
    max_len,
    top_k,
    temperature,
    burnin,
    max_iter,
    cuda,
    print_every=10,
    args=args
):
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        batch , metadata, full_metadata, hyper_scale= parallel_sequential_generation(
            seed_text,
            batch_size=batch_size,
            topic=topic,
            eq = template,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            verbose=False
        )

        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()

        sentences += batch
    return sentences, metadata, full_metadata, hyper_scale


degenerate = args.degenerate
top_k = args.top_k  # 40 #not used
burnin = args.burnin  # 250 #not used
temperature = args.temperature
###########

dirname = args.out_path
n_samples = args.n_samples
batch_size = args.batch_size
max_iter = args.max_iter

max_len = args.max_len
########

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

if args.degenerate:
    folder_name = "degenerate_topk_{}_burnin_{}_disc_{}_data_{}_max_iter_{}_temp_{}_shuffle_{}_block_{}_alpha_{}_beta_{}_delta_{}_gamma_{}_theta_{}_date_{}".format(
        top_k, burnin, args.disc_name, args.data_name, max_iter, temperature, args.shuffle_positions, args.block,
        args.alpha, args.beta, args.delta, args.gamma, args.theta, dt_string)
else:
    folder_name = "{}_{}_max_iter_{}_temp_{}_shuffle_{}_block_{}_alpha_{}_beta_{}_gamma_{}_delta_{}_theta_{}_phi_{}_date_{}".format(
        args.difficulty, args.seed_setup, max_iter, temperature, args.shuffle_positions, args.block, args.alpha, args.beta, args.gamma,args.delta,args.theta,args.phi, dt_string)

directory = "{}/{}".format(dirname, folder_name)
if not os.path.exists(directory):
    os.mkdir(directory)

dirname = directory

topics = ["money-a", "physics-a", "daily-a", "geometry", "probability", "money-b", "physics-b", "daily-b"]


if args.seed_setup==2:
    if args.difficulty == "advanced":
        text_in = args.text_path_da
        topic_in = args.topic_path_da
        eq_in = args.eq_path_da
    elif args.difficulty == "basic":
        text_in = args.text_path_db
        topic_in = args.topic_path_db
        eq_in = args.eq_path_db
    else:
        print("Not right seed text")
else:
    if args.difficulty == "advanced":
        text_in = args.text_path_a
        topic_in = args.topic_path_a
        eq_in = args.eq_path_a
    elif args.difficulty == "basic":
        text_in = args.text_path_b
        topic_in = args.topic_path_b
        eq_in = args.eq_path_b
    else:
        print("Not right seed text")


with open(f"{dirname}/samples.txt", "w") as f, open(f"{dirname}/opt_samples.txt", "w") as optimal_f, open(
        f"{dirname}/opt_cls.txt", "w") as optimal_class, open(f"{dirname}/opt_meta.txt", "w") as opt_meta_file, open(
    f"{dirname}/metadata.txt", "w") as f_meta, open(f"{dirname}/hyperscale.txt", "w") as f_hyper, open(f"{text_in}", "r") as data_file, open(f"{topic_in}","r") as topic_file, open(f"{eq_in}", "r") as eq_file:
    for i, (text, topic, eq) in enumerate(zip(data_file, topic_file, eq_file)):

        print("current sentence no:"+str(i))

        print(text)
        print(topic)
        print(eq)

        wandb.init(
            project="Thesis",
            name=f"sample_{i}",
            config={
                "difficulty": args.difficulty,
                "seed": args.seed_setup,
                "mlm_size": args.mlm_size,
                "no_padding": args.no_padding,
                "max_len": args.max_len,
                "max_iter": args.max_iter,
                "topic_alpha": args.alpha,
                "equation_beat": args.beta,
                "mlm_gamma": args.gamma,
                "ham_delta": args.delta,
                "bert_theta": args.theta,
                "bleurt_phi": args.phi,
            }
        )


        seed_text = text.split("\n")[0]
        topic = topics.index(topic.split("\n")[0])
        eq = eq.split("\n")[0]
        torch.cuda.empty_cache()
        bert_sents, meta_data, full_meta_data, hyper_scale = generate(
            n_samples,
            topic=topic,
            template=eq,
            seed_text=seed_text,
            batch_size=batch_size,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            args=args
        )

        sents = list(map(lambda x: " ".join(detokenize(x)), bert_sents))

        print("write to files!")

        f.write("\n".join(sents) + "\n")
        f.flush()

        # meta_data_str = [str(l) for l in meta_data]

        # f_meta.write("\n".join(meta_data_str)+"\n")
        # f_meta.flush()

        full_meta_data_str = [str(l) for l in full_meta_data]
        f_meta.write("\n".join(full_meta_data_str) + "\n")
        f_meta.flush()

        hyper_scale_str = [str(l) for l in hyper_scale]
        f_hyper.write("\n".join(hyper_scale_str) + "\n")
        f_hyper.flush()


        opt_sent, opt_cls, ind = get_opt_sent(sents, meta_data)
        optimal_f.write(opt_sent + "\n")
        optimal_f.flush()

        opt_meta_str = str(full_meta_data[ind])
        opt_meta_file.write(opt_meta_str + "\n")
        opt_meta_file.flush()

        optimal_class.write(str(opt_cls) + "\n")
        optimal_class.flush()


        wandb.finish()
