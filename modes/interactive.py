import torch
import os
import numpy as np
from torch import from_numpy
import math
import sqlite3
import json
import pickle
import itertools
import time
from utils.utils import format_elapsed
from utils.hparams import make_hparams
from voli.end_to_end_qa import EndToEndQA
import argparse
from utils import constants
from utils.utils import get_data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from rouge_score import rouge_scorer

def check_mem():
    
    mem = os.popen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split("\n")[0].split(',')
    
    return mem

def main():
    
    total, used = check_mem()
    
    total = int(total)
    used = int(used)
        
    max_mem = int(total * 0.94)
    block_mem = max_mem - used
    print('Block mem', block_mem)
        
    x = torch.rand((256,1024,block_mem), device=torch.device('cuda:0'))
    del x

def run_interactive(hparams):
    #_, labels, _, _ = get_data()

    #main()

    DATASET = 'hcm'

    database = pickle.load(open(DATASET + '_database.pkl', 'rb'))
    questions_as_tfidf = pickle.load(open(DATASET + '_questions_as_tfidf.pkl', 'rb'))
    questions_as_str = pickle.load(open(DATASET + '_questions_as_str.pkl', 'rb'))
    tfidf_vocab = pickle.load(open(DATASET + '_tfidf_vocab.pkl', 'rb'))
    tfidf_df = pickle.load(open(DATASET + '_tfidf_df.pkl', 'rb'))

    torch.autograd.set_detect_anomaly(True)

    if hparams.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(hparams.numpy_seed))
        np.random.seed(hparams.numpy_seed)

    # Make sure that pytorch is actually being initialized randomly.
    # On my cluster I was getting highly correlated results from multiple
    # runs, but calling reset_parameters() changed that. A brief look at the
    # pytorch source code revealed that pytorch initializes its RNG by
    # calling std::random_device, which according to the C++ spec is allowed
    # to be deterministic.
    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)


    print("Initializing model...")

    model = EndToEndQA(database, questions_as_tfidf, questions_as_str, tfidf_vocab, tfidf_df)
    loaded_model = torch.load('./models/hcm_coeff_0.01_endtoendqa_best_loss=55.25.pt')
    #loaded_model = torch.load('./models/meqsum_coeff_0.01_endtoendqa_best_loss=13.14.pt')
    model.load_state_dict(loaded_model['state_dict'])
    model.cuda()

    print("Training...")
    model_name = hparams.model_name

    print("This is ", model_name)

    model.eval()
    
    match_bert_score = 0

    chq = ""
    idx = 0

    while not chq == 'STOP':
         # hparams.batch_size x len(article_sentences)
        idx += 1
        print('======', str(idx), '======')
        chq = input('CHQ or "STOP":')
        model_outputs = model(chq.strip())
        print('Generated FAQ:', model_outputs[6])
        print('Matched FAQ:', model_outputs[5])
        print('Selected Answers:', ' '.join(model_outputs[2]), '\n')
        match_bert_score += float(model_outputs[7].data.cpu())
        del model_outputs[4]
        del model_outputs[3]
        del model_outputs[1]
        del model_outputs[0]
        del model_outputs
        torch.cuda.empty_cache()

    match_bert_score /= idx

    print(
        "Match BERT Score: {}".format(
            match_bert_score
        )
    )

