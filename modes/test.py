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
from scipy.sparse import csr_matrix

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

def run_test(hparams):
    #_, labels, _, _ = get_data()

    #main()

    database = pickle.load(open('meqsum_database.pkl', 'rb'))
    questions_as_tfidf = pickle.load(open('meqsum_questions_as_tfidf.pkl', 'rb'))
    questions_as_str = pickle.load(open('meqsum_questions_as_str.pkl', 'rb'))
    tfidf_vocab = pickle.load(open('meqsum_tfidf_vocab.pkl', 'rb'))
    tfidf_df = pickle.load(open('meqsum_tfidf_df.pkl', 'rb'))

    chqs = pickle.load(open('meqsum_chqs.pkl', 'rb'))
    faqs = pickle.load(open('meqsum_faqs.pkl', 'rb'))

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
    loaded_model = torch.load('/vault/datasets/khalil/EndToEndQA/models/meqsum_no_match_loss_endtoendqa_best_loss=12.10.pt')
    #loaded_model = torch.load('/vault/datasets/khalil/EndToEndQA/models/meqsum_coeff_0.01_endtoendqa_best_loss=13.14.pt')
    model.load_state_dict(loaded_model['state_dict'])
    model.cuda()

    print('parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))


    print("Training...")
    model_name = hparams.model_name

    print("This is ", model_name)
    start_time = time.time()

    dev_start_time = time.time()

    test_chqs = chqs['valid']
    test_faqs = faqs['valid']
    num_test = len(test_chqs)

    model.eval()

    total_loss_value = 0
    total_summarization_loss = 0
    total_match_faq_nll_loss = 0
    total_as2_unsupervised_loss = 0

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    faq_rouge1 = 0
    faq_rouge2 = 0
    faq_rougel = 0
    match_rouge1 = 0
    match_rouge2 = 0
    match_rougel = 0
    
    match_bert_score = 0

    output_file = open('meqsum_ref_as_gen.txt', 'w')
    output_file2 = open('meqsum_gen.txt', 'w')

    ref_file = open('meqsum_ref_for_annotation.tsv', 'w')
    match_file = open('meqsum_mat_for_annotation.tsv', 'w')
    matched_questions = set()
    print('question_index\tquestion\tanswer_sentence\tannotation', file=ref_file)
    print('question_index\tquestion\tanswer_sentence\tannotation', file=match_file)

    SELECTED_IDX = set([0,1,2,3,4,5,6,7,9,13,18,19,21,26,27,32,36,43,44,45,47,49])
    precisions = []
    recalls = []
    f1s = []
    accuracy = []

    for idx in tqdm(range(num_test)):
        model_outputs = model(test_chqs[idx], faq=test_faqs[idx], test=True, ref_as_gen=True, test_idx=idx) # replacing generated FAQ by reference FAQ
        print('======', str(idx), '======', file=output_file)
        print('CHQ:', test_chqs[idx], file=output_file)
        print('Generated FAQ:', model_outputs[6], file=output_file)
        print('Reference FAQ:', test_faqs[idx], file=output_file)
        print('Matched FAQ:', model_outputs[5], file=output_file)
        print('Questions:', model_outputs[8], file=output_file)
        print('Tokens:', model_outputs[9], file=output_file)
        print('Selected Answers:', ' '.join(model_outputs[2]), '\n', file=output_file)
        if idx in SELECTED_IDX:
            if test_faqs[idx] not in matched_questions:
                for answer_sentence in model_outputs[10]:
                    if test_faqs[idx] not in answer_sentence:
                        print('{}\t{}\t{}\t'.format(str(idx), test_faqs[idx].replace('\n',' ').replace('\t', ' '), answer_sentence.replace('\n',' ').replace('\t', ' ')), file=ref_file)
                        print('{}\t{}\t{}\t'.format(str(idx), model_outputs[5].replace('\n',' ').replace('\t', ' '), answer_sentence.replace('\n',' ').replace('\t', ' ')), file=match_file)
            matched_questions.add(test_faqs[idx])
        match_true = model_outputs[5]
        ranking_true = model_outputs[11]
        y_true = np.zeros(len(ranking_true))
        y_true[ranking_true[:5]] = 1
        del model_outputs
        model_outputs = model(test_chqs[idx], faq=test_faqs[idx], test=True, test_idx=idx) # NOT replacing generated FAQ by reference FAQ
        print('======', str(idx), '======', file=output_file2)
        print('CHQ:', test_chqs[idx], file=output_file2)
        print('Generated FAQ:', model_outputs[6], file=output_file2)
        print('Reference FAQ:', test_faqs[idx], file=output_file2)
        print('Matched FAQ:', model_outputs[5], file=output_file2)
        print('Selected Answers:', ' '.join(model_outputs[2]), '\n', file=output_file2)
        match_bert_score += float(model_outputs[7].data.cpu())
        match_pred = model_outputs[5]
        accuracy.append(int(match_true == match_pred))
        ranking_pred = model_outputs[11]
        y_pred = np.zeros(len(ranking_pred))
        y_pred[ranking_pred[:5]] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        faq_rouge = scorer.score(test_faqs[idx], model_outputs[6])
        faq_rouge1 += faq_rouge['rouge1'].fmeasure
        faq_rouge2 += faq_rouge['rouge2'].fmeasure
        faq_rougel += faq_rouge['rougeL'].fmeasure
        match_rouge = scorer.score(model_outputs[5], model_outputs[6])
        match_rouge1 += match_rouge['rouge1'].fmeasure
        match_rouge2 += match_rouge['rouge2'].fmeasure
        match_rougel += match_rouge['rougeL'].fmeasure
        total_summarization_loss += float(model_outputs[0][0].data.cpu().numpy())
        total_match_faq_nll_loss += hparams.match_nll_coeff * float(model_outputs[1][0].data.cpu().numpy())
        total_as2_unsupervised_loss += hparams.as2_loss_coeff * (float(model_outputs[3].data.cpu().numpy()) + float(model_outputs[4].data.cpu().numpy()))
        del model_outputs[4]
        del model_outputs[3]
        del model_outputs[1]
        del model_outputs[0]
        del model_outputs
        torch.cuda.empty_cache()

    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)
    accuracy = np.mean(accuracy)

    ref_file.close()
    match_file.close()

    faq_rouge1 /= num_test
    faq_rouge2 /= num_test
    faq_rougel /= num_test
    match_rouge1 /= num_test
    match_rouge2 /= num_test
    match_rougel /= num_test
    match_bert_score /= num_test

    output_file.close()

    total_loss_value = total_summarization_loss + total_match_faq_nll_loss + total_as2_unsupervised_loss

    print(
        "loss {} summarization-loss {} match-faq-nll-loss {} as2-unsupervised-loss {}"
        "dev-elapsed {} "
        "total-elapsed {}\nFAQ ROUGE scores: R1 {} R2 {} RL {}\nMatch ROUGE scores: R1 {} R2 {} RL {}\nMatch BERT Score: {} Match Precision: {} Match Recall: {} Match F1 Score: {} Match FAQ Accuracy: {}".format(
            total_loss_value/num_test, total_summarization_loss/num_test, total_match_faq_nll_loss/num_test, total_as2_unsupervised_loss/num_test,
            format_elapsed(dev_start_time),
            format_elapsed(start_time), faq_rouge1, faq_rouge2, faq_rougel, match_rouge1, match_rouge2, match_rougel, match_bert_score,
            precision, recall, f1, accuracy
        ), file=output_file2
    )
    output_file2.close()

