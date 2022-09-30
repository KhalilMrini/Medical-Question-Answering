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

def check_mem():
    
    mem = os.popen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split("\n")[0].split(',')
    
    return mem

def main():
    
    total, used = check_mem()
    
    total = int(total)
    used = int(used)
        
    max_mem = int(total * 0.96)
    block_mem = max_mem - used
    print('Block mem', block_mem)
        
    x = torch.rand((256,1024,block_mem), device=torch.device('cuda:0'))
    del x

def run_train_hcm(hparams):
    #_, labels, _, _ = get_data()

    database = pickle.load(open('hcm_database.pkl', 'rb'))
    questions_as_tfidf = pickle.load(open('hcm_questions_as_tfidf.pkl', 'rb'))
    questions_as_str = pickle.load(open('hcm_questions_as_str.pkl', 'rb'))
    tfidf_vocab = pickle.load(open('hcm_tfidf_vocab.pkl', 'rb'))
    tfidf_df = pickle.load(open('hcm_tfidf_df.pkl', 'rb'))

    chqs = pickle.load(open('hcm_chqs.pkl', 'rb'))
    faqs = pickle.load(open('hcm_faqs.pkl', 'rb'))

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

    #main()

    print("Initializing model...")

    model = EndToEndQA(database, questions_as_tfidf, questions_as_str, tfidf_vocab, tfidf_df)
    model.cuda()

    print("Initializing optimizer...")

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    trainer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)

    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr

    warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, 'max',
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience,
        verbose=True,
    )
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= hparams.learning_rate_warmup_steps:
            set_lr(iteration * warmup_coeff)

    clippable_parameters = trainable_parameters
    grad_clip_threshold = np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm


    print("Training...")
    total_processed = 0
    current_processed = 0
    #check_every = hparams.epoch_steps / hparams.checks_per_epoch
    best_loss = np.inf
    best_model_path = None
    model_name = hparams.model_name

    print("This is ", model_name)
    start_time = time.time()

    

    def check_dev(epoch_num):
        nonlocal best_loss
        nonlocal best_model_path

        dev_start_time = time.time()

        valid_chqs = chqs['valid']
        valid_faqs = faqs['valid']
        num_valid = len(valid_chqs)

        model.eval()

        total_loss_value = 0
        total_summarization_loss = 0
        total_match_faq_nll_loss = 0
        total_as2_unsupervised_loss = 0

        for idx in tqdm(range(num_valid)):
            model_outputs = model(valid_chqs[idx], faq=valid_faqs[idx]) # hparams.batch_size x len(article_sentences)
            total_summarization_loss += float(model_outputs[0][0].data.cpu().numpy())
            total_match_faq_nll_loss += hparams.match_nll_coeff * float(model_outputs[1][0].data.cpu().numpy())
            total_as2_unsupervised_loss += hparams.as2_loss_coeff * (float(model_outputs[3].data.cpu().numpy()) + float(model_outputs[4].data.cpu().numpy()))
            del model_outputs[4]
            del model_outputs[3]
            del model_outputs[1]
            del model_outputs[0]
            del model_outputs
            torch.cuda.empty_cache()

        total_loss_value = total_summarization_loss + total_match_faq_nll_loss + total_as2_unsupervised_loss

        print(
            "loss {} summarization-loss {} match-faq-nll-loss {} as2-unsupervised-loss {}"
            "dev-elapsed {} "
            "total-elapsed {}".format(
                total_loss_value, total_summarization_loss, total_match_faq_nll_loss, total_as2_unsupervised_loss,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if total_loss_value < best_loss :
            if best_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_loss = total_loss_value
            best_model_path = "{}hcm_coeff_{}_endtoendqa_best_loss={:.2f}".format(
                hparams.model_path_base, hparams.match_nll_coeff, total_loss_value)
            print("Saving new best model to {}...".format(best_model_path))
            torch.save({
                'hparams': hparams,
                'state_dict': model.state_dict(),
                'trainer' : trainer.state_dict(),
                }, best_model_path + ".pt")

    train_chqs = chqs['train']
    train_faqs  = faqs['train']
    num_train = len(train_chqs)
    check_every = num_train
    train_idx = 0

    for epoch in itertools.count(start=1):
        if epoch > hparams.num_epochs:
            break
        #check_dev(epoch)
        epoch_start_time = time.time()
        epoch_idx = 0

        for step in range(0, num_train):#hparams.epoch_steps, hparams.batch_size):
            trainer.zero_grad()
            schedule_lr(total_processed // hparams.batch_size)

            model.train()

            model_outputs = model(train_chqs[step], faq=train_faqs[step]) # hparams.batch_size x len(article_sentences)
            sum_loss = model_outputs[0][0].data.cpu().numpy()
            match_loss = hparams.match_nll_coeff * model_outputs[1][0].data.cpu().numpy()
            as2_loss = hparams.as2_loss_coeff * model_outputs[3].data.cpu().numpy()
            as2_select_loss = hparams.as2_loss_coeff * model_outputs[4].data.cpu().numpy()
            loss = 0
            if sum_loss > 0:
                loss += model_outputs[0][0]
            if match_loss > 0:
                loss += hparams.match_nll_coeff * model_outputs[1][0]
            if as2_loss > 0:
                loss += hparams.as2_loss_coeff * model_outputs[3]
            if as2_select_loss > 0:
                loss += hparams.as2_loss_coeff * model_outputs[4]
            loss_value = float(loss.data.cpu().numpy())
            batch_loss_value = loss_value
            if loss_value > 0:
                loss.backward()
            del loss
            del model_outputs[4]
            del model_outputs[3]
            del model_outputs[1]
            del model_outputs[0]
            torch.cuda.empty_cache()
            train_idx = train_idx + hparams.batch_size
            epoch_idx = epoch_idx + hparams.batch_size
            if train_idx > num_train:
                train_idx = 0
            total_processed += hparams.batch_size
            current_processed += hparams.batch_size

            grad_norm = torch.nn.utils.clip_grad_norm_(clippable_parameters, grad_clip_threshold)

            trainer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "summarization-loss {:.4f} "
                "match-faq-nll-loss {:.4f} "
                "as2-unsupervised-loss {:.4f} "
                "as2-selection-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    step,#epoch_idx // hparams.batch_size,
                    num_train,#int(np.ceil(hparams.epoch_steps / hparams.batch_size)),
                    total_processed,
                    batch_loss_value,
                    sum_loss,
                    match_loss,
                    as2_loss,
                    as2_select_loss,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev(epoch)

        # adjust learning rate at the end of an epoch
        if hparams.step_decay:
            if (total_processed // hparams.batch_size + 1) > hparams.learning_rate_warmup_steps:
                scheduler.step(best_loss)