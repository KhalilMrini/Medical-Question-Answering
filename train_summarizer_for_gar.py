from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

PATH = '../'

class CustomTextDataset(Dataset):
    def __init__(self, dataset_name, mode, tokenizer):
        self.src = [line.strip() for line in 
                    open(PATH + dataset_name + '/' + mode + '.source').readlines()]
        self.src_encodings = tokenizer(self.src, truncation=True, padding=True)
        self.tgt = [line.strip() for line in 
                    open(PATH + dataset_name + '/' + mode + '.target').readlines()]
        self.tgt_encodings = tokenizer(self.tgt, truncation=True, padding=True)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.src_encodings.items()}
        item['labels'] = self.tgt_encodings['input_ids'][idx]
        return item

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

training_args = Seq2SeqTrainingArguments(
    output_dir="./results_hcm",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    save_steps=1000,
    fp16=True,
)

dataset = 'HealthCareMagic'

train_dataset = CustomTextDataset(dataset, 'train', tokenizer)
eval_dataset = CustomTextDataset(dataset, 'dev', tokenizer)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
