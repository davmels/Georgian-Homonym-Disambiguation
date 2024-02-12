import pandas as pd
import numpy as np
import datasets
from .functions import *

'''
    Text Classification task => TC
'''
def fine_tune_transformer_TC(targets = [0, 1, 2], equal = False):
    print("Data preparation phase")

    from sklearn.metrics import f1_score
    df = pd.read_excel('datasets/dataset.xlsx') # not '../datasets/dataset.xlsx' since we run main.py
    df = df[['sentence', 'label']]
    df = df[df['label'].isin(targets)]
    if equal:
        df = equalise(df)
    targets_put_down(df)

    df_tr, df_ts = split_data(df, test_ratio=0.2)

    df_tr.to_csv('tmpTrain.csv', index=False)
    df_ts.to_csv('tmpTest.csv', index=False)
    split = datasets.load_dataset('csv', data_files={
        'train': 'tmpTrain.csv',
        'test' : 'tmpTest.csv'
    })

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

    model = AutoModelForSequenceClassification.from_pretrained('Davit6174/georgian-distilbert-mlm', num_labels=len(targets),
                                                             ignore_mismatched_sizes=True, from_tf = True)
    tokenizer = AutoTokenizer.from_pretrained('Davit6174/georgian-distilbert-mlm')

    def tokenize_fn(batch):
        return tokenizer(batch['sentence'], truncation=True)

    tokenized_dataset = split.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(output_dir='save_dir_TC',
                                      evaluation_strategy='epoch',
                                      save_strategy='epoch',
                                      num_train_epochs=20,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=64,
                                      )

    def compute_metrics(logits_and_labels):
        logits, labels = logits_and_labels
        predictions = np.argmax(logits, axis=-1)
        acc = np.mean(predictions == labels)
        f1 = f1_score(labels, predictions, average='micro')
        return {'accuracy': acc, 'f1_score': f1}

    trainer = Trainer(model,
                      training_args,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset['test'],
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics)

    trainer.train()
