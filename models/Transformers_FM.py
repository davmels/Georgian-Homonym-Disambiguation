import pandas as pd
from .functions import *


def fine_tune_transformer_FM(targets = [0, 1, 2], equal = False):
    from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer
    from sklearn.model_selection import train_test_split

    print('loading model')
    model = AutoModelForMaskedLM.from_pretrained('Davit6174/georgian-distilbert-mlm', from_tf=True)
    tokenizer = AutoTokenizer.from_pretrained('Davit6174/georgian-distilbert-mlm')

    print('creating mask fill dataset.')
    create_data_for_mask_fill(targets)
    df = pd.read_excel('finetune_FM.xlsx')

    if equal:
        df = equalise(df)
    targets_put_down(df)

    train_df, test_df = train_test_split(df, test_size=0.2)
    test_df.to_excel('test_FM.xlsx', index=False) #saving testing data for later

    #Data tokenization for training.
    tokenized_sentences_masked_ids = tokenizer(list(train_df['masked'].values), return_tensors='pt',
                                               truncation=True, padding=True, max_length=32)['input_ids']

    tokenized_sentences_label_ids = tokenizer(list(train_df['label'].values), return_tensors='pt', truncation=True,
                                              padding=True, max_length=32)['input_ids']

    tokenized_sentences_masked_ids_ts = tokenizer(list(train_df['masked'].values), return_tensors='pt',
                                                  truncation=True, padding=True, max_length=32)['input_ids']

    tokenized_sentences_label_ids_ts = tokenizer(list(train_df['label'].values), return_tensors='pt', truncation=True,
                                                 padding=True, max_length=32)['input_ids']

    #Preparing data to be trained and validated in a specific format.
    dataset = [{
        'input_ids': ids,
        'labels': labs,
    } for ids, labs in zip(tokenized_sentences_masked_ids, tokenized_sentences_label_ids)]

    evalset = [{
        'input_ids': ids,
        'labels': labs,
    } for ids, labs in zip(tokenized_sentences_masked_ids_ts, tokenized_sentences_label_ids_ts)]


    training_args = TrainingArguments(
        output_dir="save_dir_TF",
        num_train_epochs=20,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        save_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
    )

    trainer.train()

'''
    My way of preparing data for Fill-Mask task.
'''
def create_data_for_mask_fill(targets):
    df = pd.read_excel('datasets/dataset.xlsx')
    df = df[df['label'].isin(targets)]
    dct = {'masked': [], 'label': [], 'original': [], 'target': []}
    filename = 'datasets/full-homonym-sentences-ბარ.txt'
    duct = {}
    fr = open(filename, 'r', encoding='utf-8')

    for sent in fr.readlines():
        sen = " ".join([x for x in sent.strip().split() if x!='0'])
        ind = len([x for x in sent.strip().split()[:6] if x == '0'])
        duct[sen] = 6 - ind

    masker = {0: 'თოხი', 1: 'დაბლობი', 2: 'კაფე'}

    for i in range(len(df)):
        new_sentence = []
        masked_sentence = []
        sentence = df['sentence'].values[i]
        words = sentence.split(" ")
        for j in range(len(words)):
            if sentence in duct:
                if j == duct[sentence]:
                    new_sentence.append(masker[df['label'].values[i]])
                    masked_sentence.append('[MASK]')
                else:
                    new_sentence.append(words[j])
                    masked_sentence.append(words[j])
            else:
                if j == 6:
                    new_sentence.append(masker[df['label'].values[i]])
                    masked_sentence.append('[MASK]')
                else:
                    new_sentence.append(words[j])
                    masked_sentence.append(words[j])
        dct['masked'].append(' '.join(masked_sentence))
        dct['label'].append(' '.join(new_sentence))
        dct['original'].append(df['sentence'].values[i])
        dct['target'].append(df['label'].values[i])

    df2 = pd.DataFrame(dct)
    df2.to_excel('finetune_FM.xlsx')
