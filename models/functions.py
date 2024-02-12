import pandas as pd
import numpy as np
from gensim.models import Word2Vec

'''
    Utility functions -- are used for data preprocessing.
'''

def split_data(df, test_ratio = 0.15, val_ratio = 0.):
    from sklearn.model_selection import train_test_split
    # Split the filtered DataFrame into training and test sets
    train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df['label'])
    if val_ratio != 0:
        tr_df, val_df = train_test_split(train_df, test_size=val_ratio, stratify=train_df['label'])
        return tr_df, val_df, test_df
    else:
        return train_df, test_df

'''
    if targets are [3, 7, 8] 
    the function converts them to: [0, 1, 2]   *(3->0,  7->1,  8->2)*
    since the training functions requre the class labels to be given like that (in ascending order from 0 to K)
'''
def targets_put_down(df):
    classes = set(df['label'].values)
    vah = 0
    for all in classes:
        df['label'].replace(all, vah, inplace=True)
        vah += 1

#eqaulises the number of sentences in all the classes.
def equalise(df):
    import pandas as pd
    from imblearn.under_sampling import RandomUnderSampler

    X = df.drop('label', axis=1)
    y = df['label']

    rus = RandomUnderSampler(sampling_strategy='auto')

    X_resampled, y_resampled = rus.fit_resample(X, y)

    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='label')],
                             axis=1)

    return df_resampled

def get_data_specific_targets(filename, targets):
    df = pd.read_excel(filename)
    filtered_df = df[df['label'].isin(targets)]
    return filtered_df


def get_final_data(df, word2vec_model, vectorised = True, concat = False):
    text, target = df['sentence'].values, df['label'].values
    ultret = []
    for i in range(len(text)):
        ret = text[i].split(' ')
        rett = []
        if len(ret) < 13:
            ret.extend([0] * (13 - len(ret)))

        if vectorised:
            for j in range(13):
                rett.append(vectorise_word(word2vec_model, ret[j]))
        else:
            rett = ret

        if concat:
            ultret.append(np.concatenate(rett))
        else:
            ultret.append(np.array(rett))
    return np.asarray(ultret), target

def vectorise_word(model, word):
    if word == '0':
        return np.zeros(128) #I consider 128-dimensional vectors...
    if word in model.wv.key_to_index:
        return model.wv[word]
    return np.zeros(128) #If a word is not in the vocabulary, I return zeros.

'''
    convert numerical class values into one-hot vectors:
    example:
        classes: 0,1,2
        one-hot classes:  [1,0,0],  [0,1,0],  [0,0,1]
'''
def targets_one_hot(tar, hou):
    ret = []
    for i in range(len(tar)):
        hey = np.zeros(hou)
        hey[tar[i]] = 1
        ret.append(hey)
    return ret



def loadWord2VecModel(filename):
    return Word2Vec.load(filename)
