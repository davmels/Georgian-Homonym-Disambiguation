from .functions import *

def train_LSTM(targets = [0, 1, 2], show=True, equal = False, showConfMat = False, show_incorrect = False):
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix
    from keras.callbacks import EarlyStopping
    import keras
    N = len(targets)

    multi_class = True if N > 2 else False
    if show:
        print("loading model")

    model = load_w2v_model()

    if show:
        print("preprocessing data")
    df = get_data_specific_targets('datasets/dataset.xlsx', targets = targets)

    if equal:
        df = equalise(df) #eqaulises the number of sentences in all the classes.

    targets_put_down(df)

    df_tr, df_val, df_ts = split_data(df, val_ratio=0.2, test_ratio=0.2)
    tr, tr_t = get_final_data(df_tr, model)
    val, val_t = get_final_data(df_val, model)
    ts, ts_t = get_final_data(df_ts, model)
    tsc, ts_tc = get_final_data(df_ts, model, vectorised=False)
    if multi_class:
        tr_t = targets_one_hot(tr_t, N)
        val_t = targets_one_hot(val_t, N)
        ts_t = targets_one_hot(ts_t, N)

    if show:
        print("creating model")

    act = 'softmax' if multi_class else 'sigmoid'
    num = len(tr_t[0]) if multi_class else 1

    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(13, 128), return_sequences=True),
        keras.layers.LSTM(64),
        keras.layers.Dense(num, activation=act)
    ])

    optimiser = keras.optimizers.Adam()

    if multi_class:
        model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])

    batch_size = 32
    num_epochs = 40
    if show:
        print("training model")

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    verb = 2 if show else 0
    model.fit(tf.convert_to_tensor(tr), tf.convert_to_tensor(tr_t), validation_data=(tf.convert_to_tensor(val), tf.convert_to_tensor(val_t)),
              batch_size=batch_size, epochs=num_epochs, callbacks=[early_stopping], verbose=verb)

    y_pred = model.predict(tf.convert_to_tensor(ts))

    # Convert probabilities to class predictions
    y_pred_classes = np.argmax(y_pred, axis=1) if multi_class else (y_pred > 0.5)
    y_true_classes = np.argmax(ts_t, axis=1) if multi_class else ts_t

    if showConfMat:
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        print("Confusion Matrix:")
        print(conf_matrix)

    scores = model.evaluate(tf.convert_to_tensor(ts), tf.convert_to_tensor(ts_t), verbose=0)
    print('LSTM Test accuracy: ', scores[1])

    #this shows the incorrectly classified cases:
    if show_incorrect:
        for i in range(len(y_pred_classes)):
            if y_true_classes[i] != y_pred_classes[i]:
                print(tsc[i])
                print("real: %d, predicted:%d"% (y_true_classes[i],y_pred_classes[i]))

    return model

'''
    this function downloads remotely stored word2vec model from: https://huggingface.co/davmel/ka_word2vec
'''
def load_w2v_model():
    from gensim.models import Word2Vec
    from huggingface_hub import hf_hub_download
    repo_id = "davmel/ka_word2vec"

    filename = "word2vec/model_CC_128-10.model"
    filename2 = "word2vec/model_CC_128-10.model.syn1neg.npy"
    filename3 = "word2vec/model_CC_128-10.model.wv.vectors.npy"

    # Downloading the files
    hf_hub_download(repo_id=repo_id, filename=filename2)
    hf_hub_download(repo_id=repo_id, filename=filename3)
    wv2 = hf_hub_download(repo_id=repo_id, filename=filename)
    model = Word2Vec.load(wv2)
    return model
