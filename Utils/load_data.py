import pickle
import numpy as np
import pandas as pd
from data_preprocessing.BERTtokenizer import BiobertEmbedding

NF = 'No Finding'
LA = 'Lung Opacity'
LABELS = 'Labels'
def get_multimodal_data(TRAIN, VAL, TEST, IMG, TXT, max_words):
    train_df = pd.read_pickle(TRAIN)
    val_df = pd.read_pickle(VAL)
    test_df = pd.read_pickle(TEST)
    if longitudinal:
        print("Preparing train data")
        x1_train = train_df[TXT].values
        x2_train = train_df[IMG].values
        y_train = train_df[LABELS].values

        print("Preparing val data")
        x1_val = val_df[TXT].values
        x2_val = val_df[IMG].values
        y_val = val_df[LABELS].values

        print("Preparing test data")
        x1_test = test_df[TXT].values
        x2_test = test_df[IMG].values
        y_test = test_df[LABELS].values
    else:
        print("Preparing train data")
        x1_train = train_df[TXT].astype(str).values
        x2_train = train_df[IMG].values
        for idx, path in enumerate(x2_train):
            filename = path
            x2_train[idx] = filename

        y_train = train_df[[NF, 'Enlarged Cardiomediastinum', 'Cardiomegaly', LA,
                        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                        'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values

        print("Preparing val data")
        x1_val = val_df[TXT].astype(str).values
        x2_val = val_df[IMG].values
        for idx, path in enumerate(x2_val):
            filename = path
            x2_val[idx] = filename

        y_val = val_df[[NF, 'Enlarged Cardiomediastinum', 'Cardiomegaly', LA,
                            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                            'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values

        print("Preparing test data")
        x1_test = test_df[TXT].astype(str).values
        x2_test = test_df[IMG].values
        for idx, path in enumerate(x2_test):
            filename = path
            x2_test[idx] = filename

        y_test = test_df[[NF, 'Enlarged Cardiomediastinum', 'Cardiomegaly', LA,
                            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                            'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values

    return x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test

def getTokenEmbed():
    print("Load tokenizer")
    with open('/home/tjvsonsbeek/Documents/physionet.org/files/mimiciii/1.4/tokenizer_reduced.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Load embedding_matrix")
    with open('/home/tjvsonsbeek/Documents/physionet.org/files/mimiciii/1.4/embedding_matrix_reduced.pickle', 'rb') as f:
        embedding_matrix = pickle.load(f)

    voc_size = len(tokenizer.word_index) + 1
    return tokenizer, embedding_matrix, voc_size

def getTargetWeights(y_tuple):
    
    weights = np.zeros(y.shape[1])
    for c in range(y.shape[1]):
        weights[c] = np.sum(y[:,c])
    weights = weights/y.shape[0]
    for c in range(y.shape[1]):
        weights[c] = 1 - weights[c]
        weights[c] = weights[c]**2
    return weights



def prepare_embeddings(t, vocab_size, model, WORD_EMBEDDINGS_SIZE):
    embedding_matrix = np.zeros((vocab_size, WORD_EMBEDDINGS_SIZE))
    for word, i in t.word_index.items():
        embedding_matrix[i] = model.wv[word]
    reverse_word_map = dict(map(reversed, t.word_index.items()))

    print("Saving tokenizer")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving embeddings of corpus")
    with open('embedding_matrix.pickle', 'wb') as f:
        pickle.dump(embedding_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_matrix, reverse_word_map
