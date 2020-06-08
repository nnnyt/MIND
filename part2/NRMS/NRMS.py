from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import gensim
import json
import os
import random
from attention import Attention
from self_attention import SelfAttention
from preprocess import preprocess_news_data, preprocess_user_data, preprocess_test_user_data

MAX_TITLE_LENGTH = 30
EMBEDDING_DIM = 300
MAX_BROWSED = 50
NEG_SAMPLE = 1

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def evaluate(impression_index, all_label_test, pred_label):
    from sklearn.metrics import roc_auc_score
    all_auc = []
    all_mrr = []
    all_ndcg5 = []
    all_ndcg10 = []
    for i in impression_index:
        begin = int(i[0])
        end = int(i[1])
        auc = roc_auc_score(all_label_test[begin:end], pred_label[begin:end])
        all_auc.append(auc)
        mrr = mrr_score(all_label_test[begin:end], pred_label[begin:end])
        all_mrr.append(mrr)
        if end - begin > 4:
            ndcg5 = ndcg_score(all_label_test[begin:end], pred_label[begin:end], 5)
            all_ndcg5.append(ndcg5)
            if end - begin > 9:
                ndcg10 = ndcg_score(all_label_test[begin:end], pred_label[begin:end], 10)
                all_ndcg10.append(ndcg10)
    return np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg5), np.mean(all_ndcg10)

def write_json(embedding_matrix):
    with open('embedding_matrix.json', 'w') as f: 
        json.dump(embedding_matrix, f)

def read_json(file='embedding_matrix.json'):
    with open(file, 'r') as f:
        embedding_matrix = json.load(f)
    return embedding_matrix

def get_embedding(word_index):
    # use glove
    print('Loading glove...')
    glove_file = "../../glove_model.txt"
    glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_file,binary=False) # GloVe Model
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    not_in_model = 0
    in_model = 0
    for word, i in word_index.items(): 
        if word in glove_model:
            in_model += 1
            embedding_matrix[i] = np.asarray(glove_model[word], dtype='float32')
        else:
            not_in_model += 1
    print(str(in_model) + ' in glove model')
    print(str(not_in_model)+' words not in glove model')
    print('Embedding matrix shape: ', embedding_matrix.shape)
    return embedding_matrix

news_index, word_index, news_title = preprocess_news_data('../../data/MINDsmall_train/news.tsv', '../../data/MINDsmall_dev/news.tsv')
all_browsed_news, all_click, all_unclick, all_candidate, all_label = preprocess_user_data('../../data/MINDsmall_train/behaviors.tsv')
impression_index, all_browsed_test, all_candidate_test, all_label_test = preprocess_test_user_data('../../data/MINDsmall_dev/behaviors.tsv')

print('preprocessing trainning input...')
all_browsed_title = np.array([[ np.zeros(MAX_TITLE_LENGTH, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_news])
for i, user_browsed in enumerate(all_browsed_news):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_title[i][j] = news_title[news_index[news]]
        j += 1

all_candidate_title = np.array([[ news_title[news_index[j]] for j in i] for i in all_candidate])
all_label = np.array(all_label)

print('preprocessing testing input...')
all_browsed_title_test = np.array([[ np.zeros(MAX_TITLE_LENGTH, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_test])
for i, user_browsed in enumerate(all_browsed_test):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_title_test[i][j] = news_title[news_index[news]]
        j += 1

all_candidate_title_test = np.array([news_title[news_index[i[0]]] for i in all_candidate_test])
all_label_test = np.array(all_label_test)

if os.path.exists('embedding_matrix.json'):
    print('Load embedding matrix...')
    embedding_matrix = np.array(read_json())
else:
    embedding_matrix = get_embedding(word_index)
    write_json(embedding_matrix.tolist())

print('Building model...')
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Dot, Activation, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, Concatenate
from tensorflow.keras.models import Model
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
# model
# ------ news encoder -------
title_input = Input(shape=(MAX_TITLE_LENGTH,), dtype='int32')
title_embedded_sequences = embedding_layer(title_input)
title_embedded_sequences = Dropout(0.2)(title_embedded_sequences)
title_selfattention = SelfAttention(16, 16)([title_embedded_sequences, title_embedded_sequences, title_embedded_sequences])
title_selfattention = Dropout(0.2)(title_selfattention)
news_r = Attention(200)(title_selfattention)

news_encoder = Model([title_input], news_r)

# ----- user encoder -----
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
browsed_title_input = Input((MAX_BROWSED, MAX_TITLE_LENGTH, ), dtype='int32', name='b_t')
browsed_news = TimeDistributed(news_encoder)(browsed_title_input)
browsed_news = SelfAttention(16, 16)([browsed_news, browsed_news, browsed_news])
browsed_news = Dropout(0.2)(browsed_news)
user_r = Attention(200)(browsed_news)

# ----- candidate_news -----
candidate_title_input = Input((1+NEG_SAMPLE, MAX_TITLE_LENGTH, ), dtype='int32', name='c_t')
candidate_r = TimeDistributed(news_encoder)(candidate_title_input)

candidate_one_title_input = Input((MAX_TITLE_LENGTH, ), dtype='int32', name='c_t_1')
candidate_one_r = news_encoder([candidate_one_title_input])

# ----- click predictor -----
pred = Dot(axes=-1)([user_r, candidate_r])
pred = Activation(activation='softmax')(pred)
model = Model([browsed_title_input, candidate_title_input], pred)

pred_one = Dot(axes=-1)([user_r, candidate_one_r])
pred_one = Activation(activation='sigmoid')(pred_one)
model_test = Model([browsed_title_input, candidate_one_title_input], pred_one)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)
# plot_model(model_test, to_file='model_test.png', show_shapes=True)

train_data = {}
train_data['b_t'] = np.array(all_browsed_title)
train_data['c_t'] = np.array(all_candidate_title)
test_data = {}
test_data['b_t'] = np.array(all_browsed_title_test)
test_data['c_t_1'] = np.array(all_candidate_title_test)

print("Train model...")
model.fit(train_data, all_label, epochs=3, batch_size=50, validation_split=0.1)

print("Tesing model...")
pred_label = model_test.predict(test_data, verbose=1, batch_size=50)
pred_label = np.array(pred_label).reshape(len(pred_label))
all_label_test = np.array(all_label_test).reshape(len(all_label_test))
auc, mrr, ndcg5, ndcg10 = evaluate(impression_index, all_label_test, pred_label)
print('auc: ', auc)
print('mrr: ', mrr)
print('ndcg5: ', ndcg5)
print('ndcg10: ', ndcg10)