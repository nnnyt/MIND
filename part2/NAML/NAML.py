from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
import gensim
import json
import os
import random
from attention import Attention
from preprocess import preprocess_user_data, preprocess_test_user_data, preprocess_news_data

MAX_TITLE_LENGTH = 30
MAX_ABSTRACT_LENGTH = 100
EMBEDDING_DIM = 300
C_EMBEDDING_DIM = 100
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


all_browsed_news, all_click, all_unclick, all_candidate, all_label = preprocess_user_data('../../data/MINDsmall_train/behaviors.tsv')
word_index, category_map, subcategory_map, news_category, news_subcategory, news_abstract, news_title, news_index = preprocess_news_data('../../data/MINDsmall_train/news.tsv', '../../data/MINDsmall_dev/news.tsv')
impression_index, all_browsed_test, all_candidate_test, all_label_test = preprocess_test_user_data('../../data/MINDsmall_dev/behaviors.tsv')

print('Preprocessing trainning input...')
# proprocess input
all_browsed_title = np.array([[ np.zeros(MAX_TITLE_LENGTH, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_news])
for i, user_browsed in enumerate(all_browsed_news):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_title[i][j] = news_title[news_index[news]]
        j += 1

all_browsed_abstract = np.array([[ np.zeros(MAX_ABSTRACT_LENGTH, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_news])
for i, user_browsed in enumerate(all_browsed_news):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_abstract[i][j] = news_abstract[news_index[news]]
        j += 1

all_browsed_category = np.array([[ np.zeros(1, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_news])
for i, user_browsed in enumerate(all_browsed_news):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_category[i][j] = news_category[news_index[news]]
        j += 1

all_browsed_subcategory = np.array([[ np.zeros(1, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_news])
for i, user_browsed in enumerate(all_browsed_news):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_subcategory[i][j] = news_subcategory[news_index[news]]
        j += 1

all_browsed = np.concatenate((all_browsed_title, all_browsed_abstract, all_browsed_category, all_browsed_subcategory), axis=-1)
all_candidate_title = np.array([[ news_title[news_index[j]] for j in i] for i in all_candidate])
all_candidate_abstract = np.array([[ news_abstract[news_index[j]] for j in i] for i in all_candidate])
all_candidate_category = np.array([[ news_category[news_index[j]] for j in i] for i in all_candidate])
all_candidate_subcategory = np.array([[ news_subcategory[news_index[j]] for j in i] for i in all_candidate])
all_candidate = np.concatenate((all_candidate_title, all_candidate_abstract, all_candidate_category, all_candidate_subcategory), axis=-1)
all_label = np.array(all_label)

print('Preprocessing testing input...')
all_browsed_title_test = np.array([[ np.zeros(MAX_TITLE_LENGTH, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_test])
for i, user_browsed in enumerate(all_browsed_test):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_title_test[i][j] = news_title[news_index[news]]
        j += 1

all_browsed_abstract_test = np.array([[ np.zeros(MAX_ABSTRACT_LENGTH, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_test])
for i, user_browsed in enumerate(all_browsed_test):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_abstract_test[i][j] = news_abstract[news_index[news]]
        j += 1

all_browsed_category_test = np.array([[ np.zeros(1, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_test])
for i, user_browsed in enumerate(all_browsed_test):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_category_test[i][j] = news_category[news_index[news]]
        j += 1

all_browsed_subcategory_test = np.array([[ np.zeros(1, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_test])
for i, user_browsed in enumerate(all_browsed_test):
    j = 0
    for news in user_browsed:
        if j < MAX_BROWSED:
            all_browsed_subcategory_test[i][j] = news_subcategory[news_index[news]]
        j += 1

all_browsed_test = np.concatenate((all_browsed_title_test, all_browsed_abstract_test, all_browsed_category_test, all_browsed_subcategory_test), axis=-1)
all_candidate_title_test = np.array([news_title[news_index[i[0]]] for i in all_candidate_test])
all_candidate_abstract_test = np.array([news_abstract[news_index[i[0]]] for i in all_candidate_test])
all_candidate_category_test = np.array([news_category[news_index[i[0]]] for i in all_candidate_test])
all_candidate_subcategory_test = np.array([news_subcategory[news_index[i[0]]] for i in all_candidate_test])
all_candidate_test = np.concatenate((all_candidate_title_test, all_candidate_abstract_test, all_candidate_category_test, all_candidate_subcategory_test), axis=-1)
all_label_test = np.array(all_label_test)

if os.path.exists('embedding_matrix.json'):
    print('Load embedding matrix...')
    embedding_matrix = np.array(read_json())
else:
    embedding_matrix = get_embedding(word_index)
    write_json(embedding_matrix.tolist())


print('Build model...')
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Dot, Activation, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, Concatenate
from tensorflow.keras.models import Model
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
# model
# ------ news encoder -------
news_input = Input((MAX_TITLE_LENGTH+MAX_ABSTRACT_LENGTH+2, ), dtype='int32')
from tensorflow.keras.layers import Lambda

# title
title_input = Lambda(lambda x: x[:, : MAX_TITLE_LENGTH])(news_input)
title_embedded_sequences = embedding_layer(title_input)
title_embedded_sequences = Dropout(0.2)(title_embedded_sequences)
title_cnn = Conv1D(400, 3, padding='same', activation='relu', strides=1)(title_embedded_sequences)
title_cnn = Dropout(0.2)(title_cnn)
title_attention = Attention(200)(title_cnn)
title_attention = Reshape((1, 400))(title_attention)

# abstract
abstract_input = Lambda(lambda x: x[:, MAX_TITLE_LENGTH : MAX_ABSTRACT_LENGTH + MAX_TITLE_LENGTH])(news_input)
abstract_embedded_sequences = embedding_layer(abstract_input)
abstract_embedded_sequences = Dropout(0.2)(abstract_embedded_sequences)
abstract_cnn = Conv1D(400, 3, padding='same', activation='relu', strides=1)(abstract_embedded_sequences)
abstract_cnn = Dropout(0.2)(abstract_cnn)
abstract_attention = Attention(200)(abstract_cnn)
abstract_attention = Reshape((1, 400))(abstract_attention)

# category
category_input = Lambda(lambda x: x[:, MAX_TITLE_LENGTH + MAX_ABSTRACT_LENGTH : MAX_TITLE_LENGTH + MAX_ABSTRACT_LENGTH + 1])(news_input)
category_embedding_layer = Embedding(len(category_map) + 1,
                                     C_EMBEDDING_DIM,
                                     trainable=True)
category_embedded = category_embedding_layer(category_input)
category_dense = Dense(400, activation='relu')(category_embedded)
category_dense = Reshape((1, 400))(category_dense)

# subcategory
subcategory_input = Lambda(lambda x: x[:, MAX_TITLE_LENGTH + MAX_ABSTRACT_LENGTH + 1 : ])(news_input)
subcategory_embedding_layer = Embedding(len(subcategory_map) + 1,
                                     C_EMBEDDING_DIM,
                                     trainable=True)
subcategory_embedded = subcategory_embedding_layer(subcategory_input)
subcategory_dense = Dense(400, activation='relu')(subcategory_embedded)
subcategory_dense = Reshape((1, 400))(subcategory_dense)

# concatenate
news_r = Concatenate(axis=-2)([title_attention, abstract_attention, category_dense, subcategory_dense])
news_r = Attention(200)(news_r)

news_encoder = Model(news_input, news_r)

# ----- user encoder -----
from tensorflow.keras import backend as K
browsed_input = Input((MAX_BROWSED, MAX_TITLE_LENGTH + MAX_ABSTRACT_LENGTH + 2, ), dtype='int32', name='browsed')
browsed_news = TimeDistributed(news_encoder)(browsed_input)
user_r = Attention(200)(browsed_news)

# ----- candidate_news -----
candidate_input = Input((1+NEG_SAMPLE, MAX_TITLE_LENGTH + MAX_ABSTRACT_LENGTH + 2, ), dtype='int32', name='candidate')
candidate_r = TimeDistributed(news_encoder)(candidate_input)

candidate_one_input = Input((MAX_TITLE_LENGTH + MAX_ABSTRACT_LENGTH + 2, ), dtype='int32', name='candidate_1')
candidate_one_r = news_encoder(candidate_one_input)

# ----- click predictor -----
pred = Dot(axes=-1)([user_r, candidate_r])
pred = Activation(activation='softmax')(pred)
model = Model([browsed_input, candidate_input], pred)

pred_one = Dot(axes=-1)([user_r, candidate_one_r])
pred_one = Activation(activation='sigmoid')(pred_one)
model_test = Model([browsed_input, candidate_one_input], 
                   pred_one)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)
# plot_model(model_test, to_file='model_test.png', show_shapes=True)

train_data = {}
train_data['browsed'] = np.array(all_browsed)
train_data['candidate'] = np.array(all_candidate)

test_data = {}
test_data['browsed'] = np.array(all_browsed_test)
test_data['candidate_1'] = np.array(all_candidate_test)

print("Train model...")
model.fit(train_data, 
          all_label,
          epochs=1, batch_size=50)

print("Tesing model...")
pred_label = model_test.predict(test_data, verbose=1, batch_size=50)
pred_label = np.array(pred_label).reshape(len(pred_label))
all_label_test = np.array(all_label_test).reshape(len(all_label_test))
auc, mrr, ndcg5, ndcg10 = evaluate(impression_index, all_label_test, pred_label)
print('auc: ', auc)
print('mrr: ', mrr)
print('ndcg5: ', ndcg5)
print('ndcg10: ', ndcg10)