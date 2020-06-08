import numpy as np
import gensim
import json
import os
from attention import Attention
from self_attention import SelfAttention
from tensorflow.keras.layers import Dense, Input, Dot, Activation, TimeDistributed
from tensorflow.keras.layers import Embedding, Dropout
from tensorflow.keras.models import Model


MAX_TITLE_LENGTH = 30
EMBEDDING_DIM = 300
MAX_BROWSED = 50
NEG_SAMPLE = 1


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


def get_embedding_matrix(word_index):
    if os.path.exists('embedding_matrix.json'):
        print('Load embedding matrix...')
        embedding_matrix = np.array(read_json())
    else:
        embedding_matrix = get_embedding(word_index)
        write_json(embedding_matrix.tolist())
    return embedding_matrix


def build_model(word_index):
    embedding_matrix = get_embedding_matrix(word_index)
    print('Building model...')
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
    # from tensorflow.keras.utils import plot_model
    # plot_model(news_encoder, to_file='news_encoder.png', show_shapes=True)

    # ----- user encoder -----
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

    return model, model_test
        