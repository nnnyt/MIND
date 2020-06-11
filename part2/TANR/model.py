import numpy as np
import gensim
import json
import os
from attention import Attention
from tensorflow.keras.layers import Dense, Input, Dot, Activation, TimeDistributed
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, Concatenate, Reshape
from tensorflow.keras.models import Model


MAX_TITLE_LENGTH = 30
EMBEDDING_DIM = 300
MAX_BROWSED = 50
NEG_SAMPLE = 1


def write_json(embedding_matrix):
    with open('../embedding_matrix.json', 'w') as f: 
        json.dump(embedding_matrix, f)


def read_json(file='../embedding_matrix.json'):
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
    if os.path.exists('../embedding_matrix.json'):
        print('Load embedding matrix...')
        embedding_matrix = np.array(read_json())
    else:
        embedding_matrix = get_embedding(word_index)
        write_json(embedding_matrix.tolist())
    return embedding_matrix


def build_model(word_index, news_category):
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
    title_cnn = Conv1D(400, 3, padding='same', activation='relu', strides=1)(title_embedded_sequences)
    title_cnn = Dropout(0.2)(title_cnn)
    news_r = Attention(200)(title_cnn)

    news_encoder = Model([title_input], news_r, name='news_encoder')

    # ----- user encoder -----
    browsed_title_input = Input((MAX_BROWSED, MAX_TITLE_LENGTH, ), dtype='int32', name='b_t')
    browsed_news = TimeDistributed(news_encoder)(browsed_title_input)

    user_input = Input((MAX_BROWSED, 400, ), name='user_input')
    user_r = Attention(200)(user_input)
    user_encoder = Model(user_input, user_r, name='user_encoder')

    train_user_r = user_encoder(browsed_news)
    test_user_r = Input((400, ), name='test_user_r')

    browsed_topic_pred = TimeDistributed(Dense(news_category.shape[1], activation='softmax'))(browsed_news)

    # ----- candidate_news -----
    candidate_title_input = Input((1+NEG_SAMPLE, MAX_TITLE_LENGTH, ), dtype='int32', name='c_t')
    candidate_r = TimeDistributed(news_encoder)(candidate_title_input)
    candidate_topic_pred = TimeDistributed(Dense(news_category.shape[1], activation='softmax'))(candidate_r)

    candidate_one_r = Input((400, ), name="c_t_1")

    topic_pred = Concatenate(axis=-2, name='topic_pred')([browsed_topic_pred, candidate_topic_pred])

    # ----- click predictor -----
    pred = Dot(axes=-1)([train_user_r, candidate_r])
    pred = Activation(activation='softmax', name='click_pred')(pred)
    model = Model([browsed_title_input, candidate_title_input], [ pred, topic_pred])
    # model = Model([browsed_title_input, candidate_title_input], pred)

    pred_one = Dot(axes=-1)([test_user_r, candidate_one_r])
    pred_one = Activation(activation='sigmoid')(pred_one)
    model_test = Model([test_user_r, candidate_one_r], pred_one)

    return news_encoder, user_encoder, model, model_test
        