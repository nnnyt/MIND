import numpy as np
import gensim
import json
import os
from attention import Attention
from tensorflow.keras.layers import Dense, Input, Reshape, Dot, Activation, TimeDistributed, Lambda
from tensorflow.keras.layers import Conv1D, Embedding, Dropout, Concatenate
from tensorflow.keras.models import Model


MAX_TITLE_LENGTH = 30
MAX_ABSTRACT_LENGTH = 100
EMBEDDING_DIM = 300
C_EMBEDDING_DIM = 100
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


def build_news_encoder(word_index, category_map, subcategory_map):
    embedding_matrix = get_embedding_matrix(word_index)
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)
    news_input = Input((MAX_TITLE_LENGTH+MAX_ABSTRACT_LENGTH+2, ), dtype='int32')

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
    return news_encoder


def build_model(word_index, category_map, subcategory_map):
    print('Build model...')
    # model
    # ------ news encoder -------
    news_encoder = build_news_encoder(word_index, category_map, subcategory_map)

    # ----- user encoder -----
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
    return model, model_test
