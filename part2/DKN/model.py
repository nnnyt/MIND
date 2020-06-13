import numpy as np
import gensim
import json
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dot, Activation, TimeDistributed, Lambda
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, Conv2D, Reshape, Concatenate, GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from attention import Attention
from self_attention import SelfAttention


MAX_TITLE_LENGTH = 30
MAX_ENTITY_LENGTH = 30
EMBEDDING_DIM = 300
ENTITY_DIM = 100
MAX_BROWSED = 30
NEG_SAMPLE = 1


def write_json(embedding_matrix, filename='../embedding_matrix.json'):
    with open(filename, 'w') as f: 
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


def get_entity_embedding(entity_index):
    with open('embedding.json', 'r') as f:
        entity_model = json.load(f)
    embedding_matrix = np.zeros((len(entity_index) + 1, ENTITY_DIM))
    not_in_model = 0
    in_model = 0
    for word, i in entity_index.items(): 
        if word in entity_model:
            in_model += 1
            embedding_matrix[i] = np.asarray(entity_model[word], dtype='float32')
        else:
            not_in_model += 1
    print(str(in_model) + ' in embedding model')
    print(str(not_in_model)+' words not in embedding model')
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


def KCNN(word_index, entity_index):
    embedding_matrix = get_embedding_matrix(word_index)
    entity_embedding_matrix = get_entity_embedding(entity_index)
    print('Building model...')
    news_input = Input(shape=(MAX_TITLE_LENGTH+MAX_ENTITY_LENGTH, ), dtype='int32')
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)
    entity_embedding_layer = Embedding(len(entity_index) + 1,
                                        ENTITY_DIM,
                                        weights=[entity_embedding_matrix],
                                        trainable=True)

    title_input = Lambda(lambda x: x[:,: MAX_TITLE_LENGTH])(news_input)
    title_embedded_sequences = embedding_layer(title_input)
    title_embedded_sequences = Reshape((MAX_TITLE_LENGTH, EMBEDDING_DIM, 1, ))(title_embedded_sequences)

    entity_input = Lambda(lambda x: x[:, MAX_TITLE_LENGTH : ])(news_input)
    entity_embedded_sequences = entity_embedding_layer(entity_input)
    entity_embedded = Dense(EMBEDDING_DIM, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(entity_embedded_sequences)
    entity_embedded = Reshape((MAX_ENTITY_LENGTH, EMBEDDING_DIM, 1))(entity_embedded)

    embedding = Concatenate(axis=-1)([title_embedded_sequences, entity_embedded])

    cnn1 = Conv2D(100, (3,EMBEDDING_DIM), padding='valid', strides=1, activation='relu')(embedding)
    cnn1 = MaxPooling2D(pool_size=(28, 1))(cnn1)
    cnn2 = Conv2D(100, (4,EMBEDDING_DIM), padding='valid', strides=1, activation='relu')(embedding)
    cnn2 = MaxPooling2D(pool_size=(27, 1))(cnn2)
    cnn3 = Conv2D(100, (5,EMBEDDING_DIM), padding='valid', strides=1, activation='relu')(embedding)
    cnn3 = MaxPooling2D(pool_size=(26, 1))(cnn3)

    news_r = Concatenate(axis=-1)([cnn1, cnn2, cnn3])
    news_r = Reshape((300, ))(news_r)
    kcnn = Model(news_input, news_r, name='kcnn')

    return kcnn


def build_user_encoder(news_encoder):
    browsed_r_input = Input((MAX_BROWSED, 300, ), name='browsed')
    candidate_r_input = Input((300, ), name='candidate')

    candidate_news = Reshape((1, 300))(candidate_r_input)

    attention_weight = Lambda(lambda x: x[0] * x[1])([candidate_news, browsed_r_input])
    ReduceSum = Lambda(lambda z: K.sum(z, axis=-1))
    attention_weight= ReduceSum(attention_weight)
    attention_weight = Activation('softmax')(attention_weight)
    attention_weight = Reshape((MAX_BROWSED,1))(attention_weight)
    attention_r = Lambda(lambda x: x[0] * x[1])([attention_weight, browsed_r_input])
    ReduceSum = Lambda(lambda z: K.sum(z, axis=1))
    user_r = ReduceSum(attention_r)

    user_encoder = Model([browsed_r_input, candidate_r_input], user_r, name='user_encoder')
    return user_encoder


def build_model(word_index, entity_index):
    # model
    # ------ news encoder -------
    news_encoder = KCNN(word_index, entity_index)

    # ----- user encoder -------
    browsed_input = Input((MAX_BROWSED, MAX_TITLE_LENGTH + MAX_ENTITY_LENGTH, ), dtype='int32', name='browsed')
    browsed_news = TimeDistributed(news_encoder)(browsed_input)
    candidate_input = Input((MAX_TITLE_LENGTH + MAX_ENTITY_LENGTH, ), dtype='int32', name='candidate')
    candidate_news_r = news_encoder(candidate_input)

    user_encoder = build_user_encoder(news_encoder)

    train_user_r = user_encoder([browsed_news, candidate_news_r])

    test_browsed_input = Input((MAX_BROWSED, 300), name='browsed_test')
    # test_browsed_input = Input((MAX_BROWSED, 81), name='browsed_test')
    test_candidate_input = Input((300, ), name='candidate_test')
    # test_candidate_input = Input((81, ), name='candidate_test')
    test_user_r = user_encoder([test_browsed_input, test_candidate_input])

    # ----- click predictor -----
    pred = Dot(axes=-1)([train_user_r, candidate_news_r])
    pred = Activation(activation='sigmoid')(pred)

    pred_test = Dot(axes=-1)([test_user_r, test_candidate_input])
    pred_test = Activation(activation='sigmoid')(pred_test)
    
    model = Model([browsed_input, candidate_input], pred)
    model_test = Model([test_browsed_input, test_candidate_input], pred_test)

    return news_encoder, user_encoder, model, model_test

