import numpy as np
import gensim
import json
import os
from attention import Attention
from tensorflow.keras.layers import Dense, Input, Dot, Activation, TimeDistributed, GRU, Masking
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, Concatenate, Lambda, Reshape
from tensorflow.keras.models import Model


MAX_TITLE_LENGTH = 30
EMBEDDING_DIM = 300
C_EMBEDDING_DIM = 100
USER_R_DIM = 600
NEWS_R_DIM = 600
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


def build_news_encoder(word_index, category_map, subcategory_map):
    embedding_matrix = get_embedding_matrix(word_index)
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)
    news_input = Input((MAX_TITLE_LENGTH+2, ), dtype='int32')

    # title
    title_input = Lambda(lambda x: x[:, : MAX_TITLE_LENGTH])(news_input)
    title_embedded_sequences = embedding_layer(title_input)
    title_embedded_sequences = Dropout(0.2)(title_embedded_sequences)
    title_cnn = Conv1D(400, 3, padding='same', activation='relu', strides=1)(title_embedded_sequences)
    title_cnn = Dropout(0.2)(title_cnn)
    title_attention = Attention(200)(title_cnn)

    # category
    category_input = Lambda(lambda x: x[:, MAX_TITLE_LENGTH : MAX_TITLE_LENGTH + 1])(news_input)
    category_embedding_layer = Embedding(len(category_map) + 1,
                                        C_EMBEDDING_DIM,
                                        trainable=True)
    category_embedded = category_embedding_layer(category_input)
    category_embedded = Reshape((100,))(category_embedded)
    
    # subcategory
    subcategory_input = Lambda(lambda x: x[:, MAX_TITLE_LENGTH + 1 : ])(news_input)
    subcategory_embedding_layer = Embedding(len(subcategory_map) + 1,
                                            C_EMBEDDING_DIM,
                                            trainable=True)
    subcategory_embedded = subcategory_embedding_layer(subcategory_input)
    subcategory_embedded = Reshape((100, ))(subcategory_embedded)

    news_r = Concatenate(axis=-1)([title_attention, category_embedded, subcategory_embedded])
    
    news_encoder = Model(news_input, news_r, name='news_encoder')
    return news_encoder


def build_user_encoder(news_encoder, user_index, model_type='ini'):
    browsed_news = Input((MAX_BROWSED, NEWS_R_DIM, ), name='browsed')
    browsed_news = Masking(mask_value=0.0)(browsed_news)

    user_input = Input((1, ), dtype='int32', name='user')
    if model_type == 'ini':
        user_embedding_layer = Embedding(len(user_index) + 1,
                                        USER_R_DIM,
                                        embeddings_initializer='zeros',
                                        trainable=True)
        user_long_r = user_embedding_layer(user_input)
        user_long_r = Reshape((USER_R_DIM, ))(user_long_r)
        user_r = GRU(USER_R_DIM)(browsed_news,initial_state=user_long_r)
    else:
        half_dim = int(USER_R_DIM / 2)
        user_embedding_layer = Embedding(len(user_index) + 1,
                                        half_dim,
                                        embeddings_initializer='zeros',
                                        trainable=True)
        user_long_r = user_embedding_layer(user_input)
        user_long_r = Reshape((half_dim, ))(user_long_r)
        user_r = GRU(half_dim)(browsed_news)
        user_r = Concatenate()([user_r, user_long_r])

    user_encoder = Model([browsed_news, user_input], user_r, name='user_encoder')
    return user_encoder


def build_model(word_index, category_map, subcategory_map, user_index, model_type='ini'):
    print('Building model...')

    # model
    # ------ news encoder -------
    news_encoder = build_news_encoder(word_index, category_map, subcategory_map)

    # ----- user encoder -----
    browsed_input = Input((MAX_BROWSED, MAX_TITLE_LENGTH + 2, ), dtype='int32', name='browsed')
    browsed_news = TimeDistributed(news_encoder)(browsed_input)
    user_id = Input((1, ), dtype='int32', name='user_id')

    user_encoder = build_user_encoder(news_encoder, user_index, model_type)
    train_user_r = user_encoder([browsed_news, user_id])

    test_user_r = Input((USER_R_DIM, ), name='test_user_r')
    # ----- candidate_news -----
    candidate_input = Input((1+NEG_SAMPLE, MAX_TITLE_LENGTH + 2, ), dtype='int32', name='candidate')
    candidate_r = TimeDistributed(news_encoder)(candidate_input)

    candidate_one_r = Input((NEWS_R_DIM, ), name="candidate_1")

    # ----- click predictor -----
    pred = Dot(axes=-1)([train_user_r, candidate_r])
    pred = Activation(activation='softmax')(pred)
    model = Model([browsed_input, user_id, candidate_input], pred)

    pred_one = Dot(axes=-1)([test_user_r, candidate_one_r])
    pred_one = Activation(activation='sigmoid')(pred_one)
    model_test = Model([test_user_r, candidate_one_r], 
                    pred_one)

    return news_encoder, user_encoder, model, model_test