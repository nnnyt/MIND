from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
import gensim
import json
import os
from attention import Attention

MAX_TITLE_LENGTH = 30
MAX_ABSTRACT_LENGTH = 100
EMBEDDING_DIM = 300
C_EMBEDDING_DIM = 100
MAX_SENTS = 50
NEG_SAMPLE = 4

def preprocess_user_data(filename):
    # TODO
    print("Preprocessing user data...")

    with open(filename, 'r') as f:
        for l in f:
            userID, time, history, impressions = l.strip('\n').split('\t')
            history = history.split()
            impressions = impressions.split()


def preprocess_news_data(filename):
    print('Preprocessing news...')
    all_texts = []
    category_map = {}
    subcategory_map = {}
    titles = []
    abstracts = []
    categories = []
    subcategories = []

    with open(filename, 'r') as f:
        for l in f:
            id, category, subcategory, title, abstract, url, entity = l.strip('\n').split('\t')
            title = title.lower()
            abstract = abstract.lower()
            all_texts.append(title + ". " + abstract)
            # map every category to a number
            if category not in category_map:
                category_map[category] = len(category_map)
            # map every subcategory to a number
            if subcategory not in subcategory_map:
                subcategory_map[subcategory] = len(subcategory_map)
            titles.append(title)
            abstracts.append(abstract)
            categories.append(category)
            subcategories.append(subcategory)
            
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    word_index = tokenizer.word_index # a dict: word_index[word]=index
    print('Found %s unique tokens.' % len(word_index))
    # print(word_index)

    # title
    news_title = np.zeros((len(titles), MAX_TITLE_LENGTH), dtype='int32')
    for i, title in enumerate(titles):
        wordTokens = text_to_word_sequence(title)
        k = 0
        for _, word in enumerate(wordTokens):
            if k < MAX_TITLE_LENGTH:
                news_title[i, k] = word_index[word]
                k = k + 1
    
    # abstract
    news_abstract = np.zeros((len(abstracts), MAX_ABSTRACT_LENGTH), dtype='int32')
    for i, abstract in enumerate(abstracts):
        wordTokens = text_to_word_sequence(abstract)
        k = 0
        for _, word in enumerate(wordTokens):
            if k < MAX_ABSTRACT_LENGTH:
                news_abstract[i, k] = word_index[word]
                k = k + 1
    # category & subcategory
    news_category = np.zeros((len(categories), 1), dtype='int32')
    k = 0
    for category in categories:
        news_category[k][0] = category_map[category]
        k += 1
    news_subcategory = np.zeros((len(subcategories), 1), dtype='int32')
    k = 0
    for subcategory in subcategories:
        news_subcategory[k][0] = subcategory_map[subcategory]
        k += 1

    return word_index, category_map, subcategory_map, news_category, news_subcategory, news_abstract, news_title

def preprocess():
    preprocess_user_data('../../data/MINDsmall_train/behaviors.tsv')
    word_index, category_map, subcategory_map, news_category, news_subcategory, news_abstract, news_title = preprocess_news_data('../../data/MINDsmall_train/news.tsv')
    return word_index, category_map, subcategory_map, news_category, news_subcategory, news_abstract, news_title

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


# preprocess_news_data('../../data/MINDsmall_train/news.tsv')
word_index, category_map, subcategory_map, news_category, news_subcategory, news_abstract, news_title = preprocess_news_data('../../data/MINDsmall_train/news.tsv')
# word_index, category_map, subcategory_map, news_category, news_subcategory, news_abstract, news_title = preprocess()

if os.path.exists('embedding_matrix.json'):
    print('Load embedding matrix...')
    embedding_matrix = np.array(read_json())
else:
    embedding_matrix = get_embedding(word_index)
    write_json(embedding_matrix.tolist())

print('Build model...')
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Dot, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, Concatenate
from tensorflow.keras.models import Model
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
# model
# ------ news encoder -------
# title
title_input = Input(shape=(MAX_TITLE_LENGTH,), dtype='int32')
title_embedded_sequences = embedding_layer(title_input)
title_embedded_sequences = Dropout(0.2)(title_embedded_sequences)
title_cnn = Conv1D(400, 3, padding='same', activation='relu', strides=1)(title_embedded_sequences)
title_cnn = Dropout(0.2)(title_cnn)
title_attention = Attention(200)(title_cnn)
title_attention = Reshape((1, 400))(title_attention)

# abstract
abstract_input = Input(shape=(MAX_ABSTRACT_LENGTH,), dtype='int32')
abstract_embedded_sequences = embedding_layer(abstract_input)
abstract_embedded_sequences = Dropout(0.2)(abstract_embedded_sequences)
abstract_cnn = Conv1D(400, 3, padding='same', activation='relu', strides=1)(abstract_embedded_sequences)
abstract_cnn = Dropout(0.2)(abstract_cnn)
abstract_attention = Attention(200)(abstract_cnn)
abstract_attention = Reshape((1, 400))(abstract_attention)

# category
category_input = Input(shape=(1,), dtype='int32')
category_embedding_layer = Embedding(len(category_map) + 1,
                                     C_EMBEDDING_DIM,
                                     trainable=True)
category_embedded = category_embedding_layer(category_input)
# category_embedded = Flatten()(category_embedded)
category_dense = Dense(400, activation='relu')(category_embedded)
category_dense = Reshape((1, 400))(category_dense)

# subcategory
subcategory_input = Input(shape=(1,), dtype='int32')
subcategory_embedding_layer = Embedding(len(subcategory_map) + 1,
                                     C_EMBEDDING_DIM,
                                     trainable=True)
subcategory_embedded = subcategory_embedding_layer(subcategory_input)
# subcategory_embedded = Flatten(subcategory_embedded)
subcategory_dense = Dense(400, activation='relu')(subcategory_embedded)
subcategory_dense = Reshape((1, 400))(subcategory_dense)

# concatenate
news_r = Concatenate(axis=-2)([title_attention, abstract_attention, category_dense, subcategory_dense])
news_r = Attention(200)(news_r)

news_encoder = Model([title_input, abstract_input, category_input, subcategory_input], news_r)

# ----- user encoder -----
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
browsed_title_input = [Input((MAX_TITLE_LENGTH, ), dtype='int32') for _ in range(MAX_SENTS)]
browsed_abstract_input = [Input((MAX_ABSTRACT_LENGTH, ), dtype='int32') for _ in range(MAX_SENTS)]
browsed_category_input = [Input((1, ), dtype='int32') for _ in range(MAX_SENTS)]
browsed_subcategory_input = [Input((1, ), dtype='int32') for _ in range(MAX_SENTS)]
browsed_news = [news_encoder([browsed_title_input[i], browsed_abstract_input[i], browsed_category_input[i], browsed_subcategory_input[i]]) for i in range(MAX_SENTS)]
# browsed_news = Concatenate(axis=-2)(browsed_news)
browsed_news = Concatenate(axis=-2)([Lambda(lambda x: K.expand_dims(x,axis=1))(news) for news in browsed_news])

user_r = Attention(200)(browsed_news)

# ----- candidate_news -----
candidate_title_input = [Input((MAX_TITLE_LENGTH, ), dtype='int32') for _ in range(1+NEG_SAMPLE)]
candidate_abstract_input = [Input((MAX_ABSTRACT_LENGTH, ), dtype='int32') for _ in range(1+NEG_SAMPLE)]
candidate_category_input = [Input((1, ), dtype='int32') for _ in range(1+NEG_SAMPLE)]
candidate_subcategory_input = [Input((1, ), dtype='int32') for _ in range(1+NEG_SAMPLE)]
candidate_r = [news_encoder([candidate_title_input[i], candidate_abstract_input[i], candidate_category_input[i], candidate_subcategory_input[i]]) for i in range(1+NEG_SAMPLE)]

# ----- click predictor -----
pred = [Dot(axes=-1)([user_r, candidate_r[i]]) for i in range(1+NEG_SAMPLE)]
pred = Concatenate()(pred)
pred = Activation(activation='softmax')(pred)

model = Model(browsed_title_input + browsed_abstract_input + browsed_category_input + browsed_subcategory_input +
               candidate_title_input + candidate_abstract_input + candidate_category_input + candidate_subcategory_input
               , pred)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
