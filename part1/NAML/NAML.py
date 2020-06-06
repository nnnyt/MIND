from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical
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
            

def preprocess_news_data(filename):
    print('Preprocessing news...')
    all_texts = []
    category_map = {}
    titles = []
    abstracts = []
    categories = []

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
            titles.append(title)
            abstracts.append(abstract)
            categories.append(category)
            
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
    news_category = []
    k = 0
    for category in categories:
        news_category.append(category_map[category])
        k += 1
    news_category = to_categorical(np.asarray(news_category))

    return word_index, category_map, news_category, news_abstract, news_title

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
word_index, category_map, news_category, news_abstract, news_title = preprocess_news_data('../../data/MINDsmall_train/news.tsv')
if os.path.exists('embedding_matrix.json'):
    print('Load embedding matrix...')
    embedding_matrix = np.array(read_json())
else:
    embedding_matrix = get_embedding(word_index)
    write_json(embedding_matrix.tolist())

print('Split train and validation...')
total = len(news_title)
train_num = int(0.8 * total)

import random
random.seed(212)
random.shuffle(news_category)
random.seed(212)
random.shuffle(news_title)
random.seed(212)
random.shuffle(news_abstract)

train_category = news_category[:train_num]
train_title = news_title[:train_num]
train_abstract = news_abstract[:train_num]
test_category = news_category[train_num:]
test_title = news_title[train_num:]
test_abstract = news_abstract[train_num:]

print('Build model...')
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, Concatenate
from tensorflow.keras.models import Model
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
# model
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


# concatenate
news_r = Concatenate(axis=-2)([title_attention, abstract_attention])
news_r = Attention(200)(news_r)

preds = Dense(news_category.shape[1], activation='softmax')(news_r)

model = Model([title_input, abstract_input], preds)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print('Train model...')
model.fit([train_title, train_abstract], train_category,
          validation_data=([test_title, test_abstract], test_category),
          epochs=10, batch_size=50)
