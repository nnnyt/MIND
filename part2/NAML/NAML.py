from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
import gensim
import json
import os
import random
from attention import Attention
# from tensorflow.keras.utils import to_categorical

MAX_TITLE_LENGTH = 30
MAX_ABSTRACT_LENGTH = 100
EMBEDDING_DIM = 300
C_EMBEDDING_DIM = 100
MAX_SENTS = 50
NEG_SAMPLE = 1

def preprocess_user_data(filename):
    # TODO
    print("Preprocessing user data...")
    browsed_news = []
    impression_news = []
    with open(filename, 'r') as f:
        for l in f:
            userID, time, history, impressions = l.strip('\n').split('\t')
            history = history.split()
            browsed_news.append(history)
            impressions = [x.split('-') for x in impressions.split()]
            impression_news.append(impressions)
    impression_pos = []
    impression_neg = []
    for impressions in impression_news:
        pos = []
        neg = []
        for news in impressions:
            if int(news[1]) == 1:
                pos.append(news[0])
            else:
                neg.append(news[0])
        impression_pos.append(pos)
        impression_neg.append(neg)
    all_browsed_news = []
    all_click = []
    all_unclick = []
    all_candidate = []
    all_label = []
    for k in range(len(browsed_news)):
        browsed = browsed_news[k]
        pos = impression_pos[k]
        neg = impression_neg[k]
        for pos_news in pos:
            all_browsed_news.append(browsed)
            all_click.append([pos_news])
            neg_news = random.sample(neg, NEG_SAMPLE)
            all_unclick.append(neg_news)
            all_candidate.append([pos_news]+neg_news)
            all_label.append([1] + [0] * NEG_SAMPLE)
            
    print('original behavior: ', len(browsed_news))
    print('processed behavior: ', len(all_browsed_news))
    return all_browsed_news, all_click, all_unclick, all_candidate, all_label


def preprocess_news_data(filename):
    print('Preprocessing news...')
    all_texts = []
    category_map = {}
    subcategory_map = {}
    titles = []
    abstracts = []
    categories = []
    subcategories = []
    news_index = {}

    with open(filename, 'r') as f:
        for l in f:
            id, category, subcategory, title, abstract, url, entity = l.strip('\n').split('\t')
            if id not in news_index:
                news_index[id] = len(news_index)
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
    # news_category = []
    news_category = np.zeros((len(categories), 1), dtype='int32')
    k = 0
    for category in categories:
        news_category[k][0] = category_map[category]
        # news_category.append(category)
        k += 1
    # news_category = to_categorical(np.asarray(news_category))
    # news_subcategory = []
    news_subcategory = np.zeros((len(subcategories), 1), dtype='int32')
    k = 0
    for subcategory in subcategories:
        news_subcategory[k][0] = subcategory_map[subcategory]
        # news_subcategory.append(subcategory)
        k += 1
    # news_subcategory = to_categorical(np.asarray(news_subcategory))

    return word_index, category_map, subcategory_map, news_category, news_subcategory, news_abstract, news_title, news_index

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
word_index, category_map, subcategory_map, news_category, news_subcategory, news_abstract, news_title, news_index = preprocess_news_data('../../data/MINDsmall_train/news.tsv')

# proprocess input
all_browsed_title = [[ news_title[news_index[j]] for j in i] for i in all_browsed_news]
all_browsed_abstract = [[ news_abstract[news_index[j]] for j in i] for i in all_browsed_news]
all_browsed_category = [[ news_category[news_index[j]] for j in i] for i in all_browsed_news]
all_browsed_subcategory = [[ news_subcategory[news_index[j]] for j in i] for i in all_browsed_news]
all_candidate_title = [[ news_title[news_index[j]] for j in i] for i in all_candidate]
all_candidate_abstract = [[ news_abstract[news_index[j]] for j in i] for i in all_candidate]
all_candidate_category = [[ news_category[news_index[j]] for j in i] for i in all_candidate]
all_candidate_subcategory = [[ news_subcategory[news_index[j]] for j in i] for i in all_candidate]

total = len(all_browsed_title)
train_num = int(0.8 * total)
train_browsed_title = all_browsed_title[:train_num]
train_browsed_abstract = all_browsed_abstract[:train_num]
train_browsed_category = all_browsed_category[:train_num]
train_browsed_subcategory = all_browsed_subcategory[:train_num]
train_candidate_abstract = all_candidate_abstract[:train_num]
train_candidate_title = all_candidate_title[:train_num]
train_candidate_category = all_candidate_category[:train_num]
train_candidate_subcategory = all_candidate_subcategory[:train_num]
train_label = all_label[:train_num]

test_browsed_title = all_candidate_title[train_num:]
test_browsed_abstract =  all_candidate_abstract[train_num:]
test_browsed_category = all_candidate_category[train_num:]
test_browsed_subcategory = all_candidate_subcategory[train_num:]
test_candidate_title = all_candidate_title[train_num:]
test_candidate_abstract = all_candidate_abstract[train_num:]
test_candidate_category = all_candidate_category[train_num:]
test_candidate_subcategory = all_candidate_subcategory[train_num:]
test_label = all_label[train_num:]

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
print("Train model...")
model.fit(train_browsed_title+train_browsed_abstract+train_browsed_category+train_browsed_subcategory+
          train_candidate_title + train_candidate_abstract+train_candidate_category+ train_candidate_subcategory,
          train_label,
          validation_data=(test_browsed_title+test_browsed_abstract+test_browsed_category+test_browsed_subcategory+
                            test_candidate_title+test_candidate_abstract+test_candidate_category+test_candidate_subcategory
                            , test_label),
          epochs=10, batch_size=50)
