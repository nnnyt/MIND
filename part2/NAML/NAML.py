from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
import gensim
import json
import os
import random
from attention import Attention

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

def preprocess_user_data(filename):
    print("Preprocessing user data...")
    browsed_news = []
    impression_news = []
    with open(filename, "r") as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 1)
    use_data = data[:use_num]
    for l in use_data:
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

def preprocess_test_user_data(filename):
    print("Preprocessing test user data...")
    with open(filename, 'r') as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 0.01)
    use_data = data[:use_num]
    impression_index = []
    all_browsed_test = []
    all_candidate_test = []
    all_label_test = []
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        history = history.split()
        impressions = [x.split('-') for x in impressions.split()]
        begin = len(all_candidate_test)
        end = len(impressions) + begin
        impression_index.append([begin, end])
        for news in impressions:
            all_browsed_test.append(history)
            all_candidate_test.append([news[0]])
            all_label_test.append([int(news[1])])
    print('test samples: ', len(all_label_test))
    return impression_index, all_browsed_test, all_candidate_test, all_label_test

def preprocess_news_data(filename, filename_2):
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
    
    with open(filename_2, 'r') as f:
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
    print('Found %s unique news.' % len(news_index))
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
            
all_candidate_title = np.array([[ news_title[news_index[j]] for j in i] for i in all_candidate])
all_candidate_abstract = np.array([[ news_abstract[news_index[j]] for j in i] for i in all_candidate])
all_candidate_category = np.array([[ news_category[news_index[j]] for j in i] for i in all_candidate])
all_candidate_subcategory = np.array([[ news_subcategory[news_index[j]] for j in i] for i in all_candidate])
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

all_candidate_title_test = np.array([[ news_title[news_index[j]] for j in i] for i in all_candidate_test])
all_candidate_abstract_test = np.array([[ news_abstract[news_index[j]] for j in i] for i in all_candidate_test])
all_candidate_category_test = np.array([[ news_category[news_index[j]] for j in i] for i in all_candidate_test])
all_candidate_subcategory_test = np.array([[ news_subcategory[news_index[j]] for j in i] for i in all_candidate_test])
all_label_test = np.array(all_label_test)

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
category_dense = Dense(400, activation='relu')(category_embedded)
category_dense = Reshape((1, 400))(category_dense)

# subcategory
subcategory_input = Input(shape=(1,), dtype='int32')
subcategory_embedding_layer = Embedding(len(subcategory_map) + 1,
                                     C_EMBEDDING_DIM,
                                     trainable=True)
subcategory_embedded = subcategory_embedding_layer(subcategory_input)
subcategory_dense = Dense(400, activation='relu')(subcategory_embedded)
subcategory_dense = Reshape((1, 400))(subcategory_dense)

# concatenate
news_r = Concatenate(axis=-2)([title_attention, abstract_attention, category_dense, subcategory_dense])
news_r = Attention(200)(news_r)

news_encoder = Model([title_input, abstract_input, category_input, subcategory_input], news_r)

# ----- user encoder -----
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
browsed_title_input = [Input((MAX_TITLE_LENGTH, ), dtype='int32', name='b_t'+str(_)) for _ in range(MAX_BROWSED)]
browsed_abstract_input = [Input((MAX_ABSTRACT_LENGTH, ), dtype='int32', name='b_a'+str(_)) for _ in range(MAX_BROWSED)]
browsed_category_input = [Input((1, ), dtype='int32', name='b_c'+str(_)) for _ in range(MAX_BROWSED)]
browsed_subcategory_input = [Input((1, ), dtype='int32', name='b_sc'+str(_)) for _ in range(MAX_BROWSED)]
browsed_news = [news_encoder([browsed_title_input[i], browsed_abstract_input[i], browsed_category_input[i], browsed_subcategory_input[i]]) for i in range(MAX_BROWSED)]
# browsed_news = Concatenate(axis=-2)(browsed_news)
browsed_news = Concatenate(axis=-2)([Lambda(lambda x: K.expand_dims(x,axis=1))(news) for news in browsed_news])

user_r = Attention(200)(browsed_news)

# ----- candidate_news -----
candidate_title_input = [Input((MAX_TITLE_LENGTH, ), dtype='int32', name='c_t'+str(_)) for _ in range(1+NEG_SAMPLE)]
candidate_abstract_input = [Input((MAX_ABSTRACT_LENGTH, ), dtype='int32', name='c_a'+str(_)) for _ in range(1+NEG_SAMPLE)]
candidate_category_input = [Input((1, ), dtype='int32', name='c_c'+str(_)) for _ in range(1+NEG_SAMPLE)]
candidate_subcategory_input = [Input((1, ), dtype='int32',name='c_sc'+str(_)) for _ in range(1+NEG_SAMPLE)]
candidate_r = [news_encoder([candidate_title_input[i], candidate_abstract_input[i], candidate_category_input[i], candidate_subcategory_input[i]]) for i in range(1+NEG_SAMPLE)]

candidate_one_title_input = Input((MAX_TITLE_LENGTH, ), dtype='int32', name='c_t_1')
candidate_one_abstract_input = Input((MAX_ABSTRACT_LENGTH, ), dtype='int32', name='c_a_1')
candidate_one_category_input = Input((1, ), dtype='int32', name='c_c_1')
candidate_one_subcategory_input = Input((1, ), dtype='int32', name='c_sc_1')
candidate_one_r = news_encoder([candidate_one_title_input, candidate_one_abstract_input, candidate_one_category_input, candidate_one_subcategory_input])
# ----- click predictor -----
pred = [Dot(axes=-1)([user_r, candidate_r[i]]) for i in range(1+NEG_SAMPLE)]
pred = Concatenate()(pred)
pred = Activation(activation='softmax')(pred)
model = Model(browsed_title_input + candidate_title_input + 
              browsed_abstract_input + candidate_abstract_input + 
              browsed_category_input + candidate_category_input + 
              browsed_subcategory_input + candidate_subcategory_input
               , pred)

pred_one = Dot(axes=-1)([user_r, candidate_one_r])
pred_one = Activation(activation='sigmoid')(pred_one)
model_test = Model(browsed_title_input + [candidate_one_title_input] + 
                   browsed_abstract_input + [candidate_one_abstract_input] + 
                   browsed_category_input + [candidate_one_category_input] +
                   browsed_subcategory_input + [candidate_one_subcategory_input],
                   pred_one)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)

print("Processing trainning data...")
train_data = {}
for j in range(MAX_BROWSED):
    train_data['b_t'+str(j)] = []
    train_data['b_a'+str(j)] = []
    train_data['b_c'+str(j)] = []
    train_data['b_sc'+str(j)] = []
    for i in range(len(all_browsed_title)):
        train_data['b_t'+str(j)].append(all_browsed_title[i][j])
        train_data['b_a'+str(j)].append(all_browsed_abstract[i][j])
        train_data['b_c'+str(j)].append(all_browsed_category[i][j])
        train_data['b_sc'+str(j)].append(all_browsed_subcategory[i][j])
    train_data['b_t'+str(j)] = np.array(train_data['b_t'+str(j)])
    train_data['b_a'+str(j)] = np.array(train_data['b_a'+str(j)])
    train_data['b_c'+str(j)] = np.array(train_data['b_c'+str(j)])
    train_data['b_sc'+str(j)] = np.array(train_data['b_sc'+str(j)])
print(train_data['b_t0'].shape)
for j in range(1+NEG_SAMPLE):
    train_data['c_t'+str(j)] = []
    train_data['c_a'+str(j)] = []
    train_data['c_c'+str(j)] = []
    train_data['c_sc'+str(j)] = []
    for i in range(len(all_candidate_title)):
        train_data['c_t'+str(j)].append(all_candidate_title[i][j])
        train_data['c_a'+str(j)].append(all_candidate_abstract[i][j])
        train_data['c_c'+str(j)].append(all_candidate_category[i][j])
        train_data['c_sc'+str(j)].append(all_candidate_subcategory[i][j])
    train_data['c_t'+str(j)] = np.array(train_data['c_t'+str(j)])
    train_data['c_a'+str(j)] = np.array(train_data['c_a'+str(j)])
    train_data['c_c'+str(j)] = np.array(train_data['c_c'+str(j)])
    train_data['c_sc'+str(j)] = np.array(train_data['c_sc'+str(j)])

print('Processing testing data...')
test_data = {}
for j in range(MAX_BROWSED):
    test_data['b_t'+str(j)] = []
    test_data['b_a'+str(j)] = []
    test_data['b_c'+str(j)] = []
    test_data['b_sc'+str(j)] = []
    for i in range(len(all_browsed_title_test)):
        test_data['b_t'+str(j)].append(all_browsed_title_test[i][j])
        test_data['b_a'+str(j)].append(all_browsed_abstract_test[i][j])
        test_data['b_c'+str(j)].append(all_browsed_category_test[i][j])
        test_data['b_sc'+str(j)].append(all_browsed_subcategory_test[i][j])
    test_data['b_t'+str(j)] = np.array(test_data['b_t'+str(j)])
    test_data['b_a'+str(j)] = np.array(test_data['b_a'+str(j)])
    test_data['b_c'+str(j)] = np.array(test_data['b_c'+str(j)])
    test_data['b_sc'+str(j)] = np.array(test_data['b_sc'+str(j)])
for j in range(1):
    test_data['c_t_1'] = []
    test_data['c_a_1'] = []
    test_data['c_c_1'] = []
    test_data['c_sc_1'] = []
    for i in range(len(all_candidate_title_test)):
        test_data['c_t_1'].append(all_candidate_title_test[i][j])
        test_data['c_a_1'].append(all_candidate_abstract_test[i][j])
        test_data['c_c_1'].append(all_candidate_category_test[i][j])
        test_data['c_sc_1'].append(all_candidate_subcategory_test[i][j])
    test_data['c_t_1'] = np.array(test_data['c_t_1'])
    test_data['c_a_1'] = np.array(test_data['c_a_1'])
    test_data['c_c_1'] = np.array(test_data['c_c_1'])
    test_data['c_sc_1'] = np.array(test_data['c_sc_1'])

print("Train model...")
model.fit(train_data, 
          all_label,
          epochs=8, batch_size=50)

print("Tesing model...")
pred_label = model_test.predict(test_data, verbose=1, batch_size=50)
pred_label = np.array(pred_label).reshape(len(pred_label))
all_label_test = np.array(all_label_test).reshape(len(all_label_test))
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
print('auc: ', np.mean(all_auc))
print('mrr: ', np.mean(all_mrr))
print('ndcg5: ', np.mean(all_ndcg5))
print('ndcg10: ', np.mean(all_ndcg10))
    