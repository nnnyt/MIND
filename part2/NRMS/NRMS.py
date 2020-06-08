import numpy as np
from attention import Attention
from self_attention import SelfAttention
from preprocess import preprocess_news_data, preprocess_user_data, preprocess_test_user_data
from model import build_model


MAX_TITLE_LENGTH = 30
EMBEDDING_DIM = 300
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


def get_train_input(news_index, news_title):
    all_browsed_news, all_click, all_unclick, all_candidate, all_label = preprocess_user_data('../../data/MINDsmall_train/behaviors.tsv')
    print('preprocessing trainning input...')
    all_browsed_title = np.array([[ np.zeros(MAX_TITLE_LENGTH, dtype='int32')for i in range(MAX_BROWSED)] for _ in all_browsed_news])
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < MAX_BROWSED:
                all_browsed_title[i][j] = news_title[news_index[news]]
            j += 1

    all_candidate_title = np.array([[ news_title[news_index[j]] for j in i] for i in all_candidate])
    all_label = np.array(all_label)
    return all_browsed_title, all_candidate_title, all_label


def get_test_input(news_index_test, news_r_test):
    impression_index, all_browsed_test, all_candidate_test, all_label_test = preprocess_test_user_data('../../data/MINDsmall_dev/behaviors.tsv')
    print('preprocessing testing input...')
    all_browsed_title_test = np.array([[ np.zeros(256, dtype='float32') for i in range(MAX_BROWSED)] for _ in all_browsed_test])
    for i, user_browsed in enumerate(all_browsed_test):
        j = 0
        for news in user_browsed:
            if j < MAX_BROWSED:
                all_browsed_title_test[i][j] = news_r_test[news_index_test[news]]
            j += 1
    all_candidate_title_test = np.array([news_r_test[news_index_test[i[0]]] for i in all_candidate_test])
    all_label_test = np.array(all_label_test)
    return impression_index, all_browsed_title_test, all_candidate_title_test, all_label_test


if __name__ == "__main__":
    news_index, word_index, news_title, news_index_test, news_title_test = preprocess_news_data('../../data/MINDsmall_train/news.tsv', '../../data/MINDsmall_dev/news.tsv')
    all_browsed_title, all_candidate_title, all_label = get_train_input(news_index, news_title)
    # impression_index, all_browsed_title_test, all_candidate_title_test, all_label_test = get_test_input(news_index, news_title)

    news_encoder, model, model_test = build_model(word_index)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    plot_model(model_test, to_file='model_test.png', show_shapes=True)

    train_data = {}
    train_data['b_t'] = np.array(all_browsed_title)
    train_data['c_t'] = np.array(all_candidate_title)

    print("Train model...")
    model.fit(train_data, all_label, epochs=1, batch_size=50, validation_split=0.1)

    print("Get news representations for test...")
    news_r_test = news_encoder.predict(news_title_test, verbose=1, batch_size=50)

    print("Testing model...")
    impression_index, all_browsed_title_test, all_candidate_title_test, all_label_test = get_test_input(news_index_test, news_r_test)
    # print(len(news_r_test)) 42416
    # print(len(news_r_test[0])) 256
    # print(type(news_r_test[0][0])) float32

    test_data = {}
    test_data['user_input'] = np.array(all_browsed_title_test)
    test_data['c_t_1'] = np.array(all_candidate_title_test)

    pred_label = model_test.predict(test_data, verbose=1, batch_size=50)
    pred_label = np.array(pred_label).reshape(len(pred_label))
    all_label_test = np.array(all_label_test).reshape(len(all_label_test))
    auc, mrr, ndcg5, ndcg10 = evaluate(impression_index, all_label_test, pred_label)
    print('auc: ', auc)
    print('mrr: ', mrr)
    print('ndcg5: ', ndcg5)
    print('ndcg10: ', ndcg10)