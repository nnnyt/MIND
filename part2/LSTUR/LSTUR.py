import numpy as np
from preprocess import preprocess_user_data, preprocess_test_user_data, preprocess_news_data
from model import build_model


MAX_TITLE_LENGTH = 30
MAX_ABSTRACT_LENGTH = 100
EMBEDDING_DIM = 300
C_EMBEDDING_DIM = 100
USER_R_DIM = 600
NEWS_R_DIM = 600
MAX_BROWSED = 50
NEG_SAMPLE = 1
TYPE = 'ini'


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


def get_train_input(news_category, news_subcategory, news_title, news_index):
    all_browsed_news, all_click, all_unclick, all_candidate, all_label, all_user, user_index = preprocess_user_data('../../data/MINDsmall_train/behaviors.tsv')
    
    all_browsed_title = np.zeros((len(all_browsed_news), MAX_BROWSED, MAX_TITLE_LENGTH), dtype='int32')
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < MAX_BROWSED:
                all_browsed_title[i][j] = news_title[news_index[news]]
            j += 1

    all_browsed_category = np.zeros((len(all_browsed_news), MAX_BROWSED, 1), dtype='int32')
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < MAX_BROWSED:
                all_browsed_category[i][j] = news_category[news_index[news]]
            j += 1

    all_browsed_subcategory = np.zeros((len(all_browsed_news), MAX_BROWSED, 1), dtype='int32')
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < MAX_BROWSED:
                all_browsed_subcategory[i][j] = news_subcategory[news_index[news]]
            j += 1

    all_browsed = np.concatenate((all_browsed_title, all_browsed_category, all_browsed_subcategory), axis=-1)
    all_candidate_title = np.array([[ news_title[news_index[j]] for j in i] for i in all_candidate])
    all_candidate_category = np.array([[ news_category[news_index[j]] for j in i] for i in all_candidate])
    all_candidate_subcategory = np.array([[ news_subcategory[news_index[j]] for j in i] for i in all_candidate])
    all_candidate = np.concatenate((all_candidate_title, all_candidate_category, all_candidate_subcategory), axis=-1)
    all_user = np.array(all_user)
    print(all_user.shape)
    all_label = np.array(all_label)
    return all_browsed, all_candidate, all_label, all_user, user_index


def get_test_input(news_r_test, news_index_test, impression_index, user_index, user_browsed_test, all_user_test, all_candidate_test, all_label_test):
    user_browsed_news_test = np.zeros((len(user_browsed_test), MAX_BROWSED, NEWS_R_DIM), dtype='float32')
    for i, user_browsed in enumerate(user_browsed_test):
        j = 0
        for news in user_browsed:
            if j < MAX_BROWSED:
                user_browsed_news_test[i][j] = news_r_test[news_index_test[news]]
            j += 1
    user_id_test = np.array(all_user_test)
    all_candidate_news_test = np.array([news_r_test[news_index_test[i[0]]] for i in all_candidate_test])
    all_label_test = np.array(all_label_test)
    return user_browsed_news_test, user_id_test, all_candidate_news_test, all_label_test, impression_index, user_index, all_user_test


if __name__ == "__main__":
    word_index, category_map, subcategory_map, news_category, news_subcategory, news_title, news_index, news_index_test, all_news_test = preprocess_news_data('../../data/MINDsmall_train/news.tsv', '../../data/MINDsmall_dev/news.tsv')

    print('Preprocessing trainning input...')
    all_browsed, all_candidate, all_label, all_user, user_index = get_train_input(news_category, news_subcategory, news_title, news_index)
    impression_index, user_index, user_browsed_test, all_user_test, all_candidate_test, all_label_test, user_index_test = preprocess_test_user_data('../../data/MINDsmall_dev/behaviors.tsv', user_index)
    
    news_encoder, user_encoder, model, model_test= build_model(word_index, category_map, subcategory_map, user_index, TYPE)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    # from tensorflow.keras.utils import plot_model
    # plot_model(model, to_file='model.png', show_shapes=True)
    # plot_model(model_test, to_file='model_test.png', show_shapes=True)
    # plot_model(news_encoder, to_file='news_encoder.png', show_shapes=True)
    # plot_model(user_encoder, to_file='user_encoder.png', show_shapes=True)

    train_data = {}
    train_data['browsed'] = np.array(all_browsed)
    train_data['user_id'] = np.array(all_user)
    train_data['candidate'] = np.array(all_candidate)

    print('Train model...')
    model.fit(train_data, all_label, epochs=1, batch_size=64, validation_split=0.1)
    
    print("Get news representations for test...")
    news_r_test = news_encoder.predict(all_news_test, verbose=1, batch_size=50)

    print("Tesing model...")
    user_browsed_news_test, user_id_test, all_candidate_news_test, all_label_test, impression_index, user_index, all_user_test = get_test_input(news_r_test, news_index_test, impression_index, user_index, user_browsed_test, all_user_test, all_candidate_test, all_label_test)

    print("Get user representations...")
    print(user_browsed_news_test.shape)
    print(user_id_test.shape)
    user_r_test = user_encoder.predict([user_browsed_news_test, user_id_test], verbose=1, batch_size=64)

    all_user_r_test = np.zeros((len(all_user_test), USER_R_DIM))
    print(len(all_user_test))
    print(len(user_index_test))
    # print(user_index_test)
    for i, user in enumerate(all_user_test):
        a = user_r_test[user_index_test[user]]
        all_user_r_test[i] = a
    
    test_data = {}
    test_data['test_user_r'] = np.array(all_user_r_test)
    test_data['candidate_1'] = np.array(all_candidate_news_test)

    pred_label = model_test.predict(test_data, verbose=1, batch_size=50)
    pred_label = np.array(pred_label).reshape(len(pred_label))
    all_label_test = np.array(all_label_test).reshape(len(all_label_test))
    auc, mrr, ndcg5, ndcg10 = evaluate(impression_index, all_label_test, pred_label)
    print('auc: ', auc)
    print('mrr: ', mrr)
    print('ndcg5: ', ndcg5)
    print('ndcg10: ', ndcg10)