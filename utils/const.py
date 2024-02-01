import os


def init_setting_Amazon():
    global load_path, user_vocab, query_vocab, src_train, src_val, src_test, rec_train, rec_val, rec_test

    load_path = "data/Amazon"
    user_vocab = os.path.join(load_path, 'vocab/user_vocab.pkl')
    query_vocab = os.path.join(load_path, 'vocab/query_vocab.pkl')

    src_train = os.path.join(load_path, 'dataset/src_train.pkl')
    src_val = os.path.join(load_path, 'dataset/src_val.pkl')
    src_test = os.path.join(load_path, 'dataset/src_test.pkl')

    rec_train = os.path.join(load_path, 'dataset/rec_train.pkl')
    rec_val = os.path.join(load_path, 'dataset/rec_val.pkl')
    rec_test = os.path.join(load_path, 'dataset/rec_test.pkl')

    global user_map_vocab, item_map_vocab, session_map_vocab
    user_map_vocab = os.path.join(load_path, 'vocab/user_vocab_np.pkl')
    item_map_vocab = os.path.join(load_path, 'vocab/item_vocab_np.pkl')
    session_map_vocab = os.path.join(load_path,
                                     'vocab/src_session_vocab_np.pkl')

    global item_id_num, item_id_dim, item_feature_list, item_text_feature
    item_feature_list = ['item_id']
    item_text_feature = []
    item_id_num = 61935 + 1
    item_id_dim = 32

    global user_id_num, user_id_dim, user_feature_list

    user_feature_list = ['user_id']
    user_id_num = 68223
    user_id_dim = 32

    global word_id_num, word_id_dim
    word_id_num = 1866
    word_id_dim = 32

    global final_emb_size
    final_emb_size = 64

    global max_rec_his_len, max_src_session_his_len, max_session_item_len, max_query_word_len
    max_rec_his_len = 30
    max_src_session_his_len = 30
    max_session_item_len = 1
    max_query_word_len = 50


def init_setting_KuaiSAR():
    global load_path, user_vocab, query_vocab, src_train, src_val, src_test, rec_train, rec_val, rec_test

    load_path = "data/KuaiSAR"
    user_vocab = os.path.join(load_path, 'vocab/user_vocab.pkl')
    query_vocab = os.path.join(load_path, 'vocab/query_vocab.pkl')

    src_train = os.path.join(load_path, 'dataset/src_train.pkl')
    src_val = os.path.join(load_path, 'dataset/src_val.pkl')
    src_test = os.path.join(load_path, 'dataset/src_test.pkl')

    rec_train = os.path.join(load_path, 'dataset/rec_train.pkl')
    rec_val = os.path.join(load_path, 'dataset/rec_val.pkl')
    rec_test = os.path.join(load_path, 'dataset/rec_test.pkl')

    global user_map_vocab, item_map_vocab, session_map_vocab
    user_map_vocab = os.path.join(load_path, 'vocab/user_vocab_np.pkl')
    item_map_vocab = os.path.join(load_path, 'vocab/item_vocab_np.pkl')
    session_map_vocab = os.path.join(load_path,
                                     'vocab/src_session_vocab_np.pkl')

    global item_feature_list, item_text_feature, item_id_num, first_level_category_id_num, second_level_category_id_num, \
    item_id_dim, first_level_category_id_dim, second_level_category_id_dim

    item_feature_list = [
        'item_id', 'caption', 'first_level_category_id',
        'second_level_category_id'
    ]
    item_text_feature = ['caption']

    item_id_num = 673415 + 1
    first_level_category_id_num = 38
    second_level_category_id_num = 297
    item_id_dim = 32
    first_level_category_id_dim = 32
    second_level_category_id_dim = 32

    global user_id_num, onehot_feat1_num, onehot_feat2_num, search_active_level_num, rec_active_level_num,\
        user_feature_list

    user_feature_list = [
        'user_id', 'onehot_feat1', 'onehot_feat2', 'search_active_level',
        'rec_active_level'
    ]
    user_id_num = 22700
    onehot_feat1_num = 8
    onehot_feat2_num = 3
    search_active_level_num = 7
    rec_active_level_num = 4

    global user_id_dim, onehot_feat1_dim, onehot_feat2_dim, search_active_level_dim, rec_active_level_dim
    user_id_dim = 32
    onehot_feat1_dim = 32
    onehot_feat2_dim = 32
    search_active_level_dim = 32
    rec_active_level_dim = 32

    global word_id_num, word_id_dim, final_emb_size
    word_id_num = 394912
    word_id_dim = 32
    final_emb_size = 64

    global max_rec_his_len, max_src_session_his_len, max_session_item_len, max_query_word_len
    max_rec_his_len = 30
    max_src_session_his_len = 30
    max_session_item_len = 5
    max_query_word_len = 50
