import argparse
from collections import defaultdict

import networkx as nx
import pandas as pd
import scipy.sparse as sp
from scipy.io import loadmat
from tqdm import tqdm

from utils import *

torch.multiprocessing.set_sharing_strategy('file_system')
np.set_printoptions(threshold=1000000)
saved_path = './'


def load_and_save_epinions():
    rating_mat = loadmat(f'datasets/{args.dataset}/rating_with_timestamp.mat')
    rating = rating_mat['rating_with_timestamp']
    df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
    df.drop(columns=['cate', 'help'], inplace=True)
    df = preprocess_uir(df, prepro='origin', pos_threshold=3)
    df.drop(columns=['rate'], inplace=True)
    u2i = np.array(df.values, dtype=np.int32)

    uu_elist = loadmat(f'datasets/{args.dataset}/trust.mat')['trust']
    u2u = np.array(uu_elist, dtype=np.int32)

    save_path = f'datasets/{args.dataset}/u2ui.npz'
    np.savez(file=save_path, u2u=u2u, u2i=u2i)
    print('saved at', save_path)


# ----------------------------------------------------------------------------------------

def filter_and_reid(prepro='3filter', level='u'):
    '''
    Raw u2i (8021100, 3)
    min user = 1
    max user = 1968703
    num user = 1968703
    min item = 1
    max item = 209393
    num item = 209393

    Raw u2u edges: 19042100

    '''

    u2ui = np.load(f'./datasets/{args.dataset}/u2ui.npz')
    u2u, u2i = u2ui['u2u'], u2ui['u2i']
    df = pd.DataFrame(data=u2i, columns=['user', 'item', 'ts'])
    df.drop_duplicates(subset=['user', 'item', 'ts'], keep='first', inplace=True)

    print('Raw u2i', df.shape)
    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    print('num user =', len(np.unique(df.values[:, 0])))
    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    print('num item =', len(np.unique(df.values[:, 1])))

    df = preprocess_uir(df, prepro=prepro, level=level)

    print('Processed u2i', df.shape)
    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    print('num user =', len(np.unique(df.values[:, 0])))
    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    print('num item =', len(np.unique(df.values[:, 1])))

    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
    u2i = df.values

    user_idmap = defaultdict(int)  # src id -> new id
    user_idmap[0] = 0
    user_num = 1

    for i, (user, item, ts) in tqdm(enumerate(u2i)):
        if user_idmap[user] == 0:
            user_idmap[user] = user_num
            user_num += 1

        u2i[i, 0] = user_idmap[user]

    print('Raw u2u edges:', len(u2u))
    new_uu_elist = []
    for u1, u2 in tqdm(u2u):
        new_u1 = user_idmap[u1]
        new_u2 = user_idmap[u2]
        if new_u1 and new_u2:
            new_uu_elist.append([new_u1, new_u2])

    print('Processed u2u edges:', len(new_uu_elist))
    u2u = np.array(new_uu_elist).astype(np.int32)
    u2i = u2i.astype(np.int32)

    save_path = f'datasets/{args.dataset}/reid_u2ui.npz'
    np.savez(file=save_path, u2u=u2u, u2i=u2i)

    print('saved at', save_path)


def delete_isolated_user():
    u2ui = np.load(f'datasets/{args.dataset}/reid_u2ui.npz')
    uu_elist = u2ui['u2u']
    u2i = u2ui['u2i']

    # print('Building u2u graph...')
    # user_num = np.max(u2i[:, 0]) + 1
    # g = nx.Graph()
    # g.add_nodes_from(list(range(user_num)))
    # g.add_edges_from(uu_elist)
    # g.remove_node(0)
    #
    # isolated_user_set = set(nx.isolates(g))
    # print('Isolated user =', len(isolated_user_set))
    #
    # new_u2i = []
    # for user, item, ts in tqdm(u2i):
    #     if user not in isolated_user_set:
    #         new_u2i.append([user, item, ts])
    #
    # new_u2i = np.array(new_u2i, dtype=np.int32)
    #
    # print('No isolated user u2i =', new_u2i.shape)
    #
    # user_idmap = defaultdict(int)  # src id -> new id
    # user_idmap[0] = 0
    # user_num = 1
    # for i, (user, item, ts) in tqdm(enumerate(new_u2i)):
    #     if user_idmap[user] == 0:
    #         user_idmap[user] = user_num
    #         user_num += 1
    #
    #     new_u2i[i, 0] = user_idmap[user]
    #
    # new_uu_elist = []
    # for u1, u2 in tqdm(uu_elist):
    #     new_u1 = user_idmap[u1]
    #     new_u2 = user_idmap[u2]
    #     if new_u1 and new_u2:
    #         new_uu_elist.append([new_u1, new_u2])
    #
    # new_uu_elist = np.array(new_uu_elist, dtype=np.int32)
    #
    # df = pd.DataFrame(data=new_u2i, columns=['user', 'item', 'ts'])
    # df['item'] = pd.Categorical(df['item']).codes + 1
    #
    # print(df.head(20))
    #
    # # cc_sizes = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    # # print('u2u connected components sizes (top20):', cc_sizes[:20])
    # # print('Isolated user =', np.sum(np.array(cc_sizes) == 1))
    #
    # user_num = df['user'].max() + 1
    # item_num = df['item'].max() + 1
    #
    # print('min user =', df['user'].min())
    # print('max user =', df['user'].max())
    # num_user = len(np.unique(df.values[:, 0]))
    # print('num user =', num_user)
    #
    # print('min item =', df['item'].min())
    # print('max item =', df['item'].max())
    # num_item = len(np.unique(df.values[:, 1]))
    # print('num item =', num_item)
    #
    # print(f'Loaded {args.dataset} dataset with {user_num} users, {item_num} items, '
    #       f'{len(df.values)} u2i, {len(new_uu_elist)} u2u. ')
    # new_u2i = df.values.astype(np.int32)

    num_user = max(max(uu_elist[:, 0]), max(uu_elist[:, 1]), max(u2i[:, 0]))
    num_item = max(u2i[:, 1])
    new_uu_elist = uu_elist
    new_u2i = u2i
    save_path = f'datasets/{args.dataset}/noiso_reid_u2ui.npz'
    np.savez(file=save_path, u2u=new_uu_elist, u2i=new_u2i)
    np.save(f'datasets/{args.dataset}/user_item_num.npy', np.array([num_user, num_item]))
    return num_user, num_item


def data_partition(df):
    print('Splitting train/val/test set...')
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    item_train = defaultdict(list)

    eval_users = []
    valid_items = []
    test_items = []

    user_items_dict = defaultdict(list)

    def apply_fn1(grp):
        key_id = grp['user'].values[0]
        user_items_dict[key_id] = grp[['item', 'ts']].values

    def apply_fn2(grp):
        key_id = grp['item'].values[0]
        item_train[key_id] = grp[['user', 'ts']].values

    df.groupby('user').apply(apply_fn1)
    df.groupby('item').apply(apply_fn2)

    print('Groupby user finished.')

    for user in tqdm(user_items_dict.keys()):
        nfeedback = len(user_items_dict[user])
        if nfeedback < 5:
            user_train[user] = user_items_dict[user]
        else:
            # Append user history items
            eval_users.append(user)
            user_train[user] = user_items_dict[user][:-2]

            # Second last item for validation
            valid_item = user_items_dict[user][-2][0]
            user_valid[user].append(valid_item)
            valid_items.append(valid_item)

            # Last item for test
            test_item = user_items_dict[user][-1][0]
            user_test[user].append(test_item)
            test_items.append(test_item)

            for i, j in enumerate(item_train[user_items_dict[user][-2][0]]):
                if j[0] == user:
                    tmp = item_train[user_items_dict[user][-2][0]].tolist()
                    tmp.pop(i)
                    item_train[user_items_dict[user][-2][0]] = np.array(tmp)
                    break
            for i, j in enumerate(item_train[user_items_dict[user][-1][0]]):
                if j[0] == user:
                    tmp = item_train[user_items_dict[user][-1][0]].tolist()
                    tmp.pop(i)
                    item_train[user_items_dict[user][-1][0]] = np.array(tmp)
                    break

    return user_train, user_valid, user_test, eval_users, valid_items, test_items, item_train


def gen_and_save_u2u_dict_and_split(num_user, num_item):
    # if os.path.exists('./datasets/Yelp/u2u_split_dicts.pkl'):
    #     return
    u2ui = np.load(f'datasets/{args.dataset}/noiso_reid_u2ui.npz')

    print('Building u2u graph...')
    g = nx.Graph()
    g.add_nodes_from(list(range(num_user)))
    g.add_edges_from(u2ui['u2u'])
    g.remove_node(0)

    print('To undirected graph...')
    g.to_undirected()
    # g.add_edges_from([[u, u] for u in g.nodes])
    u2u_dict = nx.to_dict_of_lists(g)

    df = pd.DataFrame(data=u2ui['u2i'], columns=['user', 'item', 'ts'])
    print('Raw u2i =', df.shape)
    df.drop_duplicates(subset=['user', 'item', 'ts'], keep='first', inplace=True)
    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
    print('Processed u2i =', df.shape)

    user_train, user_valid, user_test, eval_users, valid_items, test_items, item_train = data_partition(df)
    save_path = f'datasets/{args.dataset}/u2u_split_dicts.pkl'
    save_pkl(save_path, [
        u2u_dict, user_train, user_valid, user_test,
        eval_users, valid_items, test_items, item_train])

    print('saved at', save_path)


def get_nbr(u2u, user, nbr_maxlen):
    nbr = np.zeros([nbr_maxlen, ], dtype=np.int32)
    nbr_len = len(u2u[user])
    if nbr_len == 0:
        pass
    elif nbr_len > nbr_maxlen:
        np.random.shuffle(u2u[user])
        nbr[:] = u2u[user][:nbr_maxlen]
    else:
        nbr[:nbr_len] = u2u[user]

    return nbr


def get_nbr_iids(user_train, user, nbrs, time_splits):
    nbr_maxlen = len(nbrs)
    seq_maxlen = len(time_splits)
    nbrs_iids = np.zeros((nbr_maxlen, seq_maxlen), dtype=np.int32)

    start_idx = np.nonzero(time_splits)[0]
    if len(start_idx) == 0:
        return nbrs_iids
    else:
        start_idx = start_idx[0]

    user_first_ts = time_splits[start_idx]
    user_last_ts = time_splits[-1]

    for i, nbr in enumerate(nbrs):
        if nbr == 0 or nbr == user:
            continue

        nbr_hist = user_train[nbr]

        if len(nbr_hist) == 0:
            continue

        nbr_first_ts = nbr_hist[0][1]
        nbr_last_ts = nbr_hist[-1][1]

        if nbr_first_ts > user_last_ts or nbr_last_ts <= user_first_ts:
            continue

        sample_list = list()
        for j in range(start_idx + 1, seq_maxlen):
            start_time = time_splits[j - 1]
            end_time = time_splits[j]

            if start_time != end_time:
                sample_list = list(filter(None, map(
                    lambda x: x[0] if x[1] > start_time and x[1] <= end_time else None, nbr_hist
                )))

            if len(sample_list):
                # print('st={} et={} sl={}'.format(start_time, end_time, sample_list))
                nbrs_iids[i, j] = np.random.choice(sample_list)

    return nbrs_iids


def get_y_m_d(a):
    a_y = a // 10000
    a_m = (a % 10000) // 100
    a_d = (a % 100)
    return a_y, a_m, a_d


def get_date_dis(a, b):
    a_y, a_m, a_d = get_y_m_d(a)
    b_y, b_m, b_d = get_y_m_d(b)
    res = abs(365 * (a_y - b_y) + 30 * (a_m - b_m) + a_d - b_d)
    return res


def gen_and_save_all_user_batches(user_num, item_num):
    # eval batch for each user
    u2u_dict, user_train, user_valid, user_test, eval_users, valid_items, test_items, item_train = \
        load_pkl(f'datasets/{args.dataset}/u2u_split_dicts.pkl')
    check_meta_len = []

    def sample_one_user(user):
        seq = np.zeros(seq_maxlen, dtype=np.int32)
        pos = np.zeros(seq_maxlen, dtype=np.int32)
        ts = np.zeros(seq_maxlen, dtype=np.int32)
        nxt = user_train[user][-1, 0]
        idx = seq_maxlen - 1
        meta = np.zeros((seq_maxlen, meta_maxlen), dtype=np.int64)

        max_distime = -1
        min_distime = 999999999
        mean_distime = []
        suit_num = 0
        for (item, time_stamp) in reversed(user_train[user][:-1]):
            seq[idx] = item
            ts[idx] = time_stamp
            pos[idx] = nxt
            tmp_list = []
            for i in item_train[item]:
                if abs(i[1] - time_stamp) < time_limit:
                    distime = get_date_dis(i[1], time_stamp)
                    mean_distime.append(distime)
                    suit_num += 1
                    tmp_list.append(i[0])
                    if distime > max_distime:
                        max_distime = distime
                    if distime < min_distime:
                        min_distime = distime

            if len(tmp_list) >= meta_maxlen:
                meta[idx] = np.array(np.random.choice(tmp_list, size=meta_maxlen), dtype=np.int64)
            else:
                for i, tmp_user in enumerate(tmp_list):
                    meta[idx, i] = tmp_user
            check_meta_len.append(len(tmp_list))
            nxt = item
            idx -= 1
            if idx == -1: break

        nbr = get_nbr(u2u_dict, user, nbr_maxlen)
        nbr_iid = get_nbr_iids(user_train, user, nbr, ts)
        nbr_iid = sp.csr_matrix(nbr_iid, dtype=np.int32)
        if len(mean_distime) <= 0:
            return user, seq, pos, nbr, nbr_iid, meta, -1, -1, -1, -1, -1
        else:
            mean_distime.sort()
            min_distime = -1
            mean_distime = np.array(mean_distime)
            if len(mean_distime) % 2 == 1:
                mid_distime = mean_distime[len(mean_distime) // 2]
            else:
                mid_distime = (mean_distime[len(mean_distime) // 2] + mean_distime[(len(mean_distime) // 2) - 1]) / 2

            mean_distime = mean_distime.mean()

            return user, seq, pos, nbr, nbr_iid, meta, max_distime, min_distime, mean_distime, mid_distime, suit_num

    uid_list = []
    seq_list = []
    pos_list = []
    nbr_list = []
    nbr_iid_list = []

    meta_list = []
    max_distime_list = []
    min_distime_list = []
    mean_distime_list = []
    mid_distime_list = []
    suit_num_list = []

    for user in tqdm(range(1, user_num)):
        user, seq, pos, nbr, nbr_iid, meta, max_distime, min_distime, mean_distime, mid_distime, suit_num = sample_one_user(
            user)
        uid_list.append(user)
        seq_list.append(seq)
        pos_list.append(pos)
        nbr_list.append(nbr)
        nbr_iid_list.append(nbr_iid)

        meta_list.append(meta)
        max_distime_list.append(max_distime)
        min_distime_list.append(min_distime)
        mean_distime_list.append(mean_distime)
        mid_distime_list.append(mid_distime)
        suit_num_list.append(suit_num)

    max_distime_list = np.array(max_distime_list)
    min_distime_list = np.array(min_distime_list)
    mean_distime_list = np.array(mean_distime_list)
    mid_distime_list = np.array(mid_distime_list)
    suit_num = np.array(suit_num_list)

    print(f'max_distime max:{max_distime_list.max()} min{max_distime_list.min()} mean{max_distime_list.mean()}')
    print(f'min_distime max:{min_distime_list.max()} min{min_distime_list.min()} mean{min_distime_list.mean()}')
    print(f'mean_distime max:{mean_distime_list.max()} min{mean_distime_list.min()} mean{mean_distime_list.mean()}')
    print(f'mid_distime max:{mid_distime_list.max()} min{mid_distime_list.min()} mean{mid_distime_list.mean()}')
    print(f'suit_num max:{suit_num.max()} min{suit_num.min()} mean{suit_num.mean()}')
    print("avg meta len", np.array(check_meta_len).mean())
    # save as npz
    np.savez(
        f'datasets/{args.dataset}/time_limit{time_limit}processed_data.npz',
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        eval_users=np.array(eval_users, dtype=np.int32),
        valid_items=np.array(valid_items, dtype=np.int32),
        test_items=np.array(test_items, dtype=np.int32),
        train_uid=np.array(uid_list, dtype=np.int32),
        train_seq=np.array(seq_list, dtype=np.int32),
        train_pos=np.array(pos_list, dtype=np.int32),
        train_nbr=np.array(nbr_list, dtype=np.int32),
        train_meta=np.array(meta_list, dtype=np.int32),

        check_meta_len=np.array(check_meta_len, dtype=np.int32),
        train_nbr_iid=nbr_iid_list,
        meta=meta_list,
        max_distime_list=max_distime_list,
        min_distime_list=min_distime_list,
        mean_distime_list=mean_distime_list,
        mid_distime_list=mid_distime_list,
        suit_num_list=suit_num_list

    )

    print(f'saved at datasets/{args.dataset}/time_limit{time_limit}processed_data.npz')


def preprocess():
    load_and_save_epinions()  # ???logs??????npz

    # 1. ????????????????????????id
    filter_and_reid(prepro='origin', level=None)  # ??????item???user -> ????????????uid -> ???u2u????????????item???user
    num_user, num_item = delete_isolated_user()  # ?????????????????? -> ????????????uid -> ??????u2u???u2b???uid
    print('num_user', num_user)
    print('num_item', num_item)
    tmp = np.load(f'datasets/{args.dataset}/user_item_num.npy')
    num_user, num_item = tmp[0], tmp[1]
    num_user = num_user + 1
    num_item = num_item + 1

    # 2. ??????????????????
    gen_and_save_u2u_dict_and_split(num_user, num_item)
    gen_and_save_all_user_batches(num_user, num_item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='Epinions')
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50)
    parser.add_argument('--nbr_maxlen', type=int, default=20)
    parser.add_argument('--meta_maxlen', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_limit', type=int, default=5000000)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    seq_maxlen = args.seq_maxlen
    nbr_maxlen = args.nbr_maxlen
    meta_maxlen = args.meta_maxlen
    time_limit = args.time_limit

    preprocess()

    '''
    st = time()
    data = np.load(f'datasets/{args.dataset}/processed_data.npz', allow_pickle=True)

    user_train = data['user_train'][()]
    user_valid = data['user_valid'][()]
    user_test = data['user_test'][()]

    eval_users = data['eval_users']
    valid_items = data['valid_items']
    test_items = data['test_items']

    # idx = 0, uid = 1
    train_uid = data['train_uid']
    train_seq = data['train_seq']
    train_pos = data['train_pos']
    train_nbr = data['train_nbr']
    train_nbr_iid = data['train_nbr_iid']

    print('load time =', time()-st)    

    '''
