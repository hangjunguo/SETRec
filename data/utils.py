import numpy as np

from preprocess import ProcessedDataset


def dataset_split(dataset: ProcessedDataset, config=None, eval_args=None):
    """Split interaction records by leave one out strategy.

    Returns:
        list: List of :class:`~ProcessedDataset`, whose interaction features has been split.
    """
    # eval_args = config['eval_args']
    eval_args = eval_args

    if eval_args['split'] == 'LOO':
        dataset.sort(by=dataset.time_field)

        # grouped index
        group_by_list = dataset.inter_feat[config['USER_ID_FIELD']].numpy()
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        grouped_inter_feat_index = index.values()

        # split index by leave one out
        leave_one_num = 2  # valid and test
        next_index = [[] for _ in range(leave_one_num + 1)]
        for index in grouped_inter_feat_index:
            index = list(index)
            tot_cnt = len(index)
            legal_leave_one_num = min(leave_one_num, tot_cnt - 1)
            pr = tot_cnt - legal_leave_one_num
            next_index[0].extend(index[:pr])
            for i in range(legal_leave_one_num):
                next_index[-legal_leave_one_num + i].append(index[pr])
                pr += 1
        next_df = [dataset.inter_feat[index] for index in next_index]
        next_ds = [dataset.copy(_) for _ in next_df]
        return next_ds
    elif eval_args['split'] == 'RATIO':
        dataset.shuffle()

        ratios = eval_args['ratios']
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]
        group_by = eval_args['group_by']
        cumdata = eval_args['cumdata']

        if not group_by:
            tot_cnt = len(dataset)
            split_ids = calc_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [range(start, end) for start, end in zip([0] + split_ids, split_ids + [tot_cnt])]
        else:
            grouped_inter_feat_index = group_index(dataset.inter_feat[config['USER_ID_FIELD']].numpy())
            next_index = [[] for _ in range(len(ratios))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = calc_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start:end])

        if cumdata:
            cum_index = [range(0, end[-1] + 1) for end in next_index]
            next_df = [dataset.inter_feat[index] for index in cum_index]
        else:
            next_df = [dataset.inter_feat[index] for index in next_index]
        next_ds = [dataset.copy(_) for _ in next_df]
        return next_ds
    else:
        raise ValueError('Split strategy must be either `LOO` or `RATIO`.')


def calc_split_ids(tot, ratios):
    cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
    cnt[0] = tot - sum(cnt[1:])
    for i in range(1, len(ratios)):
        if cnt[0] <= 1:
            break
        if 0 < ratios[-i] * tot < 1:
            cnt[-i] += 1
            cnt[0] -= 1
    split_ids = np.cumsum(cnt)[:-1]
    return list(split_ids)


def group_index(group_by_list):
    index = {}
    for i, key in enumerate(group_by_list):
        if key not in index:
            index[key] = [i]
        else:
            index[key].append(i)
    return index.values()
