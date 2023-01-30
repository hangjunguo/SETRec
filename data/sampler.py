from collections import Counter

import numpy as np
from numpy.random import sample
import torch


class Sampler(object):
    def __init__(self, dataset, popularity=0):
        self.dataset = dataset

        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field

        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        self.distribution = None
        self.popularity = popularity
        self.used_ids = self.get_used_ids()
        self._build_alias_table()

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].numpy())

    def _build_alias_table(self):
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        for k in self.prob:
            self.prob[k] **= self.popularity
        self.alias = self.prob.copy()
        large_q = []
        small_q = []

        for i in self.prob:
            self.alias[i] = -1
            self.prob[i] = self.prob[i] / sum(self.prob.values()) * len(self.prob)
            if self.prob[i] > 1:
                large_q.append(i)
            elif self.prob[i] < 1:
                small_q.append(i)

        while len(large_q) != 0 and len(small_q) != 0:
            l = large_q.pop(0)
            s = small_q.pop(0)
            self.alias[s] = l
            self.prob[l] = self.prob[l] - (1 - self.prob[s])
            if self.prob[l] < 1:
                small_q.append(l)
            elif self.prob[l] > 1:
                large_q.append(l)

    def _pop_sampling(self, sample_num):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def sampling(self, sample_num):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if self.popularity == 0:
            return self._uni_sampling(sample_num)
        elif self.popularity > 0:
            return self._pop_sampling(sample_num)
        else:
            raise NotImplementedError(f'The sampling distribution [{self.distribution}] is not implemented.')

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used item_ids is the same as observed item_ids, index is user_id
            and element is a set of item_ids.
        """
        used_item_id = np.array([set() for _ in range(self.user_num)])
        for uid, iid in zip(self.dataset.inter_feat[self.uid_field].numpy(), self.dataset.inter_feat[self.iid_field].numpy()):
            used_item_id[uid].add(iid)

        for used_item_set in used_item_id:
            if len(used_item_set) + 1 == self.item_num:  # [pad] is a item.
                raise ValueError(
                    'Some users have interacted with all items, '
                    'which we can not sample negative items for them. '
                    'Please set `user_inter_num_interval` to filter those users.'
                )
        return used_item_id

    def sample_by_key_ids(self, key_ids, num):
        """Sampling by key_ids.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list))
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list))
                check_list = np.array([
                    i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                    if v in used
                ])
        return torch.tensor(value_ids)

    def sample_by_user_ids(self, user_ids, num):
        """Sampling by user_ids.
        """
        try:
            return self.sample_by_key_ids(user_ids, num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')
