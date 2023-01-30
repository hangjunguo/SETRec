import math
from logging import getLogger

import torch
import numpy as np

from data.dataset import Interaction, cat_interactions
from preprocess import FeatureType, FeatureSource


class PseudoLabelDataloader:
    """:class:`PseudoLabelDataloader` is designed to sample negative and pseudo-label items by ratio.
        By now it mainly focus on one neg-sampling method, 1-by-multi neg-sampling (point wise).

        Args:
            config (Config): The config of dataloader.
            dataset (Dataset): The dataset of dataloader.
            sampler (Sampler): The sampler of dataloader.
            shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``True``.
        """
    def __init__(self, config, dataset, sampler, pseudo_sample_num=0, model=None, shuffle=True):
        self._set_pseudo_sample_args(config, dataset, pseudo_sample_num)
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.sampler = sampler
        self.model = model
        self.batch_size = self.step = None
        self.shuffle = shuffle
        self.pr = 0
        self._init_batch_size_and_step()

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        batch_num = max(batch_size // self.times, 1)
        new_batch_size = batch_num * self.times
        self.step = batch_num
        self.batch_size = new_batch_size

    def _set_pseudo_sample_args(self, config, dataset, pseudo_sample_num):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.times = 1
        self.neg_sample_num = config['neg_sampling']
        self.pseudo_sample_num = pseudo_sample_num
        self.pseudo_candidate_scope = config['pseudo_candidate_scope']

        # POINTWISE
        self.times = 1 + self.neg_sample_num + self.pseudo_sample_num
        # self.sampling_func = self._neg_sample_by_point_wise_sampling

        self.label_field = config['LABEL_FIELD']
        dataset.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)

    def _sampling(self, inter_feat):
        if self.pseudo_sample_num > 0:  # for training student
            user_ids = inter_feat[self.uid_field].numpy()
            # negative items sampling
            neg_item_ids = self.sampler.sample_by_user_ids(user_ids, self.neg_sample_num)
            neg_inter_feat = self._neg_sample_by_point_wise_sampling(inter_feat, neg_item_ids)
            # pseudo-label items sampling
            pseudo_candidate_ids = self.sampler.sample_by_user_ids(user_ids,
                                                                   self.pseudo_sample_num * self.pseudo_candidate_scope)
            pseudo_inter_feat = self._pseudo_sample_by_point_wise_sampling(inter_feat, pseudo_candidate_ids)
            return cat_interactions([inter_feat, neg_inter_feat, pseudo_inter_feat])
        elif self.pseudo_sample_num == 0:  # for training teacher
            user_ids = inter_feat[self.uid_field].numpy()
            neg_item_ids = self.sampler.sample_by_user_ids(user_ids, self.neg_sample_num)
            neg_inter_feat = self._neg_sample_by_point_wise_sampling(inter_feat, neg_item_ids)
            return cat_interactions([inter_feat, neg_inter_feat])
        else:
            return inter_feat

    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.neg_sample_num)
        new_data[self.iid_field] = neg_item_ids
        new_data = self.dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.neg_sample_num)
        new_data.update(Interaction({self.label_field: labels}))
        return new_data  # only negative interactions

    def _pseudo_sample_by_point_wise_sampling(self, inter_feat, pseudo_item_ids):
        self.model.eval()
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.pseudo_sample_num * self.pseudo_candidate_scope)
        new_data[self.iid_field] = pseudo_item_ids
        new_data = self.dataset.join(new_data).to(self.config['gpu_id'])
        soft_labels = self.model.predict(new_data).reshape(-1, pos_inter_num)

        pos_labels, pos_fid_indices = torch.topk(soft_labels, self.pseudo_sample_num, dim=0)  # [2, pos_inter_num]
        # neg_labels, neg_fid_indices = torch.topk(soft_labels, self.pseudo_sample_num // 2, dim=0, largest=False)
        col_indices = torch.arange(pos_inter_num).long().to(self.config['gpu_id'])
        pos_fid_indices = pos_fid_indices.detach() * pos_inter_num + col_indices  # [2, pos_inter_num]
        # neg_fid_indices = neg_fid_indices.detach() * pos_inter_num + col_indices
        # fid_indices = torch.cat((pos_fid_indices.reshape(-1), neg_fid_indices.reshape(-1)))  # [2*2*pos_inter_num]
        fid_indices = pos_fid_indices.reshape(-1)  # [2*2*pos_inter_num]
        # fid_labels = torch.cat((pos_labels.reshape(-1), neg_labels.reshape(-1)))
        fid_labels = pos_labels.reshape(-1)

        new_data = new_data[fid_indices]
        new_data.update(Interaction({self.label_field: fid_labels}))
        self.model.train()
        return new_data.cpu()  # only pseudo-labeled interactions

    def get_model(self, model):
        self.model = model

    @property
    def pr_end(self):
        return len(self.dataset)

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def _next_batch_data(self):
        cur_data = self._sampling(self.dataset[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data


class EvalDataLoader:
    """:class:`EvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`PseudoLabelDataloader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self._set_neg_sample_args(config, dataset)

        # init eval dataloader config
        user_num = dataset.user_num
        dataset.sort(by=dataset.uid_field, ascending=True)
        self.uid_list = []
        start, end = dict(), dict()
        for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
            if uid not in start:
                self.uid_list.append(uid)
                start[uid] = i
            end[uid] = i
        self.uid2index = np.array([None] * user_num)
        self.uid2items_num = np.zeros(user_num, dtype=np.int64)
        for uid in self.uid_list:
            self.uid2index[uid] = slice(start[uid], end[uid] + 1)
            self.uid2items_num[uid] = end[uid] - start[uid] + 1
        self.uid_list = np.array(self.uid_list)

        # init neg sampling config
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = self.step = self.model = None
        self.shuffle = shuffle
        self.pr = 0
        self._init_batch_size_and_step()

    def _set_neg_sample_args(self, config, dataset):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field

        self.neg_sample_num = config['eval_neg_sampling']
        self.times = 1 + self.neg_sample_num
        self.sampling_func = self._neg_sample_by_point_wise_sampling
        self.label_field = config['LABEL_FIELD']
        dataset.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)

    def _init_batch_size_and_step(self):
        batch_size = self.config['eval_batch_size']

        inters_num = sorted(self.uid2items_num * self.times, reverse=True)
        batch_num = 1
        new_batch_size = inters_num[0]
        for i in range(1, len(inters_num)):
            if new_batch_size + inters_num[i] > batch_size:
                break
            batch_num = i + 1
            new_batch_size += inters_num[i]
        self.step = batch_num
        self.batch_size = new_batch_size

    def _neg_sampling(self, inter_feat):
        user_ids = inter_feat[self.uid_field].numpy()
        item_ids = inter_feat[self.iid_field].numpy()
        neg_item_ids = self.sampler.sample_by_user_ids(user_ids, self.neg_sample_num)
        return self.sampling_func(inter_feat, neg_item_ids)

    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self.dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _next_batch_data(self):
        uid_list = self.uid_list[self.pr:self.pr + self.step]
        data_list = []
        idx_list = []
        positive_u = []
        positive_i = torch.tensor([], dtype=torch.int64)

        for idx, uid in enumerate(uid_list):
            index = self.uid2index[uid]
            data_list.append(self._neg_sampling(self.dataset[index]))
            idx_list += [idx for _ in range(self.uid2items_num[uid] * self.times)]
            positive_u += [idx for _ in range(self.uid2items_num[uid])]
            positive_i = torch.cat((positive_i, self.dataset[index][self.iid_field]), 0)

        cur_data = cat_interactions(data_list)
        idx_list = torch.from_numpy(np.array(idx_list))
        positive_u = torch.from_numpy(np.array(positive_u))

        self.pr += self.step

        return cur_data, idx_list, positive_u, positive_i

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()
