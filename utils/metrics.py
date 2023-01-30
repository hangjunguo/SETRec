import torch
import numpy as np


class TopkMetrics:
    r"""class TopkMetrics implements several common top-k metrics.

    """

    def __init__(self, config):
        self.topk = config['topk']
        self.pos_idx = self.pos_len_list = None

    def collect_batch_eval(self, scores: torch.Tensor, positive_u: torch.Tensor, positive_i: torch.Tensor):
        """ Collect batch evaluation results for calculating metrics.

        Args:
            scores (Torch.Tensor): the output tensor of model with the shape of `(n_users, )`
            positive_u(Torch.Tensor): the row index of positive items for each user.
            positive_i(Torch.Tensor): the positive item id for each user.
        """
        _, topk_idx = torch.topk(scores, max(self.topk), dim=-1)  # n_users x k
        pos_matrix = torch.zeros_like(scores, dtype=torch.int)
        pos_matrix[positive_u, positive_i] = 1
        batch_pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx).cpu().clone().detach()
        if self.pos_idx is None:
            self.pos_idx = batch_pos_idx
        else:
            self.pos_idx = torch.cat((self.pos_idx, batch_pos_idx), dim=0)
        batch_pos_len_list = pos_matrix.sum(dim=1).cpu().clone().detach()
        if self.pos_len_list is None:
            self.pos_len_list = batch_pos_len_list
        else:
            self.pos_len_list = torch.cat((self.pos_len_list, batch_pos_len_list), dim=0)
        # batch_user_list = torch.unique_consecutive(interaction[self.config['USER_ID_FIELD']])

    def hit(self):
        self._to_numpy()
        result = np.cumsum(self.pos_idx, axis=1)
        result = (result > 0).astype(int)
        metric_dict = self._metric_result('hit', result)
        return metric_dict

    def mrr(self):
        self._to_numpy()
        idxs = self.pos_idx.argmax(axis=1)
        result = np.zeros_like(self.pos_idx, dtype=np.float)
        for row, idx in enumerate(idxs):
            if self.pos_idx[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        metric_dict = self._metric_result('mrr', result)
        return metric_dict

    def ndcg(self):
        self._to_numpy()
        len_rank = np.full_like(self.pos_len_list, self.pos_idx.shape[1])
        idcg_len = np.where(self.pos_len_list > len_rank, len_rank, self.pos_len_list)

        iranks = np.zeros_like(self.pos_idx, dtype=np.float)
        iranks[:, :] = np.arange(1, self.pos_idx.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(self.pos_idx, dtype=np.float)
        ranks[:, :] = np.arange(1, self.pos_idx.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(self.pos_idx, dcg, 0), axis=1)

        result = dcg / idcg
        metric_dict = self._metric_result('ndcg', result)
        return metric_dict

    def precision(self):
        self._to_numpy()
        result = self.pos_idx.cumsum(axis=1) / np.arange(1, self.pos_idx.shape[1] + 1)
        metric_dict = self._metric_result('precision', result)
        return metric_dict

    def recall(self):
        self._to_numpy()
        result = np.cumsum(self.pos_idx, axis=1) / self.pos_len_list.reshape(-1, 1)
        metric_dict = self._metric_result('recall', result)
        return metric_dict

    def _to_numpy(self):
        if torch.is_tensor(self.pos_idx):
            self.pos_idx = self.pos_idx.to(torch.bool).numpy()
        if torch.is_tensor(self.pos_len_list):
            self.pos_len_list = self.pos_len_list.numpy()

    def _metric_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], 4)
        return metric_dict

    def reset(self):
        self.pos_idx = self.pos_len_list = None
