import torch

from models.base_recommender import GeneralRecommender


class Pop(GeneralRecommender):
    r"""Top Popular model always recommends the most popular item.

    """

    def __init__(self, config, dataset):
        super(Pop, self).__init__(config, dataset)

        self.item_cnt = torch.zeros(self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False)
        self.max_cnt = None
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ['item_cnt', 'max_cnt']

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        item = interaction[self.ITEM_ID]
        self.item_cnt[item, :] = self.item_cnt[item, :] + 1

        self.max_cnt = torch.max(self.item_cnt, dim=0)[0]

        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        result = torch.true_divide(self.item_cnt[item, :], self.max_cnt)
        return result.squeeze(-1)

    def full_sort_predict(self, interaction):
        batch_user_num = interaction[self.USER_ID].shape[0]
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
        result = torch.repeat_interleave(result.unsqueeze(0), batch_user_num, dim=0)
        return result.view(-1)
