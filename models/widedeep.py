"""
Reference:
    Heng-Tze Cheng et al. "Wide & Deep Learning for Recommender Systems." in RecSys 2016.
"""

from torch.nn.init import xavier_normal_, constant_
import torch.nn as nn


from models.base_recommender import ContextRecommender
from models.sublayers import MLPLayers


class WideDeep(ContextRecommender):

    def __init__(self, config, dataset, embedding_size, mlp_hidden_size, dropout_prob):
        super(WideDeep, self).__init__(config, dataset, embedding_size)

        # load parameters info
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout_prob = dropout_prob

        # define layers and loss
        size_list = [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        widedeep_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        batch_size = widedeep_all_embeddings.shape[0]
        fm_output = self.first_order_linear(interaction)

        deep_output = self.deep_predict_layer(self.mlp_layers(widedeep_all_embeddings.view(batch_size, -1)))
        output = self.sigmoid(fm_output + deep_output)
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
