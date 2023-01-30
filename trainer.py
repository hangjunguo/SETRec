import os
from logging import getLogger
from time import time
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

from utils.metrics import TopkMetrics
from utils.utils import ensure_dir, get_local_time, early_stopping, dict2str, set_color


class Trainer(object):
    r"""

    """

    def __init__(self, config, model, learning_rate, epochs):
        self.config = config
        self.model = model

        self.logger = getLogger()
        self.learning_rate = learning_rate  ##
        self.epochs = epochs  ##
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = True
        self.test_batch_size = config['eval_batch_size']
        self.device = config['gpu_id']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.best_model_state = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.evaluator = None
        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self):
        params = self.model.parameters()
        learning_rate = self.learning_rate
        weight_decay = self.weight_decay
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict, dict): best valid score and best valid result and best model state. If valid_data is None,
             it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # evaluate
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)

                if update_flag:
                    if saved:
                        self.best_model_state = {
                            'config': self.config,
                            'epoch': epoch_idx,
                            'cur_step': self.cur_step,
                            'best_valid_score': self.best_valid_score,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        self.logger.info(set_color('Current best model state saved', 'blue'))
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        return self.best_valid_score, self.best_valid_result, self.best_model_state

    def _train_epoch(self, train_data, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None

        for batch_idx, interaction in enumerate(train_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if torch.isnan(loss):
                raise ValueError('Training loss is nan')
            loss.backward()
            self.optimizer.step()
        return total_loss

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = valid_result[self.valid_metric]
        return valid_score, valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        self.tot_item_num = eval_data.dataset.item_num
        self.evaluator = TopkMetrics(self.config)

        for batch_idx, batched_data in enumerate(eval_data):
            interaction, scores, positive_u, positive_i = self._neg_sample_batch_eval(batched_data)
            self.evaluator.collect_batch_eval(scores, positive_u, positive_i)

        result_dict = OrderedDict()
        # result_dict.update(self.evaluator.recall())
        # result_dict.update(self.evaluator.precision())
        result_dict.update(self.evaluator.hit())
        result_dict.update(self.evaluator.mrr())
        result_dict.update(self.evaluator.ndcg())

        return result_dict

    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        origin_scores = self.model.predict(interaction.to(self.device))

        col_idx = interaction[self.config['ITEM_ID_FIELD']]
        batch_user_num = positive_u[-1] + 1
        scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
        scores[row_idx, col_idx] = origin_scores
        return interaction, scores, positive_u, positive_i

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        saved_model_file = kwargs.pop('saved_model_file', self.saved_model_file)
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)
        if verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {saved_model_file}')
