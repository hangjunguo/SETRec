import os
import copy
from logging import getLogger
from collections import Counter
import pickle
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
# import matplotlib.pyplot as plt

from data.dataset import Interaction
from enum import Enum


class ProcessedDataset(object):
    def __init__(self, config: dict):  # required modify
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()
        self._from_scratch()

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """

        self._preset()
        self._get_field_from_config()
        self._load_data(self.dataset_name, self.dataset_path)
        self._data_processing()
        self._change_feat_format()

    def _preset(self):
        """Initialization useful inside attributes.
        """
        self.dataset_path = self.config['data_path']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2token_id = {}
        self.field2seqlen = {}
        self._preloaded_weight = {}

    def _get_field_from_config(self):
        """Initialization common field names.
        """
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        if (self.uid_field is None) ^ (self.iid_field is None):
            raise ValueError(
                'USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time.'
            )

    def _data_processing(self):
        """Data preprocessing, including:

        - Data filtering
        - Remap ID
        - Missing value imputation
        - Normalization
        - Preloading weights initialization
        """
        self.feat_name_list = [
            feat_name for feat_name in ['inter_feat', 'user_feat', 'item_feat']
            if getattr(self, feat_name, None) is not None
        ]
        self._data_filtering()

        self._remap_ID_all()
        self._user_item_feat_preparation()
        self._fill_nan()
        self._set_label_by_threshold()

    def _data_filtering(self):
        """Data filtering

        - Filter missing user_id or item_id
        - Remove duplicated user-item interaction
        - Remove interaction by user or item
        - K-core data filtering
        """
        self._filter_nan_user_or_item()
        # self._remove_duplication()
        self._filter_inter_by_user_or_item()
        self._filter_by_inter_num()
        self._reset_index()

    def _load_data(self, dataset_name, dataset_path):
        """Load features.

        Firstly load interaction features, then user/item features optionally,
        finally load additional features if ``config['additional_feat_suffix']`` is set.

        Args:
            dataset_name (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        self._load_inter_feat(dataset_name, dataset_path)
        self.user_feat = self._load_user_or_item_feat(dataset_name, dataset_path, FeatureSource.USER, 'uid_field')
        self.item_feat = self._load_user_or_item_feat(dataset_name, dataset_path, FeatureSource.ITEM, 'iid_field')

    def _load_inter_feat(self, token, dataset_path):
        """Load interaction features.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        inter_feat_path = os.path.join(dataset_path, f'{token}.inter')
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f'File {inter_feat_path} not exist.')

        inter_feat = self._load_feat(inter_feat_path, FeatureSource.INTERACTION)
        self.logger.debug(f'Interaction feature loaded successfully from [{inter_feat_path}].')
        self.inter_feat = inter_feat

    def _load_user_or_item_feat(self, token, dataset_path, source, field_name):
        """Load user/item features.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
            source (FeatureSource): source of user/item feature.
            field_name (str): ``uid_field`` or ``iid_field``

        Returns:
            pandas.DataFrame: Loaded feature
        """
        feat_path = os.path.join(dataset_path, f'{token}.{source.value}')
        if os.path.isfile(feat_path):
            feat = self._load_feat(feat_path, source)
            self.logger.debug(f'[{source.value}] feature loaded successfully from [{feat_path}].')
        else:
            feat = None
            self.logger.debug(f'[{feat_path}] not found, [{source.value}] features are not loaded.')

        field = getattr(self, field_name, None)
        if feat is not None and field is None:
            raise ValueError(f'{field_name} must be exist if {source.value}_feat exist.')
        if feat is not None and field not in feat:
            raise ValueError(f'{field_name} must be loaded if {source.value}_feat is loaded.')

        if field in self.field2source:
            self.field2source[field] = FeatureSource(source.value + '_id')
        return feat

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature
        """
        self.logger.debug(f'Loading feature from [{filepath}] (source: [{source}]).')

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = '\t'
        columns = []
        usecols = []
        dtype = {}
        with open(filepath, 'r') as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(':')
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f'Type {ftype} from field {field} is not supported.')
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != 'link':
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith('seq'):
                    self.field2seqlen[field] = 1
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f'No columns has been loaded from [{source}]')
            return None

        df = pd.read_csv(filepath, delimiter=field_separator, usecols=usecols, dtype=dtype)
        df.columns = columns

        seq_separator = " "
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith('seq'):
                continue
            df[field].fillna(value='', inplace=True)
            if ftype == FeatureType.TOKEN_SEQ:
                df[field] = [list(filter(None, _.split(seq_separator))) for _ in df[field].values]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [list(map(float, filter(None, _.split(seq_separator)))) for _ in df[field].values]
            self.field2seqlen[field] = max(map(len, df[field].values))

        return df

    def _get_load_and_unload_col(self, source):
        if isinstance(source, FeatureSource):
            source = source.value
        if self.config['load_col'] is None:
            load_col = None
        elif source not in self.config['load_col']:
            load_col = set()
        elif self.config['load_col'][source] == '*':
            load_col = None
        else:
            load_col = set(self.config['load_col'][source])

        if self.config['unload_col'] is not None and source in self.config['unload_col']:
            unload_col = set(self.config['unload_col'][source])
        else:
            unload_col = None

        if load_col and unload_col:
            raise ValueError(f'load_col [{load_col}] and unload_col [{unload_col}] can not be set the same time.')

        self.logger.debug(f'[{source}]: ')
        self.logger.debug(f'\t load_col: [{load_col}]')
        self.logger.debug(f'\t unload_col: [{unload_col}]')

        return load_col, unload_col

    def _filter_nan_user_or_item(self):
        """Filter NaN user_id and item_id
        """
        for field, name in zip([self.uid_field, self.iid_field], ['user', 'item']):
            feat = getattr(self, name + '_feat')
            if feat is not None:
                dropped_feat = feat.index[feat[field].isnull()]
                if len(dropped_feat):
                    self.logger.warning(
                        f'In {name}_feat, line {list(dropped_feat + 2)}, {field} do not exist, so they will be removed.'
                    )
                    feat.drop(feat.index[dropped_feat], inplace=True)
            if field is not None:
                dropped_inter = self.inter_feat.index[self.inter_feat[field].isnull()]
                if len(dropped_inter):
                    self.logger.warning(
                        f'In inter_feat, line {list(dropped_inter + 2)}, {field} do not exist, so they will be removed.'
                    )
                    self.inter_feat.drop(self.inter_feat.index[dropped_inter], inplace=True)

    def _filter_inter_by_user_or_item(self):
        """Remove interaction in inter_feat which user or item is not in user_feat or item_feat.
        """
        remained_inter = pd.Series(True, index=self.inter_feat.index)

        if self.user_feat is not None:
            remained_uids = self.user_feat[self.uid_field].values
            remained_inter &= self.inter_feat[self.uid_field].isin(remained_uids)

        if self.item_feat is not None:
            remained_iids = self.item_feat[self.iid_field].values
            remained_inter &= self.inter_feat[self.iid_field].isin(remained_iids)

        self.inter_feat.drop(self.inter_feat.index[~remained_inter], inplace=True)

    def _filter_by_inter_num(self):
        if self.uid_field is None or self.iid_field is None:
            return

        inter_num_filter = self.config['inter_num_filter']
        max_user_inter_num = inter_num_filter['max_user_inter_num']
        min_user_inter_num = inter_num_filter['min_user_inter_num']
        max_item_inter_num = inter_num_filter['max_item_inter_num']
        min_item_inter_num = inter_num_filter['min_item_inter_num']

        if max_user_inter_num is None and min_user_inter_num is None:
            user_inter_num = Counter()
        else:
            user_inter_num = Counter(self.inter_feat[self.uid_field].values)

        if max_item_inter_num is None and min_item_inter_num is None:
            item_inter_num = Counter()
        else:
            item_inter_num = Counter(self.inter_feat[self.iid_field].values)

        while True:
            ban_users = self._get_illegal_ids_by_inter_num(
                field=self.uid_field,
                feat=self.user_feat,
                inter_num=user_inter_num,
                max_num=max_user_inter_num,
                min_num=min_user_inter_num
            )
            ban_items = self._get_illegal_ids_by_inter_num(
                field=self.iid_field,
                feat=self.item_feat,
                inter_num=item_inter_num,
                max_num=max_item_inter_num,
                min_num=min_item_inter_num
            )

            if len(ban_users) == 0 and len(ban_items) == 0:
                break

            if self.user_feat is not None:
                dropped_user = self.user_feat[self.uid_field].isin(ban_users)
                self.user_feat.drop(self.user_feat.index[dropped_user], inplace=True)

            if self.item_feat is not None:
                dropped_item = self.item_feat[self.iid_field].isin(ban_items)
                self.item_feat.drop(self.item_feat.index[dropped_item], inplace=True)

            dropped_inter = pd.Series(False, index=self.inter_feat.index)
            user_inter = self.inter_feat[self.uid_field]
            item_inter = self.inter_feat[self.iid_field]
            dropped_inter |= user_inter.isin(ban_users)
            dropped_inter |= item_inter.isin(ban_items)

            user_inter_num -= Counter(user_inter[dropped_inter].values)
            item_inter_num -= Counter(item_inter[dropped_inter].values)

            dropped_index = self.inter_feat.index[dropped_inter]
            self.logger.debug(f'[{len(dropped_index)}] dropped interactions.')
            self.inter_feat.drop(dropped_index, inplace=True)

    def _get_illegal_ids_by_inter_num(self, field, feat, inter_num, max_num=None, min_num=None):
        """Given inter feat, return illegal ids, whose inter num out of [min_num, max_num]

        Args:
            field (str): field name of user_id or item_id.
            feat (pandas.DataFrame): interaction feature.
            inter_num (Counter): interaction number counter.
            max_num (int, optional): max number of interaction. Defaults to ``None``.
            min_num (int, optional): min number of interaction. Defaults to ``None``.

        Returns:
            set: illegal ids, whose inter num out of [min_num, max_num]
        """
        self.logger.debug(f'get_illegal_ids_by_inter_num: field=[{field}], max_num=[{max_num}], min_num=[{min_num}]')

        if max_num is None:
            max_num = np.inf
        if min_num is None:
            min_num = -1

        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}

        if feat is not None:
            for id_ in feat[field].values:
                if inter_num[id_] < min_num:
                    ids.add(id_)
        self.logger.debug(f'[{len(ids)}] illegal_ids_by_inter_num, field=[{field}]')

        return ids

    def _reset_index(self):
        """Reset index for all feats in :attr:`feat_name_list`.
        """
        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            if feat.empty:
                raise ValueError('Some feat is empty, please check the filtering settings.')
            feat.reset_index(drop=True, inplace=True)

    def _remap_ID_all(self):
        """Get ``config['fields_in_same_space']`` firstly, and remap each.
        """
        fields_in_same_space = self._get_fields_in_same_space()
        self.logger.debug(f'fields_in_same_space: {fields_in_same_space}')
        for field_set in fields_in_same_space:
            remap_list = self._get_remap_list(field_set)
            self._remap(remap_list)

    def _get_fields_in_same_space(self):
        fields_in_same_space = []
        fields_in_same_space = [set(_) for _ in fields_in_same_space]
        additional = []
        token_like_fields = self.token_like_fields
        for field in token_like_fields:
            count = 0
            for field_set in fields_in_same_space:
                if field in field_set:
                    count += 1
            if count == 0:
                additional.append({field})
            elif count == 1:
                continue
            else:
                raise ValueError(f'Field [{field}] occurred in `fields_in_same_space` more than one time.')

        for field_set in fields_in_same_space:
            if self.uid_field in field_set and self.iid_field in field_set:
                raise ValueError('uid_field and iid_field can\'t in the same ID space')
            for field in field_set:
                if field not in token_like_fields:
                    raise ValueError(f'Field [{field}] is not a token-like field.')

        fields_in_same_space.extend(additional)

        return fields_in_same_space

    def _get_remap_list(self, field_set):
        """Transfer set of fields in the same remapping space into remap list.

        If ``uid_field`` or ``iid_field`` in ``field_set``,
        field in :attr:`inter_feat` will be remapped firstly,
        then field in :attr:`user_feat` or :attr:`item_feat` will be remapped next, finally others.
        """
        remap_list = []
        for field, feat in zip([self.uid_field, self.iid_field], [self.user_feat, self.item_feat]):
            if field in field_set:
                field_set.remove(field)
                remap_list.append((self.inter_feat, field, FeatureType.TOKEN))
                if feat is not None:
                    remap_list.append((feat, field, FeatureType.TOKEN))
        for field in field_set:
            source = self.field2source[field]
            if isinstance(source, FeatureSource):
                source = source.value
            feat = getattr(self, f'{source}_feat')
            ftype = self.field2type[field]
            remap_list.append((feat, field, ftype))
        return remap_list

    def _remap(self, remap_list):
        """Remap tokens using :meth:`pandas.factorize`.
        """
        tokens, split_point = self._concat_remaped_tokens(remap_list)
        new_ids_list, mp = pd.factorize(tokens)
        new_ids_list = np.split(new_ids_list + 1, split_point)
        mp = np.array(['[PAD]'] + list(mp))
        token_id = {t: i for i, t in enumerate(mp)}

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = mp
                self.field2token_id[field] = token_id
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    def _concat_remaped_tokens(self, remap_list):
        """Given ``remap_list``, concatenate values in order.

        Args:
            remap_list (list)

        Returns:
            tuple: tuple of:
            - tokens after concatenation.
            - split points that can be used to restore the concatenated tokens.
        """
        tokens = []
        for feat, field, ftype in remap_list:
            if ftype == FeatureType.TOKEN:
                tokens.append(feat[field].values)
            elif ftype == FeatureType.TOKEN_SEQ:
                tokens.append(feat[field].agg(np.concatenate))
        split_point = np.cumsum(list(map(len, tokens)))[:-1]
        tokens = np.concatenate(tokens)
        return tokens, split_point

    def _user_item_feat_preparation(self):
        """Sort :attr:`user_feat` and :attr:`item_feat` by ``user_id`` or ``item_id``.
        Missing values will be filled later.
        """
        if self.user_feat is not None:
            new_user_df = pd.DataFrame({self.uid_field: np.arange(self.user_num)})
            self.user_feat = pd.merge(new_user_df, self.user_feat, on=self.uid_field, how='left')
            self.logger.debug('ordering user features by user id.')
        if self.item_feat is not None:
            new_item_df = pd.DataFrame({self.iid_field: np.arange(self.item_num)})
            self.item_feat = pd.merge(new_item_df, self.item_feat, on=self.iid_field, how='left')
            self.logger.debug('ordering item features by user id.')

    def _fill_nan(self):
        """Missing value imputation.
        """
        self.logger.debug('Filling nan')

        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            for field in feat:
                ftype = self.field2type[field]
                if ftype == FeatureType.TOKEN:
                    feat[field].fillna(value=0, inplace=True)
                elif ftype == FeatureType.FLOAT:
                    feat[field].fillna(value=feat[field].mean(), inplace=True)
                else:
                    feat[field] = feat[field].apply(lambda x: [] if isinstance(x, float) else x)

    def _set_label_by_threshold(self):
        """Generate 0/1 labels according to value of features.

        According to ``config['threshold']``, those rows with value lower than threshold will
        be given negative label, while the other will be given positive label.
        """
        threshold = self.config['threshold']
        if threshold is None:
            # add the label column for implicit dataset
            self.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
            self.inter_feat[self.label_field] = [1] * len(self.inter_feat)
            return

        self.logger.debug(f'Set label by {threshold}.')

        if len(threshold) != 1:
            raise ValueError('Threshold length should be 1.')

        self.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        for field, value in threshold.items():
            if field in self.inter_feat:
                self.inter_feat[self.label_field] = (self.inter_feat[field] >= value).astype(int)
            else:
                raise ValueError(f'Field [{field}] not in inter_feat.')
            if field != self.label_field:
                self._del_col(self.inter_feat, field)

    def set_field_property(self, field, field_type, field_source, field_seqlen):
        """Set a new field's properties.

        Args:
            field (str): Name of the new field.
            field_type (FeatureType): Type of the new field.
            field_source (FeatureSource): Source of the new field.
            field_seqlen (int): max length of the sequence in ``field``.
                ``1`` if ``field``'s type is not sequence-like.
        """
        self.field2type[field] = field_type
        self.field2source[field] = field_source
        self.field2seqlen[field] = field_seqlen

    def _del_col(self, feat, field):
        """Delete columns

        Args:
            feat (pandas.DataFrame or Interaction): the feat contains field.
            field (str): field name to be dropped.
        """
        self.logger.debug(f'Delete column [{field}].')
        if isinstance(feat, Interaction):
            feat.drop(column=field)
        else:
            feat.drop(columns=field, inplace=True)
        for dct in [self.field2id_token, self.field2token_id, self.field2seqlen, self.field2source, self.field2type]:
            if field in dct:
                del dct[field]

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`.
        """
        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            setattr(self, feat_name, self._dataframe_to_interaction(feat))

    def fields(self, ftype=None, source=None):
        """Given type and source of features, return all the field name of this type and source.
        If ``ftype == None``, the type of returned fields is not restricted.
        If ``source == None``, the source of returned fields is not restricted.

        Args:
            ftype (FeatureType, optional): Type of features. Defaults to ``None``.
            source (FeatureSource, optional): Source of features. Defaults to ``None``.

        Returns:
            list: List of field names.
        """
        ftype = set(ftype) if ftype is not None else set(FeatureType)
        source = set(source) if source is not None else set(FeatureSource)
        ret = []
        for field in self.field2type:
            tp = self.field2type[field]
            src = self.field2source[field]
            if tp in ftype and src in source:
                ret.append(field)
        return ret

    @property
    def float_like_fields(self):
        """
        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.FLOAT, FeatureType.FLOAT_SEQ])

    @property
    def token_like_fields(self):
        """
        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.TOKEN, FeatureType.TOKEN_SEQ])

    @property
    def user_num(self):
        """Get the number of different tokens of ``self.uid_field``.

        Returns:
            int: Number of different tokens of ``self.uid_field``.
        """
        self._check_field('uid_field')
        return self.num(self.uid_field)

    @property
    def item_num(self):
        """Get the number of different tokens of ``self.iid_field``.

        Returns:
            int: Number of different tokens of ``self.iid_field``.
        """
        self._check_field('iid_field')
        return self.num(self.iid_field)

    def divide_user(self):
        """Divide users according to inter number in train split.

        Returns:
            dict: Key is quantile, value is user group.
        """
        user_division = dict()
        user_inter_num = Counter(self.inter_feat[self.uid_field].numpy())
        inter_quantile = [0]
        for i in range(4):
            inter_quantile.append(np.quantile(list(user_inter_num.values()), (i+1) * 0.25))
            user_ids = [id_ for id_ in user_inter_num if user_inter_num[id_] <= inter_quantile[i+1]]
            user_division[inter_quantile[i+1]] = user_ids

        return user_division

    def _check_field(self, *field_names):
        """Given a name of attribute, check if it's exist.

        Args:
            *field_names (str): Fields to be checked.
        """
        for field_name in field_names:
            if getattr(self, field_name, None) is None:
                raise ValueError(f'{field_name} isn\'t set.')

    def num(self, field):
        """Given ``field``, for token-like fields, return the number of different tokens after remapping,
        for float-like fields, return ``1``.

        Args:
            field (str): field name to get token number.

        Returns:
            int: The number of different tokens (``1`` if ``field`` is a float-like field).
        """
        if field not in self.field2type:
            raise ValueError(f'Field [{field}] not defined in dataset.')
        if self.field2type[field] not in {FeatureType.TOKEN, FeatureType.TOKEN_SEQ}:
            return self.field2seqlen[field]
        else:
            return len(self.field2id_token[field])

    def join(self, df):
        """Given interaction feature, join user/item feature into it.

        Args:
            df (Interaction): Interaction feature to be joint.

        Returns:
            Interaction: Interaction feature after joining operation.
        """
        if self.user_feat is not None and self.uid_field in df:
            df.update(self.user_feat[df[self.uid_field]])
        if self.item_feat is not None and self.iid_field in df:
            df.update(self.item_feat[df[self.iid_field]])
        return df

    def sort(self, by, ascending=True):
        """Sort the interaction records inplace.

        Args:
            by (str or list of str): Field that as the key in the sorting process.
            ascending (bool or list of bool, optional): Results are ascending if ``True``, otherwise descending.
                Defaults to ``True``
        """
        self.inter_feat.sort(by=by, ascending=ascending)

    def shuffle(self):
        """Shuffle the interaction inplace.
        """
        self.inter_feat.shuffle()

    def __getitem__(self, index, join=True):
        df = self.inter_feat[index]
        return self.join(df) if join else df

    def __len__(self):
        return len(self.inter_feat)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def _dataframe_to_interaction(self, data):
        """Convert :class:`pandas.DataFrame` to :class:`Interaction`.
        """
        new_data = {}
        for k in data:
            value = data[k].values
            ftype = self.field2type[k]
            if ftype == FeatureType.TOKEN:
                new_data[k] = torch.LongTensor(value)
            elif ftype == FeatureType.FLOAT:
                new_data[k] = torch.FloatTensor(value)
            elif ftype == FeatureType.TOKEN_SEQ:
                seq_data = [torch.LongTensor(d[:self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            elif ftype == FeatureType.FLOAT_SEQ:
                seq_data = [torch.FloatTensor(d[:self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
        return Interaction(new_data)


class FeatureType(Enum):
    """Type of features.
    """

    TOKEN = 'token'
    FLOAT = 'float'
    TOKEN_SEQ = 'token_seq'
    FLOAT_SEQ = 'float_seq'


class FeatureSource(Enum):
    """Source of features.
    """

    INTERACTION = 'inter'
    USER = 'user'
    ITEM = 'item'
    USER_ID = 'user_id'
    ITEM_ID = 'item_id'


def convert_rating():
    # Load Data
    file = "data/" + DATASET + "/" + RATING_FILE_NAME[DATASET]
    print("reading rating file ...")

    ratings = pd.read_table(file, sep=SEP[DATASET], header=None, engine='python')
    ratings.columns = (['user_index_old', 'item_index_old', 'rating', 'timestamp'])

    # Reindex
    print("converting rating file ...")
    user_id = ratings[['user_index_old']].drop_duplicates()
    num_user = len(user_id)
    user_id['user_index'] = np.arange(len(user_id))
    # user_index_old2new = dict(zip(user_id['user_index_old'], user_id['user_index']))
    ratings = pd.merge(ratings, user_id, on=['user_index_old'], how='left')
    item_id = ratings[['item_index_old']].drop_duplicates()
    num_item = len(item_id)
    item_id['item_index'] = np.arange(len(item_id))
    # item_index_old2new = dict(zip(item_id['item_index_old'], item_id['item_index']))
    ratings = pd.merge(ratings, item_id, on=['item_index_old'], how='left')
    ratings = ratings[['user_index', 'item_index', 'rating', 'timestamp']]
    print("number of users: %d" % num_user)
    print("number of items: %d" % num_item)

    # convert to binary
    ratings['rating'][ratings['rating'] > THRESHOLD[DATASET]] = 1.0
    ratings['rating'][ratings['rating'] <= THRESHOLD[DATASET]] = 0.0

    # sample candidates for evaluation
    user_pos_items = ratings[ratings['rating'] >= 1.0].groupby('user_index')['item_index'].apply(set).to_dict()
    all_items = set(range(num_item))
    user_unseen_items = {}  # dict: {user: set(unseen_items)}
    candidates = {}  # dict: {user: ndarray(candidate_items)}
    for u in range(num_user):
        user_unseen_items[u] = all_items - user_pos_items[u]
        candidates[u] = np.random.choice(list(user_unseen_items[u]), size=499, replace=False)
    # user_rating_counts = ratings.groupby('user_index').agg({'item_index': 'count'})

    # leave the last item for test
    test_idx = ratings.groupby('user_index')['timestamp'].idxmax()
    test_df = ratings.loc[test_idx, ['user_index', 'item_index', 'rating']]
    test_samples = dict(zip(test_df['user_index'], test_df['item_index']))  # dict: {test_user: test_item}

    # leave the second last item for validation
    ratings = ratings.drop(test_idx)
    valid_idx = ratings.groupby('user_index')['timestamp'].idxmax()
    valid_df = ratings.loc[valid_idx, ['user_index', 'item_index', 'rating']]
    valid_samples = dict(zip(valid_df['user_index'], valid_df['item_index']))  # dict: {valid_user: valid_item}

    # the rest for training
    train_df = ratings.drop(valid_idx)[['user_index', 'item_index', 'rating']]
    train_mat = utils.df_to_dict(train_df)
    train_interactions = list(zip(train_df['user_index'], train_df['item_index'], train_df['rating']))

    # save preprocessed data
    save_path = "data/" + DATASET + "/"
    save_objs = [(num_user, num_item), train_mat, train_interactions, valid_samples, test_samples, candidates]
    save_filenames = ["count", "train_mat", "train_interactions", "valid_samples", "test_samples", "candidates"]
    for i, filename in enumerate(save_filenames):
        utils.dump_pickle(save_path, filename + ".pkl", save_objs[i])
