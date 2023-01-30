import datetime
import os
import pickle
import logging
import re
import yaml
import random
import numpy as np
import torch

from data.dataloader import EvalDataLoader


def df_to_dict(df):
    result = {}
    for user in df['user_index'].drop_duplicates():
        df_user = df[df['user_index'] == user]
        result[user] = dict(zip(df_user['item_index'], df_user['rating']))

    return result


# def ls_to_dict(ls: list):
#     result = {}


def dump_pickle(path, filename, obj):
    with open(path + filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path, filename):
    with open(path + filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def read_samples(path, dataset):
    save_filenames = ["count", "train_mat", "train_interactions", "valid_samples", "test_samples"]
    (num_user, num_item) = load_pickle(path + dataset, "/count.pkl")

    train_mat = load_pickle(path + dataset, "/train_mat.pkl")
    train_interactions = load_pickle(path + dataset, "/train_interactions.pkl")

    valid_samples = load_pickle(path + dataset, "/valid_samples")
    test_samples = load_pickle(path + dataset, "/test_samples")

    return num_user, num_item, train_mat, train_interactions, valid_samples, test_samples


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def load_config_file(file):
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )

    config_dict = dict()
    if file:
        with open(file, 'r', encoding='utf-8') as f:
            config_dict.update(yaml.load(f.read(), Loader=yaml_loader))

    return config_dict


def cfg2str(config):
    args_info = '\n'
    args_info += '\n'.join([(set_color("{}", 'cyan') + " =" + set_color(" {}", 'yellow')).format(arg, value)
                           for arg, value in config.items()])
    args_info += '\n\n'
    return args_info


def create_logger(dir_label, log_paras, time_run, mode, rank):
    log_code = None
    if 'train' in mode or 'load' in mode:
        log_code = 'train'
    if 'test' in mode:
        log_code = 'test'

    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    Log_file = logging.getLogger('Log_file')
    Log_screen = logging.getLogger('Log_screen')

    if rank in [-1, 0]:
        log_path = os.path.join('./logs_' + dir_label + '_' + log_code)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file_name = os.path.join(log_path, 'log_' + log_paras + time_run + '.log')
        Log_file.setLevel(logging.INFO)
        Log_screen.setLevel(logging.INFO)

        th = logging.FileHandler(filename=log_file_name, encoding='utf-8')
        th.setLevel(logging.INFO)
        th.setFormatter(formatter)
        Log_file.addHandler(th)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)

        Log_screen.addHandler(handler)
        Log_file.addHandler(handler)
    else:
        Log_file.setLevel(logging.WARN)
        Log_screen.setLevel(logging.WARN)
    return Log_file, Log_screen


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether bigger is better
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    return '    '.join([str(metric) + ' : ' + str(value) for metric, value in result_dict.items()])


def test_by_user_group(config, train_dataset, test_dataset, sampler, trainer, logger, ckpt_file):
    user_division = train_dataset.divide_user()
    test_user_list = test_dataset.inter_feat[config['USER_ID_FIELD']].cpu().numpy()
    for qt, user_group in user_division.items():
        index = []
        for uid in user_group:
            uid_index = np.asarray(test_user_list == uid).nonzero()[0].tolist()
            index.extend(uid_index)
        next_df = test_dataset.inter_feat[index]
        next_ds = test_dataset.copy(next_df)
        test_data = EvalDataLoader(config, next_ds, sampler)
        test_result = trainer.evaluate(test_data, model_file=ckpt_file)
        logger.info(set_color('User < {} test result '.format(qt),
                              'yellow') + ': \n' + dict2str(test_result))
