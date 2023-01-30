import logging
import os
import colorlog
import re

from utils.utils import ensure_dir, get_local_time
from colorama import init

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


class RemoveColorFilter(logging.Filter):

    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    """
    init(autoreset=True)
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    dir_name = os.path.join(dir_name, config['dataset'])
    ensure_dir(dir_name)
    dir_name = os.path.join(dir_name, config['model'])
    ensure_dir(dir_name)
    ps = '_'.join(list(map(lambda x: str(x), config['pseudo_sample_nums'])))
    lr = '_'.join(list(map(lambda x: str(x), config['learning_rate'])))
    eb = '_'.join(list(map(lambda x: str(x), config['embedding_size'])))
    ly = '_'.join(list(map(lambda x: str(x), config['mlp_hidden_size'][0])))
    dp = '_'.join(list(map(lambda x: str(x), config['dropout'])))
    logfilename = '{}/{}/ps{}-bs{}-lr{}-eb{}-ly{}-dp{}-{}.log'.format(
        config['dataset'], config['model'], ps, config['train_batch_size'], lr, eb, ly, dp, get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])
