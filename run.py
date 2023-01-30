import os
from logging import getLogger

import torch

from utils.utils import init_seed, load_config_file, test_by_user_group, cfg2str, set_color, get_local_time, \
    dict2str, ensure_dir
from utils.logger import init_logger
from data.utils import dataset_split
from data.dataloader import PseudoLabelDataloader, EvalDataLoader
from data.sampler import Sampler
from preprocess import ProcessedDataset
from models.dssm import DSSM
from models.widedeep import WideDeep
from models.sasrec import SASRec
# from models.pop import Pop
from trainer import Trainer

# configurations initialization
config = load_config_file('foursquare_NYC.yaml')
model_dir = os.path.join(config['checkpoint_dir'], config['dataset'])
ensure_dir(model_dir)

# init random seed
init_seed(seed=999, reproducibility=True)

# logger initialization
init_logger(config)
logger = getLogger()

# write config info into log
logger.info(cfg2str(config))

# dataset creating and filtering
dataset = ProcessedDataset(config)

# dataset splitting
train_dataset, valid_dataset, test_dataset = dataset_split(dataset, config)

sampler_pop = Sampler(dataset, config['popularity'])
sampler_uni = Sampler(dataset)
valid_data, test_data = EvalDataLoader(config, valid_dataset, sampler_uni), \
                        EvalDataLoader(config, test_dataset, sampler_uni)

torch.cuda.set_device(config['gpu_id'])

for model_id in range(config['model_num']):
    # model para
    embedding_size = config['embedding_size'][model_id]
    mlp_hidden_size = config['mlp_hidden_size'][model_id]
    dropout = config['dropout'][model_id]
    pseudo_sample_num = config['pseudo_sample_nums'][model_id]
    # trainer para
    learning_rate = config['learning_rate'][model_id]
    epoch = config['epochs'][model_id]

    # model loading and initialization
    init_seed(config['seed'], True)

    if model_id == 0:
        ckpt_file = '{}-{}-ps{}-bs{}-lr{}-eb{}-ly{}-dp{}.pth'.format(
            config['model'], model_id, pseudo_sample_num, config['train_batch_size'], learning_rate,
            embedding_size, len(mlp_hidden_size), dropout)
        ckpt_file = os.path.join(model_dir, ckpt_file)
        teacher = DSSM(config, train_dataset, embedding_size, mlp_hidden_size, dropout).to(config['device'])
        # teacher = WideDeep(config, train_dataset, embedding_size, mlp_hidden_size, dropout).to(config['device'])
        # teacher = SASRec(config, train_dataset).to(config['device'])
        if not config['has_pretrained_teacher']:
            # train first teacher model
            train_data = PseudoLabelDataloader(config, train_dataset, sampler_pop)
            trainer = Trainer(config, teacher, learning_rate, epoch)
            best_valid_score, best_valid_result, best_model_state = trainer.fit(train_data, valid_data)
            torch.save(best_model_state, ckpt_file)
            test_result = trainer.evaluate(test_data, model_file=ckpt_file)
            logger.info(set_color('Teacher {} {} best valid '.format(config['model'], model_id),
                                  'yellow') + ': \n' + dict2str(best_valid_result))
            logger.info(set_color('Teacher {} {} test result '.format(config['model'], model_id),
                                  'yellow') + ': \n' + dict2str(test_result))
            test_by_user_group(config, train_dataset, test_dataset, sampler_uni, trainer, logger, ckpt_file)
    else:
        checkpoint = torch.load(ckpt_file)
        teacher.load_state_dict(checkpoint['state_dict'])
        model = DSSM(config, train_dataset, embedding_size, mlp_hidden_size, dropout).to(config['device'])
        logger.info(model)

        # with the help of teacher, student model
        train_data = PseudoLabelDataloader(config, train_dataset, sampler_pop, pseudo_sample_num, teacher)
        trainer = Trainer(config, model, learning_rate, epoch)
        best_valid_score, best_valid_result, best_model_state = trainer.fit(train_data, valid_data)
        ckpt_file = '{}-{}-ps{}-bs{}-lr{}-eb{}-ly{}-dp{}-{}.pth'.format(
            config['model'], model_id, pseudo_sample_num, config['train_batch_size'], learning_rate,
            embedding_size, len(mlp_hidden_size), dropout, get_local_time())
        ckpt_file = os.path.join(model_dir, ckpt_file)
        torch.save(best_model_state, ckpt_file)
        # model evaluation
        test_result = trainer.evaluate(test_data, model_file=ckpt_file)
        logger.info(set_color('Student {} {} best valid '.format(config['model'], model_id),
                              'yellow') + ': \n' + dict2str(best_valid_result))
        logger.info(set_color('Student {} {} test result '.format(config['model'], model_id),
                              'yellow') + ': \n' + dict2str(test_result))
        test_by_user_group(config, train_dataset, test_dataset, sampler_uni, trainer, logger, ckpt_file)
        teacher = model

        # without the help of teacher, control model
        model = DSSM(config, train_dataset, embedding_size, mlp_hidden_size, dropout).to(config['device'])
        ctrl_train_data = PseudoLabelDataloader(config, train_dataset, sampler_uni)
        ctrl_trainer = Trainer(config, model, learning_rate, epoch)
        best_valid_score, best_valid_result, best_model_state = ctrl_trainer.fit(ctrl_train_data, valid_data)
        ctrl_ckpt_file = 'ctrl-{}-{}-{}.pth'.format(config['model'], model_id, get_local_time())
        ctrl_ckpt_file = os.path.join(model_dir, ctrl_ckpt_file)
        torch.save(best_model_state, ctrl_ckpt_file)
        # model evaluation
        test_result = ctrl_trainer.evaluate(test_data, model_file=ctrl_ckpt_file)
        logger.info(set_color('Control {} {} best valid '.format(config['model'], model_id),
                              'yellow') + ': \n' + dict2str(best_valid_result))
        logger.info(set_color('Control {} {} test result '.format(config['model'], model_id),
                              'yellow') + ': \n' + dict2str(test_result))
        test_by_user_group(config, train_dataset, test_dataset, sampler_uni, ctrl_trainer, logger, ctrl_ckpt_file)
