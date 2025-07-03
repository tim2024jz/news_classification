import argparse
import torch
import os
import random
import json
import numpy as np
from torch.cuda import device

class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # select device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.cuda))
        else:
            self.device = torch.device('cpu')

        # determine the model name and model dir
        if self.model_name is None:
            self.model_name = 'Att_BLSTM'
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # backup data
        self.__config_backup(args)

        # set the random seed
        self.__set_seed(self.seed)

    def __get_config(self):
        # Function to parse command line arguments and return the configuration
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # Several key selective parameters
        parser.add_argument('--data_dir', type=str, default='./data', help='dir to load data')
        parser.add_argument('--output_dir', type=str, default='./output', help='dir to save output')

        # Word embedding
        parser.add_argument('--embedding_path', type=str, default='E:\pythonProject\glove.6B\glove.6B.100d.txt',
                            help='pre_trained word embedding')
        parser.add_argument('--word_dim', type=int, default=100, help='dimension of word embedding')

        # Train settings
        parser.add_argument('--model_name', type=str, default=None, help='model name')
        parser.add_argument('--mode', type=int, default=1, choices=[0, 1],
                            help='running mode: 1 for training; otherwise testing')
        parser.add_argument('--seed', type=int, default=5782, help='random seed')
        parser.add_argument('--cuda', type=int, default=0, help='num of gpu device, if -1, select cpu')
        parser.add_argument('--epoch', type=int, default=15, help='max epoches during training')

        # Hyperparameters
        parser.add_argument('--batch_size', type=int, default=10, help='batch size')
        parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
        parser.add_argument('--max_len', type=int, default=100, help='max length of sentence')

        parser.add_argument('--emb_dropout', type=float, default=0.3, help='the possibility of dropout in embedding layer')
        parser.add_argument('--lstm_dropout', type=float, default=0.3, help='the possibility of dropout in (Bi)LSTM layer')
        parser.add_argument('--linear_dropout', type=float, default=0.5, help='the possibility of dropout in linear layer')
        parser.add_argument('--hidden_size', type=int, default=100, help='the dimension of hidden units in (Bi)LSTM layer')
        parser.add_argument('--layers_num', type=int, default=1, help='num of RNN layers')

        parser.add_argument('--L2_decay', type=float, default=1e-5, help='L2 weight decay')

        args = parser.parse_args()
        return args

    def __set_seed(self, seed=1234):
        # Function to set random seeds for reproducibility
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # set seed for cpu
        torch.cuda.manual_seed(seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(seed)  # set seed for all gpu

    def __config_backup(self, args):
        # Function to backup the configuration to a JSON file
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)

    def print_config(self):
        # Function to print the configuration
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])

if __name__ == '__main__':
    # Create an instance of the Config class and print the configuration
    config = Config()
    config.print_config()
