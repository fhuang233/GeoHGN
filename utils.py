import os
import pickle
import torch
import logging

import time
import json
import csv
import shutil

import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    #os.environ['PYTHONHASHSEED'] = str(seed)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def check_nan(tensor, epoch, position, weight=None, data=None):
    if not torch.all(torch.logical_not(torch.isnan(tensor))):
        if weight is not None:
            print(weight)
        if data is not None:
            print(data.idx)
        raise  Exception(position + ' contains NaN, epoch: ' + str(epoch))


def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def make_logdir(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    savetag = timestamp + '_' + args.model_name + '_' + 'repeat' + str(args.repeat) + '_cuda' + str(args.cuda)
    if args.log_mark != '':
        savetag = savetag + '_' + args.log_mark

    save_dir = args.save_dir
    if save_dir == None:
        raise Exception('save_dir can not be None!')
    train_save_dir = os.path.join(save_dir, savetag)
    log_dir = os.path.join(train_save_dir, 'log', 'train')
    model_dir = os.path.join(train_save_dir, 'model')
    result_dir = os.path.join(train_save_dir, 'result')

    create_dir([log_dir, model_dir, result_dir])
    print(log_dir)
    log_path = os.path.join(log_dir, 'Train.log')
    config_path = os.path.join(log_dir, 'Train' + args.model_name + 'Config.json')
    result_path = os.path.join(result_dir, 'Test' + args.model_name + 'Result.csv')
    
    return log_path, model_dir, config_path, result_path


class TrainLogger(logging.Logger):
    def __init__(self, args, name=None, level=logging.DEBUG):
        self._log_path, self._model_dir, self._config_path, self._result_path = make_logdir(args)
        
        super(TrainLogger, self).__init__(name, level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    "%Y-%m-%d %H:%M:%S")
        self.handlers = self._get_handlers(formatter)
        self._save_config(args)
    
    def _get_handlers(self, formatter): 
            # Create a file handler
        file_handler = logging.FileHandler(self._log_path)
        file_handler.setFormatter(formatter)

        # use StreamHandler for print
        print_handler = logging.StreamHandler()
        print_handler.setFormatter(formatter)
        
        return [file_handler, print_handler]
    
    def _save_config(self, args):
        with open(self._config_path, 'w') as f:
            f.write(json.dumps(vars(args)))
    
    def save_model(self, model, message):
        model_path = os.path.join(self._model_dir, message + '.pt')
        torch.save(model.state_dict(), model_path)
        print("model has been saved to %s." % (model_path))
        return model_path
    
    def load_model(self, model, ckpt):
        model.load_state_dict(torch.load(ckpt))
    
    def save_result(self, result, fieldnames=['Dataset', 'rmse', 'pr']):
        
        with open(self._result_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in result:
                writer.writerow(row)
    
    def save_best_model(self, ckpt):
        model_path = os.path.join(self._model_dir, 'best_model' + '.pt')
        shutil.copy(ckpt, model_path)
        
    
