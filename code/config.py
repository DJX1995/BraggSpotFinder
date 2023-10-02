import yaml
from easydict import EasyDict as edict

config = edict()

config.LOG_DIR = ''
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''
config.VERBOSE = False
config.TAG = ''

# dataset related
config.dataset = edict()
config.dataset.name = 'image'
config.dataset.datatype = 'image'
config.dataset.datadir = '../data/'
config.dataset.splitpath = '../data/all_data/data_split_new.json'
config.dataset.fullpath = '../data/all_data/data_full_path.json'
config.dataset.datapath = '../data/all_data/data_combine_without_negative_rpos.h5py'

# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# model related params
config.model = edict()
config.model.hidden = 64
config.model.kernel = 3
config.model.feature = [64, 128, 256]  # bottleneck is 512 dim
config.model.drop_rate = 0.1

# train
config.train = edict()
config.train.subset = None
config.train.lr = 0.001
config.train.decay_weight = 0.1
config.train.decay_step = 6
config.train.n_epoch = 8
config.train.batch_size = 12
config.train.resume = False
config.train.save_best = True
config.train.model_id = 'r6_no_negative'
config.train.theta = 0.0

config.loss = edict()
config.loss.NAME = 'total_loss'

# test
config.test = edict()
config.test.datadir = '../data/all_data/data_combine_without_negative_rpos.h5py'
config.test.start_epoch = 0
config.test.print_freqence = 1
config.test.TIOU = []
config.test.NMS_THRESH = 0.4
config.test.INTERVAL = 1
config.test.EVAL_TRAIN = False
config.test.batch_size = 12


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))