import sys, os, logging, shutil
from torch.utils.tensorboard import SummaryWriter   
import torch, random
import numpy as np
from collections import Counter
ALL_DATASETS=['APTOS','DEEPDR','FGADR','IDRID','MESSIDOR','RLDR']
ESDG_DATASETS = ['APTOS','DEEPDR','FGADR','IDRID','MESSIDOR','RLDR','DDR','EYEPACS']
ALL_METHODS = ['GDRNet', 'ERM', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen', 'ProtoSim']

def count_samples_per_class(targets, num_classes):
    counts = Counter()
    for y in targets:
        counts[int(y)] += 1
    return [counts[i] if counts[i] else np.inf for i in range(num_classes)]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_log(args, cfg, log_path, train_loader_length, dataset_size, timestamp):
    assert cfg.ALGORITHM in ALL_METHODS
    #if not cfg.RANDOM:
    setup_seed(cfg.SEED)
        
    init_output_foler(cfg, log_path)
    writer = SummaryWriter(os.path.join(log_path, 'tensorboard'))
    writer.add_text('config', str(args))

    logging.basicConfig(filename=f'./result/{timestamp}/{cfg.ALGORITHM}/log.txt', level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("----------------------------------------------------------------------------")
    logging.info(cfg.DATASET.TARGET_DOMAINS)
    logging.info("----------------------------------------------------------------------------")
    logging.info("{} iterations per epoch".format(train_loader_length))
    logging.info("We have {} images in train set, {} images in val set, and {} images in test set.".format(dataset_size[0], dataset_size[1], dataset_size[2]))
    logging.info(str(args))
    logging.info(str(cfg))
    return writer

def init_output_foler(cfg, log_path):
    if os.path.isdir(log_path):
        if cfg.OVERRIDE:
            shutil.rmtree(log_path)
        else:
            if os.path.exists(os.path.join(log_path, 'done')):
                print('Already trained, exit')
                exit()
            elif not cfg.RESUME:
                #shutil.rmtree(log_path)
                pass
    else:
        os.makedirs(log_path)

def get_scheduler(optimizer, max_epoch):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch * 0.5], gamma=0.1)
    return scheduler

def update_writer_ours(writer, epoch, loss_avg, loss_ce_avg, loss_dc_avg, loss_scaled_dc_avg, loss_sim_avg, loss_scaled_sim_avg):
    logging.info('epoch: {}, total loss: {}'.format(epoch, loss_avg.mean()))
    logging.info('epoch: {}, ce loss: {}'.format(epoch, loss_ce_avg.mean()))
    logging.info('epoch: {}, center_loss: {}'.format(epoch, loss_dc_avg.mean()))
    logging.info('epoch: {}, scaled_center_loss: {}'.format(epoch, loss_scaled_dc_avg.mean()))
    logging.info('epoch: {}, sim_loss: {}'.format(epoch, loss_sim_avg.mean()))
    logging.info('epoch: {}, scaled_sim_loss: {}'.format(epoch, loss_scaled_sim_avg.mean()))
    writer.add_scalar('info/total_loss', loss_avg.mean(), epoch)
    writer.add_scalar('info/ce_loss', loss_ce_avg.mean(), epoch)
    writer.add_scalar('center_loss/center_loss', loss_dc_avg.mean(), epoch)
    writer.add_scalar('sim_loss/sim_loss', loss_sim_avg.mean(), epoch)

def log_proto_weights(writer, prototypes, epoch):
    with torch.no_grad():
        prototype_values = prototypes.cpu().numpy()
    writer.add_histogram('prototypes', prototype_values, global_step=epoch)
    
def update_writer(writer, epoch, scheduler, loss_avg):
    logging.info('epoch: {}, total loss: {}'.format(epoch, loss_avg.mean()))
    writer.add_scalar('info/lr', scheduler.get_last_lr()[0], epoch) 
    writer.add_scalar('info/loss', loss_avg.mean(), epoch)

class MovingAverage:
    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.named_parameters = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.named_parameters[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.named_parameters[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data
    
class LossCounter:
    def __init__(self, start = 0):
        self.sum = start
        self.iteration = 0
    def update(self, num):
        self.sum += num
        self.iteration += 1
    def mean(self):
        return self.sum * 1.0 / self.iteration