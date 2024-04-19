import algorithms
import os
from utils.validate import *
from utils.args import *
from utils.misc import *
from utils.results_to_excel import *
from dataset.data_manager import get_dataset
from tqdm import tqdm
from clearml import Task
from datetime import datetime
from group_features import *

def remove_current_element(input_list, curr_element):
    return [element for element in input_list if element != curr_element]

def run_algo(target_domain, source_domains, cfg, timestamp, dataset_num, excel_data, args):
    log_path = os.path.join(f'./result/{timestamp}/{cfg.ALGORITHM}/{target_domain[0]}', cfg.OUTPUT_PATH)
    proto_path = os.path.join('artifacts', 'saved_prototypes', target_domain[0], 'init_prototypes.pkl')

    #datasets info
    cfg.DATASET.SOURCE_DOMAINS = source_domains
    cfg.DATASET.TARGET_DOMAINS = target_domain
    cfg.DATASET.NUM_SOURCE_DOMAINS = len(source_domains)
    cfg.TIMESTAMP = timestamp

    #setup clearML if not in debug mode
    if not cfg.DEBUG:
        task = Task.init(project_name=f'{cfg.PROJECT_NAME}/{timestamp}', 
                         task_name=cfg.DATASET.TARGET_DOMAINS[0],
                        )
        
    # init
    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    writer = init_log(args, cfg, log_path, len(train_loader), dataset_size, timestamp)
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg, proto_path)
    algorithm.cuda()

    num_epochs = cfg.EPOCHS

    # train
    iterator = tqdm(range(num_epochs))
    scheduler = get_scheduler(algorithm.optimizer, num_epochs)

    best_performance = 0.0
        
    for i in iterator:
        epoch = i + 1

        loss_avg = LossCounter()
        loss_ce_avg = LossCounter()
        loss_sim_avg = LossCounter()

        for image, mask, label, domain, img_index in train_loader:
            algorithm.train()
            minibatch = [image.cuda(), mask.cuda(), label.cuda().long(), domain.cuda().long()]
            loss_dict_iter = algorithm.update(minibatch, cfg)
            loss_avg.update(loss_dict_iter['loss']) 
            loss_ce_avg.update(loss_dict_iter['ce_loss'])
            loss_sim_avg.update(loss_dict_iter['sim_loss'])

        update_writer_ours(writer, epoch, scheduler, loss_avg, loss_ce_avg, loss_sim_avg)
        
        alpha = algorithm.update_epoch(epoch)
        if cfg.ALGORITHM == 'ProtoSim':
            algorithm.saving_last(cfg, log_path)

        scheduler.step()

        if epoch >= cfg.WARM_UP_EPOCHS:
            if epoch == cfg.WARM_UP_EPOCHS:
                proto_path = os.path.join('artifacts', 'saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], "init_prototypes.pkl")
                feat_type='init'
            else: 
                proto_path = os.path.join('artifacts', 'saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], "curr_prototypes.pkl")
                feat_type='curr'

            group_features_obj = Grouped_Features(cfg, log_path, use_best_backbone=False)
            for image, _, label, domain, img_index in train_loader:
                minibatch = [image.cuda(), label.cuda().long(), domain.cuda().long()]
                group_features_obj.all_features_grouped(minibatch, cfg, epoch, feat_type=feat_type)
            group_features_obj.calculate_mean_vector(cfg, feat_type)
            
            algorithm.sim_loss_criterion.init_prototypes(proto_path)

        # validation
        if epoch % cfg.VAL_EPOCH == 0:
            val_auc, test_auc, _, _ = algorithm.validate(val_loader, test_loader, writer)
            if val_auc > best_performance:
                best_performance = val_auc
                algorithm.save_model(log_path)

            prototypes = algorithm.sim_loss_criterion.get_prototypes()
            log_proto_weights(writer, prototypes, epoch)
    
    algorithm.renew_model(log_path)
    _, test_auc, test_acc, test_f1 = algorithm.validate(val_loader, test_loader, writer)

    metrics = {
                'acc': test_acc,
                'auc': test_auc,
                'f1': test_f1,
              }
    model_num = dataset_num + 1
    excel_data = add_data(excel_data, model_num , metrics)
    num_domains = len(cfg.DATASET.SOURCE_DOMAINS) + 1

    if num_domains ==  model_num:
        save_to_excel(log_path, excel_data, cfg)

    if cfg.SAVE_PROTOTYPES:
        algorithm.sim_loss_criterion.save_prototypes_info(cfg, cfg.EPOCHS)
        group_features_obj = Grouped_Features(cfg, log_path, use_best_backbone=True)
        for image, _, label, domain, img_index in train_loader:
            minibatch = [image.cuda(), label.cuda().long(), domain.cuda().long()]
            group_features_obj.all_features_grouped(minibatch, cfg, epoch, feat_type='last')

    os.mknod(os.path.join(log_path, 'done'))
    writer.close()
    if not cfg.DEBUG:
        task.close()
    
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    domain_names = ['deepdr', 'idrid', 'rldr', 'fgadr', 'aptos', 'messidor_2']

    excel_data = initialize_excel()
    args = get_args()
    cfg = setup_cfg(args)
    
    for idx, domain in enumerate(domain_names):
        target_domain = [domain]
        source_domains = remove_current_element(domain_names, domain)
        run_algo(target_domain=target_domain, source_domains=source_domains, cfg=cfg, \
                timestamp=timestamp, dataset_num=idx, excel_data=excel_data, args=args)

    

    

    
