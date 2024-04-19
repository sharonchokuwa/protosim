from dataset.data_manager import get_dataset
from group_features import *
from utils.args import *
import os
import torch
import pickle

MIN_MAX_DICT = {'deepdr': {'min': float('inf'), 'max': float('-inf')}, 
                'idrid':  {'min': float('inf'), 'max': float('-inf')},
                'rldr':   {'min': float('inf'), 'max': float('-inf')},
                'fgadr':  {'min': float('inf'), 'max': float('-inf')},
                'aptos':  {'min': float('inf'), 'max': float('-inf')},
                'messidor_2': {'min': float('inf'), 'max': float('-inf')},
               }

STDS = {'deepdr': 0.0, 
                'idrid':  0.0,
                'rldr':   0.0,
                'fgadr':  0.0,
                'aptos':  0.0,
                'messidor_2': 0.0,
               }

def remove_current_element(input_list, curr_element):
    return [element for element in input_list if element != curr_element]

def init_prototypes(cfg, path_to_model, train_loader, epoch, target):
    group_features_obj = Grouped_Features(cfg, path_to_model, use_best_backbone=False)
    for image, _, label, domain, img_index in train_loader:
        minibatch = [image.cuda(), label.cuda().long(), domain.cuda().long()]
        group_features_obj.all_features_grouped(minibatch, cfg, epoch, feat_type='init')
        group_features_obj.find_min_max(minibatch)
        group_features_obj.get_stds(minibatch)

    path_to_grouped_feats = os.path.join('artifacts', 'init_saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], 'all_features.pkl')
    with open(path_to_grouped_feats, 'rb') as file:
        domain_members_dict = pickle.load(file)
    
    dummy_tensor = torch.randn(2048)
    proto_dict = {d: {c: dummy_tensor for c in range(cfg.DATASET.NUM_CLASSES)} for d in range(cfg.DATASET.NUM_SOURCE_DOMAINS)}

    for domain_id, domain in domain_members_dict.items():
        for label_id, label_members in domain.items():
            for feature_vector in label_members:
                print(f"Domain-Label[{domain_id}][{label_id}]")
                proto_dict[domain_id][label_id] = feature_vector
                break

    path_to_prototypes = os.path.join('artifacts', 'init_saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], 'init_prototypes.pkl')
    with open(path_to_prototypes, 'wb') as file:
        pickle.dump(proto_dict, file)

    MIN_MAX_DICT[target]['min'] = group_features_obj.curr_min
    MIN_MAX_DICT[target]['max'] = group_features_obj.curr_max
    overall_std_dev = torch.sqrt(group_features_obj.running_sum_of_squares / group_features_obj.count)
    STDS[target] = overall_std_dev

def init_func(target_domain, source_domains, cfg, dataset_num, args, timestamp):
    print("Target Dataset: ", target_domain[0])
    #datasets info
    cfg.DATASET.SOURCE_DOMAINS = source_domains
    cfg.DATASET.TARGET_DOMAINS = target_domain
    cfg.DATASET.NUM_SOURCE_DOMAINS = len(source_domains)

    train_loader, _, _, _ = get_dataset(cfg)
    epoch = 0
    path_to_model = os.path.join('trained_backbones', target_domain[0])
    init_prototypes(cfg, path_to_model, train_loader, epoch, target_domain[0])

if __name__ == "__main__":
    timestamp = '2023-10-23_18-59-32'
    domain_names = ['deepdr', 'idrid', 'rldr', 'fgadr', 'aptos', 'messidor_2']
    args = get_args()
    cfg = setup_cfg(args)

    for idx, domain in enumerate(domain_names):
        target_domain = [domain]
        source_domains = remove_current_element(domain_names, domain)
        init_func(target_domain=target_domain, source_domains=source_domains, cfg=cfg, dataset_num=idx,  args=args, timestamp=timestamp)

    save_path = os.path.join('artifacts', 'init_saved_prototypes', 'min_max.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(MIN_MAX_DICT, file)
    
    std_save_path = os.path.join('artifacts', 'init_saved_prototypes', 'stds.pkl')
    with open(std_save_path, 'wb') as file:
        pickle.dump(STDS, file)
    print('STDS =', STDS)

    
