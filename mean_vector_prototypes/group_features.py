import pickle
import torch
import os
import modeling.model_manager as models

class Grouped_Features(torch.nn.Module):
    def __init__(self, cfg, log_path, use_best_backbone=True):
        super(Grouped_Features, self).__init__()
        self.network = models.get_net(cfg)
        if use_best_backbone:
            self.get_trained_backbone_best(log_path)
        else:
            self.get_trained_backbone_last(log_path)
        self.curr_min = float('inf')
        self.curr_max = float('-inf')
        self.running_sum_of_squares = 0.0
        self.count = 0
        
    def get_trained_backbone_best(self, log_path):
        net_path = os.path.join(log_path, 'backbone_model.pth')
        self.network.load_state_dict(torch.load(net_path))
        for param in self.network.parameters():
            param.requires_grad = False
        self.network.cuda()
        self.network.eval()
        print("Model loaded for best prototypes saving")

    def get_trained_backbone_last(self, log_path):
        last_checkpoint = torch.load(os.path.join(log_path, 'last_checkpoint.pth'))  
        self.network.load_state_dict(last_checkpoint['backbone_model_state_dict'])
        for param in self.network.parameters():
            param.requires_grad = False
        self.network.cuda()
        self.network.eval()
        print("Model loaded for prototypes initialization")

    def get_stds(self, minibatch):
        image, _, _, = minibatch
        feature = self.network(image)
        std_deviation = torch.std(feature)
        squared_std_dev = std_deviation ** 2
        # Update the running sum of squares and count
        self.running_sum_of_squares += squared_std_dev
        self.count += 1

    def find_min_max(self, minibatch):
        image, _, _ = minibatch
        feature = self.network(image)

        min_values = torch.min(feature)
        max_values = torch.max(feature)

        min_value = torch.min(min_values)
        max_value= torch.max(max_values)

        if min_value < self.curr_min:
            self.curr_min = min_value.item()
        if max_value > self.curr_max:
            self.curr_max = max_value.item()


    def save_mean_pickled_file(self, save_path, means_dict):
        directory_path = os.path.dirname(save_path)
        # if os.path.exists(save_path):
        #     print("Means dict pickle already exits")
        #     # raise ValueError(f'{save_path}')
        # else:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(save_path, 'wb') as file:
            pickle.dump(means_dict, file)

    def calculate_mean_vector(self, cfg, feat_type):
        print('Calculating mean vectors')
        if feat_type == 'init':
            features_dict_path = os.path.join('artifacts', 'saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], "init_features.pkl")
            save_path = os.path.join('artifacts', 'saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], "init_prototypes.pkl")
        elif feat_type == 'curr':
            features_dict_path = os.path.join('artifacts', 'saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], "curr_features.pkl")
            save_path = os.path.join('artifacts', 'saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], "curr_prototypes.pkl")

        with open(features_dict_path, 'rb') as file:
            prototypes_dict = pickle.load(file)

        means_dict = {key1: {key2: [] for key2 in prototypes_dict[key1]} for key1 in prototypes_dict}

         # Calculate mean vectors for each combination of domain and class.
        for key1, inner_dict in prototypes_dict.items():
            for key2, feature_list in inner_dict.items():
                if feature_list:  # Check if there are feature vectors for this combination.
                    # Stack feature vectors into a single tensor.
                    stacked_features = torch.stack(feature_list, dim=0)
                    # Calculate the mean vector for this combination.
                    mean_vector = torch.mean(stacked_features, dim=0)
                    means_dict[key1][key2] = mean_vector

        
        self.save_mean_pickled_file(save_path, means_dict)

    def save_grouped_features_pickled_file(self, save_path, grouped_dict):
        if os.path.exists(save_path):
            with open(save_path, 'rb') as file:
                existing_data = pickle.load(file)

            for key1, inner_dict in grouped_dict.items():
                for key2, value_list in inner_dict.items():
                    existing_data[key1][key2].extend(value_list)

            with open(save_path, 'wb') as file:
                pickle.dump(existing_data, file)
        else:
            directory_path = os.path.dirname(save_path)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            with open(save_path, 'wb') as file:
                pickle.dump(grouped_dict, file)
                
    def all_features_grouped(self, minibatch, cfg, epoch,  feat_type='init'):   
        print('Grouping features....')
        domain_class_dict = {d: {c: [] for c in range(cfg.DATASET.NUM_CLASSES)} for d in range(cfg.DATASET.NUM_SOURCE_DOMAINS)}
        image_batch, label_batch, domain_batch = minibatch

        for image, label, domain in zip(image_batch, label_batch, domain_batch):
            label = int(label)
            domain = int(domain)
            features = self.network(image.unsqueeze(dim=0))
            domain_class_dict[domain][label].append(features)

        if feat_type == 'init':
            save_path = os.path.join('artifacts', 'saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], 'init_features.pkl')
        elif feat_type == 'curr':
            save_path = os.path.join('artifacts', 'saved_prototypes', cfg.DATASET.TARGET_DOMAINS[0], 'curr_features.pkl')
        elif feat_type == 'last':
            save_path = os.path.join('artifacts', 'prototypes', cfg.TIMESTAMP, cfg.DATASET.TARGET_DOMAINS[0], str(epoch), 'last_features.pkl')
        self.save_grouped_features_pickled_file(save_path, domain_class_dict)
       
            
        