"""
This code is partially borrowed from https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import copy

class Sim_Loss(nn.Module):
    def __init__(self, cfg, vector_size, proto_path):
        super(Sim_Loss, self).__init__()
        self.prototypes = nn.Parameter(torch.empty(cfg.DATASET.NUM_SOURCE_DOMAINS, cfg.DATASET.NUM_CLASSES, vector_size))

    def init_prototypes(self, proto_path):
        with open(proto_path, 'rb') as file:
            proto_dict = pickle.load(file)

        with torch.no_grad():
            for d in proto_dict:
                for c in proto_dict[d]:
                    self.prototypes[d, c] = proto_dict[d][c]
        
    def update_prototype(self, domain, label, new_prototype):
        with torch.no_grad():
            self.prototypes[domain][label] = new_prototype
    
    def get_prototypes(self,):
        return self.prototypes

    def save_prototypes(self, prototypes_path):
        directory_path = os.path.dirname(prototypes_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(prototypes_path, 'wb') as file:
            pickle.dump(self.prototypes, file)
    
    def save_prototypes_info(self, cfg, epoch):
        prototypes_path = os.path.join('artifacts', 'prototypes', cfg.TIMESTAMP, cfg.DATASET.TARGET_DOMAINS[0], str(epoch), 'prototypes.pkl')
        self.save_prototypes(prototypes_path)

    def broadcast_shape(self, sim_a, cfg):
        desired_shape = (sim_a.shape[0], cfg.DATASET.NUM_SOURCE_DOMAINS, cfg.DATASET.NUM_CLASSES)
        num_dims_to_add = len(desired_shape) - len(sim_a.shape)
        for _ in range(num_dims_to_add):
            sim_a = sim_a.unsqueeze(-1)

        sim_a = sim_a.expand(*desired_shape)
        return sim_a

    def calculate_sim(self, feat, w_prototype, cfg, dim):
        sim = torch.exp(-torch.norm(feat - w_prototype, dim=dim) ** 2 / 2 * cfg.SIGMA ** 2 )
        return sim

    def max_with_zero(self, a, b):
        max_result = torch.max(a, b)
        max_result[max_result < 0] = 0
        return max_result
    
    def forward(self, feat, domain_batch, label_batch, cfg):
        epsilon = 1e-6
        prots_a = self.prototypes[domain_batch, label_batch]  # Shape: [batch_size, 2048]

        # Calculate similarities between samples and prototypes
        sim_a = self.calculate_sim(feat, prots_a, cfg, dim=1)  # Shape: [batch_size]
        all_prototypes = self.prototypes.clone()
        all_prototypes = all_prototypes.permute(0, 2, 1)  # Shape: [NUM_SOURCE_DOMAINS, 2048, NUM_CLASSES]
        all_prototypes = all_prototypes.unsqueeze(0)  # Shape: [1, NUM_SOURCE_DOMAINS, 2048, NUM_CLASSES]
        feat = feat.unsqueeze(2).unsqueeze(1)  # from [batch_size, 2048]---> [batch_size, 1, 2048, 1]
        all_similarities_v0 = self.calculate_sim(feat, all_prototypes, cfg, dim=2)  # Shape: [batch_size, NUM_SOURCE_DOMAINS, NUM_CLASSES]
        all_similarities = all_similarities_v0.clone()

        # Exclude the current domain and label from the loss calculation ---> remove sim_a
        all_similarities[torch.arange(all_similarities.size(0)), domain_batch, label_batch] = 0

        # Compute the sample similarity loss in a vectorized way
        sim_b = all_similarities.clone()
        sim_c = all_similarities.clone()
        
        sim_b = sim_b[torch.arange(all_similarities.size(0)), :, label_batch] # Shape: [batch_size, NUM_SOURCE_DOMAINS]
        sim_c = sim_c[torch.arange(all_similarities.size(0)), domain_batch, :] # Shape: [batch_size, NUM_CLASSES]

        sim_b = sim_b.unsqueeze(2) # Shape: [batch_size, NUM_SOURCE_DOMAINS, 1]
        sim_c = sim_c.unsqueeze(2)  # Shape: [batch_size, NUM_CLASSES, 1]
        sim_c = torch.transpose(sim_c, 2, 1) # Shape: [batch_size, 1, NUM_CLASSES]
        sim_a = self.broadcast_shape(sim_a, cfg) # Shape: [batch_size, NUM_SOURCE_DOMAINS, NUM_CLASSES]

        sim_result = sim_b + sim_c - sim_a
        maxed_out = torch.max(sim_result, torch.zeros_like(sim_result))
        maxed = maxed_out.clone()
        #set the j = i == 0
        maxed[torch.arange(maxed.size(0)), domain_batch, :] = 0
        maxed[torch.arange(maxed.size(0)), :, label_batch] = 0

        loss_sum = cfg.S_LAMBDA * torch.sum(maxed)
        return loss_sum


    # For loop for the update
    # def update(self, minibatch, cfg, log_path):
    #     image_batch, mask_batch, label_batch, domain_batch = minibatch
    #     self.optimizer.zero_grad()

    #     all_features = self.network(image_batch)
    #     output = self.classifier(all_features)
    #     ce_loss = F.cross_entropy(output, label_batch)

    #     batch_sim_loss = 0.0
    #     dim = 0
    #     epsilon = 1e-6
    #     for image, label, domain in zip(image_batch, label_batch, domain_batch):
    #         label = int(label)
    #         domain = int(domain)
    #         feat = self.network(image.unsqueeze(dim=dim))

    #         sample_sim_loss = 0.0
    #         prot_a = self.prototypes[domain][label]
    #         feat = feat.squeeze()
    #         sim_a = self.calculate_sim(feat, prot_a, cfg, dim=dim)
    #         for d in range(cfg.DATASET.NUM_SOURCE_DOMAINS):
    #             for c in range(cfg.DATASET.NUM_CLASSES):
    #                 if d == domain or c == label:
    #                     continue
    #                 prot_b = self.prototypes[d][label]
    #                 sim_b = self.calculate_sim(feat, prot_b, cfg, dim=dim)
    #                 prot_c = self.prototypes[domain][c]
    #                 sim_c = self.calculate_sim(feat, prot_c, cfg, dim=dim)

    #                 curr_loss = self.criterion(sim_a, sim_b, sim_c)
    #                 sample_sim_loss += curr_loss
    #         batch_sim_loss += sample_sim_loss

    #     batch_sim_loss = batch_sim_loss + epsilon
    #     second_batch_sim_loss = self.second(all_features, domain_batch, label_batch, cfg) 
    #     total_loss = ce_loss + (batch_sim_loss * cfg.S_LAMBDA)
    #     total_loss.backward()
    #     self.optimizer.step()

    #     return {'loss': total_loss,
    #             'sim_loss': batch_sim_loss,
    #             'second': second_batch_sim_loss,
    #             'ce_loss': ce_loss,
    #            }

     
# DGRNet loss function
class DahLoss(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        # self.domain_num_dict = {'MESSIDOR': 1744,
        #                         'IDRID': 516,
        #                         'DEEPDR': 2000,
        #                         'FGADR': 1842,
        #                         'APTOS': 3662,
        #                         'RLDR': 1593}
        
        # self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
        #                         'IDRID': [175, 26, 163, 89, 60],
        #                         'DEEPDR': [917, 214, 402, 353, 113],
        #                         'FGADR': [100, 211, 595, 646, 286],
        #                         'APTOS': [1804, 369, 999, 192, 294],
        #                         'RLDR': [165, 336, 929, 98, 62]}
#
        self.domain_num_dict = {
                            'messidor_2': 1744,
                            'idrid': 516,
                            'deepdr': 1600,
                            'fgadr': 1842,
                            'aptos': 3656,
                            'rldr': 1593
                        }

        self.label_num_dict = {
                            'messidor_2': [1016, 269, 347, 75, 35], #ok
                            'idrid': [175, 26, 163, 89, 60], #ok
                            'deepdr': [714, 186, 326, 282, 92],
                            'fgadr': [100, 211, 595, 646, 286], # ok
                            'aptos': [1801, 369, 998, 193, 295],
                            'rldr': [165, 336, 929, 98, 62] #ok
                        }

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)            

        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss
