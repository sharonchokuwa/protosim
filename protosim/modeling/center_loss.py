import torch
import torch.nn as nn
import os
import pickle

'''Center Loss adapted from https://github.com/KaiyangZhou/pytorch-center-loss/tree/master'''

class DomainCenterLoss(nn.Module):
    """Domain Center loss.
    
    Args:
        num_domains (int): number of domains.
        num_classes_per_domain (int): number of classes per domain.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_domains=5, num_classes_per_domain=5, feat_dim=2048):
        super(DomainCenterLoss, self).__init__()
        self.num_domains = num_domains
        self.num_classes_per_domain = num_classes_per_domain
        self.feat_dim = feat_dim
        self.num_classes = num_domains * num_classes_per_domain

        self.centers = nn.Parameter(torch.randn(self.num_domains, self.num_classes_per_domain, self.feat_dim).cuda())

    def forward(self, x, domain_labels, class_labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers.view(-1, self.feat_dim), 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.view(-1, self.feat_dim).t())

        classes = torch.arange(self.num_classes).long().cuda()
        class_labels = class_labels + domain_labels * self.num_classes_per_domain
        labels = class_labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

    def get_centers(self, ):
        return self.centers.view(self.num_domains, self.num_classes_per_domain, self.feat_dim)

    def save_prototypes(self, prototypes_path):
        directory_path = os.path.dirname(prototypes_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(prototypes_path, 'wb') as file:
            pickle.dump(self.centers, file)
    
    def save_prototypes_info(self, cfg, epoch):
        prototypes_path = os.path.join('artifacts', 'prototypes', cfg.TIMESTAMP, cfg.DATASET.TARGET_DOMAINS[0], str(epoch), 'prototypes.pkl')
        self.save_prototypes(prototypes_path)

