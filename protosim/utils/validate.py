import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
from modeling.center_loss import DomainCenterLoss

# validate the algorithm by AUC, accuracy and f1 score on val/test datasets#

def algorithm_validate(algorithm, center_loss_obj, sim_loss_obj, cfg, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        val_center_loss = 0
        val_sim_loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for batch_idx, (image, label, domain, _ ) in enumerate(data_loader):            
            image = image.cuda()
            label = label.cuda().long()
            domain = domain.cuda().long()

            output = algorithm.predict(image)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

            feat = algorithm.get_feat(image)

            center_loss = center_loss_obj(feat, domain, label)
            val_center_loss += center_loss

            prototypes = center_loss_obj.get_centers()
            sim_loss = sim_loss_obj(feat, domain, label, cfg, prototypes)
            val_sim_loss += sim_loss
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)
        val_sim_loss = val_sim_loss / len(data_loader)
        val_center_loss = val_center_loss / len(data_loader)

        if val_type in ['val', 'test']:
            writer.add_scalar('{}/accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('{}/loss'.format(val_type), loss, epoch)
            writer.add_scalar('{}/auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('{}/f1'.format(val_type), f1, epoch)     
            writer.add_scalar('{}_center_loss/center_loss'.format(val_type), val_center_loss, epoch)   
            writer.add_scalar('{}_sim_loss/sim_loss'.format(val_type), val_sim_loss, epoch)   
                
            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format
            (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return auc_ovo, loss, acc, f1

