import torch
from utils import calc_auc, calc_aupr, calc_other_metrics
import traceback
from torch import nn
import numpy as np
import random

def train(model, train_dataloader, optimizer, device, lr, train_entity2index=None):
    model.train()
    avg_loss = 0
    avg_reg_loss = 0
    affected_batches = 0

    size = len(train_dataloader)

    for i, data in enumerate(train_dataloader):
            if data is not None :
                h, t, pos_t, neg_t, _, _,mask = data  # indices
                h = h.to(device)
                t = t.to(device)
                pos_t = pos_t.to(device)
                neg_t = neg_t.to(device)
                loss, reg_loss = model.train_step(h, t, pos_t, neg_t, mask)
                if loss is not None :
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.clone().detach().to('cpu').item()
                    avg_reg_loss += reg_loss.clone().detach().to('cpu').item()
                else :
                    continue
            #except Exception as e:
            else :
                #print(93, 'procedure except', e, traceback.format_exc())
                #break
                #if data is None :
                #   affected_batches += 1
                continue
    print("90, The number of affected batches in the train procedure : ", affected_batches)
    return avg_loss / size, avg_reg_loss / size


def test(model, test_dataloader, device, test_entity2index):
    seed = 120 # 20, 80, 100, 200
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.eval()
    all_logit = []
    all_label = []
    affected_batches = 0
    #num_eval_samples = 0
    with torch.no_grad():
        model = model.eval()
        for i, data in enumerate(test_dataloader):
            if data is not None :
                h, t, pos_t, neg_t, pos_label, neg_label, mask = data
                test_index2entity = {value: key for key, value in test_entity2index.items()}
                
                h = h.to(device)
                pos_t = pos_t.to(device)
                t = t.to(device)
                neg_t = neg_t.to(device)
                model = model.to(device)
                score = model.predict(h, pos_t, neg_t, test_entity2index, mask)
                if score is not None :
                    all_logit = all_logit + score.to(device).tolist()
                    if neg_label.numel() != 0 :
                        pos_label_flat = pos_label.contiguous().view(-1)
                        neg_label_flat = neg_label.contiguous().view(-1)
                        if mask is not None :
                            pos_label_mask = mask['pos_labels'].contiguous().view(-1)
                            neg_label_mask = mask['neg_labels'].contiguous().view(-1)
                            pos_label_valid = pos_label_flat[pos_label_mask]
                            neg_label_valid = neg_label_flat[neg_label_mask]
                            label = torch.cat((pos_label_valid.to(device), neg_label_valid.to(device))) if pos_t.numel() != 0 else neg_label_valid
                        else :
                            label = torch.cat((pos_label_flat.to(device), neg_label_flat.to(device)))
                        
                    else :
                        #print(144, "procedure", neg_label)
                        label = pos_label.view(-1)
                        if mask is not None :
                            pos_label_mask = mask['pos_labels'].contiguous().view(-1)
                            #label = label[~pos_label_mask]
                            label = label[pos_label_mask]
                    label = label.to(device)
                    if score.shape != label.shape :
                        print(58, "procedure, score and label has shapes varied", score.shape, label.shape, h, t, pos_t, neg_t, pos_label, neg_label)
                        #print(58, h, t, torch.count_nonzero(pos_t), torch.count_nonzero(neg_t), torch.count_nonzero(pos_label), torch.count_nonzero(neg_label))
                        break
                    all_label = all_label + label.tolist()
                #except Exception as e:
                #    if data is None :
                #        affected_batches += 1
                else :
                    continue
            else :
                continue
        print("173, The number of affected batches in the test procedure : ", affected_batches)
    auc = calc_auc(all_label, all_logit)
    aupr = calc_aupr(all_label, all_logit)
    other_metrics = calc_other_metrics(all_label, all_logit)
    return auc, aupr, other_metrics, all_logit, all_label
