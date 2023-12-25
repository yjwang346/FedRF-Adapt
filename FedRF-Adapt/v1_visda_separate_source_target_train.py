from select import select
import sys
sys.path.append("..")

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from data import numpyDataset, ForeverDataIterator, MyDataset
import os
import pickle
from torchvision import datasets, transforms
from linear_classifier import bottleneck
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, Adam
import torch
torch.cuda.current_device()
# from v2_federated_model import federated_source_and_target_model
from v1_no_TCA_multi_source_model import notca_federated_source_and_target_model, notca_federated_source_and_target_model_office_caltech, twice_notca_federated_source_and_target_model_office_caltech, visda_RFMMD_notca_federated_source_target_model
from TorchRandomF import RFF_perso
# from RandomF import RFF_perso
from torchvision import models, transforms as T
from visda_dataset import get_dataset
import time
from tqdm import tqdm
from dataset_federated import dataset_federated, digit_five_dataset_federated, office_10_dataset_federated
import random
import numpy as np
import math
# os.environ['max_split_size_mb'] = '2048'

device = torch.device("cuda:0")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

# 获取 accuracy
def wyj_check_y_label_for_voting(y_preds, y_sups, class_names):
    y_sups = y_sups.ravel()
    class_num_list = torch.bincount(y_sups)
    if class_num_list.size(dim=0) < len(class_names):
        class_num_list = torch.cat((class_num_list, torch.zeros(len(class_names)-class_num_list.size(dim=0)).to(y_sups.device))).to(y_sups.device)

    # acc_num = [0] * len(class_names)
    acc_num = torch.zeros(len(class_names)).to(y_sups.device)
    all_acc_num = torch.tensor(0).to(y_sups.device)
    nums = torch.tensor(y_sups.size(dim=0)).to(y_sups.device)
    
    # get every source classifier result on target transfer faeture, 
    # y_preds.size()=>(n, 4, 10),4 is the number of source clients, and 10 is the class num; we first get the pred_class of each source clents; then every target feature got a size=(4,1) result, so we need to squeeze tensor to get a size=(4,) result;
    # 
    _, y_preds = torch.topk(y_preds, 1, dim=2)
    # y_preds = y_preds.ravel()
    y_preds = torch.squeeze(y_preds)
    y_preds, _ =torch.mode(y_preds, dim=1) # y_preds is size=(n,) tensor;

    for i in range(nums):
        if y_preds[i] == y_sups[i]:
            all_acc_num += 1
            acc_num[y_preds[i]] += 1

    prd_acc_list = acc_num.to(torch.float32) / torch.maximum(class_num_list, torch.tensor(1.0).to(y_sups.device))
    mean_class_acc = prd_acc_list.mean()
    
    aug_cls_acc_str = ',  '.join(['{}: {:.3%}'.format(class_names[cls_i], acc_num[cls_i]/class_num_list[cls_i])for cls_i in range(len(class_names))])

    return all_acc_num/nums, mean_class_acc, aug_cls_acc_str 


def wyj_check_y_label(y_preds, y_sups, class_names):
    y_sups = y_sups.ravel()
    class_num_list = torch.bincount(y_sups)
    if class_num_list.size(dim=0) < len(class_names):
        class_num_list = torch.cat((class_num_list, torch.zeros(len(class_names)-class_num_list.size(dim=0)).to(y_sups.device))).to(y_sups.device)

    acc_num = torch.zeros(len(class_names)).to(y_sups.device)
    all_acc_num = torch.tensor(0).to(y_sups.device)
    nums = torch.tensor(y_sups.size(dim=0)).to(y_sups.device)

    _, y_preds = torch.topk(y_preds, 1, dim=1)
    y_preds = y_preds.ravel()

    for i in range(nums):
        if y_preds[i] == y_sups[i]:
            all_acc_num += 1
            acc_num[y_preds[i]] += 1

    prd_acc_list = acc_num.to(torch.float32) / torch.maximum(class_num_list, torch.tensor(1.0).to(y_sups.device))
    mean_class_acc = prd_acc_list.mean()
    
    aug_cls_acc_str = ',  '.join(['{}: {:.3%}'.format(class_names[cls_i], acc_num[cls_i]/class_num_list[cls_i])for cls_i in range(len(class_names))])

    return all_acc_num/nums, mean_class_acc, aug_cls_acc_str

def accuracy(output, target):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = 1 # max(topk)
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        # _, labels = target.topk(maxk, 1, True, True)
        _, pred = torch.topk(output, 1, dim=1)
        pred = pred.flatten()
        # pred = pred.t()
        # print('pred.size() = ', pred.size())
        # print('target[None].size() = ', target[None].size())
        correct = pred.eq(target)
        
        correct_k = correct.flatten().sum(dtype=torch.float32)
        res = (correct_k * (100.0 / batch_size))

        return res

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

if __name__ == '__main__':
    # data loading
    # dataset_office: "digit_five", "visda", "office31", "office_caltech_10" 
    #                 "mnist", "svhn", "usps";
    # "office_caltech": "amazon", "webcam", "dslr", "caltech"
    # "digit_five": "mnist", "usps", "svhn", "mnist-m", "synthetic_digits" 

    dataset_name = "visda"

    # for Ts in [10, 20, 50, 100, 200, 400, 800]:        
    for Ts in [100]:
        # Setup output
        data = time.strftime("%Y%m%d-%H-%M", time.localtime())
        exp = data + 'Classifier_Average'
        # log_file = 'noTCA_no_source_init_random_subset_{}_{}.txt'.format(target, exp)
        log_file = 'visda_Ts={}_real_real_three_stage_only_RF_MMD_noTCA_ssource_init_{}.txt'.format(exp)

        if log_file is not None:
            if os.path.exists(log_file):
                print('Output log file {} already exists'.format(log_file))
        def log(text, end='\n'):
            print(text)
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(text + end)
                    f.flush()
                    f.close() 
                    
        # hyper-parameters
        sigma = 0.5
        kernel = 'rbf'
        feature_dim = 2048 # 25088
        highest_acc = []
        highest_ave_acc = []
        trade_off1 = 1
        trade_off2 = 1
        mu_list = [10]

        num_classes = 10
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        epoches = 30 
        iter_num = 200 
        pretrain_epoch = 1
        target_self_adapt_epoch = 25
        exp_repeated_num = 5

        log("epoches = {}, iter_num = {}.".format(epoches, iter_num))
        log("no TCA version means that the model pipeline only contains feature extractor and classifier.")
        log("three stage means that the pretrain_epoch = {}, only need source cls_loss, the the last half loss = cls_loss + source_mmd_loss + target_mmd_loss, the third stage only target client trains via loss = loss_IM".format(pretrain_epoch))
        # log("source init means that when epoch < (3), loss = cls_loss in order to init source model, and when epoch >= (3), loss = cls_loss + source_mmd_loss + target_mmd_loss")
        log("only RF-based MMD participants in MMD Loss calculation.")
        # log("Twice MMD means that mmd_loss = MMD(X_S, X_T) + MMD(\Sigma_S, \Sigma_T)")

        for currently_i in range(exp_repeated_num):
            batch_ns = 128
            batch_nt = batch_ns
            
            # loading datasets            
            if dataset_name == "visda":
                root = "/data/dataset/visda-2017"
                visda_source = "train"
                visda_target = "validation"
                train_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]) 
                source_dataset, target_dataset, num_classes, class_names = \
                    get_dataset(root, visda_source, visda_target, train_transform, val_transform=train_transform)     
                class_num = len(class_names)           
                
                source_loader1 = torch.utils.data.DataLoader( source_dataset, 
                                                        batch_size=batch_ns, 
                                                        num_workers=4, 
                                                        shuffle=True, 
                                                        drop_last=True)
                target_loader = torch.utils.data.DataLoader( target_dataset, 
                                                        batch_size=batch_nt, 
                                                        num_workers=4, 
                                                        shuffle=True, 
                                                        drop_last=True)
                # DataLoader and ForeverDataIterator
                train_source_iter1 = ForeverDataIterator(source_loader1)
                train_target_iter = ForeverDataIterator(target_loader)


            log("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            log('currently_i = {}'.format(currently_i))
            log('Ts = {}'.format(Ts))
            log("currently_i = {}".format(currently_i))
            log("Experiments for Robustness (Fig 5 in the paper) on communication cycle num Ts = {}:".format(Ts))
            log('dataset = {}, \t source = {}, \t target = {}'.format(dataset_name, visda_source, visda_target)) 
            log("sigma = {}, kernel = {}, feature_dim = {}, n_features=(feature_dim // 2)".format(sigma, kernel, feature_dim))               

            # model setting
            # classifier = notca_federated_source_and_target_model(dataset=dataset_name, feature_dim=feature_dim,class_num=class_num).to(device)
            # classifier = twice_notca_federated_source_and_target_model_office_caltech(feature_dim=feature_dim, n_features=(feature_dim//2), sigma=sigma, kernel='rbf', class_num=class_num).to(device)
            classifier = visda_RFMMD_notca_federated_source_target_model(feature_dim=feature_dim, n_features=(feature_dim//2), sigma=sigma, kernel='rbf', class_num=class_num).to(device)
            value_loss = nn.CrossEntropyLoss()
            learning_rate = 2e-4


            # log setting message
            model_setting = 'resnet50'
            log('Model: {} '.format(model_setting))
            log('       :  learning_rate={lr}, batch_ns={batch_ns}, batch_nt={batch_nt}, iter_num={iter_num}, trade_off1={trade_off1}, '.format(lr=learning_rate, batch_ns=batch_ns, batch_nt=batch_nt, iter_num=iter_num, trade_off1=trade_off1))
            optimizer = Adam(classifier.parameters(), lr=learning_rate)

            max_acc = torch.tensor(0).to(device)
            max_ace_acc = torch.tensor(0).to(device)
            mmd_average_list = [1, 1 ,1, 1]

            # train
            init_flag = 0
            target_self_train_flag = 0
            for epoch in range(epoches):
                classifier.train()
                source_acc = torch.tensor(0).to(device)
                tar_acc = torch.tensor(0).to(device)
                source_num = torch.tensor(4).to(device)
                for iter_i in tqdm(range(iter_num)):
                    x_s1, labels_s1 = next(train_source_iter1)
                    x_t, labels_t = next(train_target_iter)
                    x_s1 = x_s1.to(device)
                    x_t = x_t.to(device)

                    labels_s1 = labels_s1.to(torch.int64).to(device)
                    labels_t = labels_t.to(torch.int64).to(device)

                    x_s1, x_t, ys1, yt = classifier(x_s1, x_t)
                    
                    cls_loss = value_loss(ys1, torch.flatten(labels_s1, start_dim=0))
                    
                    # mmd_loss calculating
                    temp_target_Sigma_yi = x_t.detach()
                    temp_source1_Sigma_yi = x_s1.detach()
                    
                    source_mmd_loss = classifier.mmd_loss(x_s1, temp_target_Sigma_yi)

                    # if (epoch * iter_num + iter_i) % Ts == 0:
                    # if random_i_list != []:
                    # selected_temp_source_Sigma_yi = torch.stack([all_temp_source_Sigma_yi_list[random_i] for random_i in random_i_list]).sum(dim=0)
                    # all_temp_source_Sigma_yi = torch.stack([all_temp_source_Sigma_yi_list[random_i] for random_i in selected_list]).sum(dim=0)
                    # mmd_loss_target = classifier.mmd_loss(selected_temp_source_Sigma_yi, x_t)

                    # a random source message
                    # random_source_ind = random.randint(0, 3)
                    # selected_source_xs = all_temp_source_Sigma_yi_list[random_source_ind]
                    # mmd_loss_target = classifier.mmd_loss(selected_source_xs, x_t)

                    # random subset of source message
                    target_mmd_loss = classifier.mmd_loss(temp_source1_Sigma_yi, x_t)

                    # loss function
                    target_mmd_loss = trade_off2 * target_mmd_loss

                    # 
                    if epoch < (3):           
                        loss = cls_loss
                    elif epoch < target_self_adapt_epoch:
                        loss = cls_loss + source_mmd_loss + target_mmd_loss 
                    else:
                        if target_self_train_flag == 0:
                            classifier.only_target_train()
                            target_self_train_flag = 1
                        target_output_p = torch.exp(yt)
                        entropy_loss = torch.mean(Entropy(target_output_p))
                        msoftmax = target_output_p.mean(dim=1)
                        entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                        entropy_loss = entropy_loss
                        loss = entropy_loss

                    # model backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                

                    # model evaluation
                    source_cls_acc = accuracy(ys1, labels_s1)
                    trans_cls_acc, _, _cls_str = wyj_check_y_label(yt, labels_t, class_names)

                    tar_acc = tar_acc + trans_cls_acc
                    if trans_cls_acc > max_acc:
                        max_acc = trans_cls_acc

                    if epoch >= pretrain_epoch and epoch < target_self_adapt_epoch:
                        classifier.FedAvg_classifier() # random_i_list 

                ave_trans_acc = tar_acc / iter_num
                if ave_trans_acc > max_ace_acc:
                    max_ace_acc = ave_trans_acc

                # full evaluate on target dataset
                all_acc_point, mean_class_acc, cls_acc_str = wyj_check_y_label(yt, labels_t, class_names)

                if epoch < (pretrain_epoch):  
                    log("$$$ epoch={} < pretrain_epoch={}, only source model initialization, no mmd_loss participants in loss.backward()".format(epoch, pretrain_epoch))
                if epoch >= target_self_adapt_epoch:
                    log("$$$ epoch={} < target_self_adapt_epoch={}, only source target trains, loss = los_IM".format(epoch, target_self_adapt_epoch))

                log('epoch = {} , \t SOURCE_acc = {:.5}'.format(epoch, source_cls_acc.item()))
                log('\t TARGET: mean acc={:.3%}, mean class acc={:.3%}'.format(all_acc_point, mean_class_acc))
                log('\t \t: this_iteration_avg_ace_acc={}'.format(ave_trans_acc))
                log('\t \t: max_avg_ace_acc={}'.format(max_ace_acc))
                log('\t   per class:  {}'.format(cls_acc_str))              

            highest_acc.append(max_acc.item())
            highest_ave_acc.append(max_ace_acc.item())
            log("Ts = {}".format(Ts))
            log('highest_acc = {}, \t highest_ave_acc = {}'.format(highest_acc, highest_ave_acc))

            