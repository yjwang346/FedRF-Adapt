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
from v1_FedRF_Adapt_model import notca_federated_source_and_target_model, notca_federated_source_and_target_model_office_caltech, twice_notca_federated_source_and_target_model_office_caltech
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

device = torch.device("cuda:3")
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

def digit_five_ForeverDataIterator(source_dataset1, source_dataset2, source_dataset3, source_dataset4, target_dataset, batch_ns, batch_nt):
    source_loader1 = torch.utils.data.DataLoader( source_dataset1, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    source_loader2 = torch.utils.data.DataLoader( source_dataset2, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    source_loader3 = torch.utils.data.DataLoader( source_dataset3, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    source_loader4 = torch.utils.data.DataLoader( source_dataset4, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True) 
    target_loader = torch.utils.data.DataLoader( target_dataset, 
                                            batch_size=batch_nt, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    return [source_loader1, source_loader2, source_loader3, source_loader4, target_loader] 

def office_caltech_ForeverDataIterator(source_dataset1, source_dataset2, source_dataset3, target_dataset, batch_ns, batch_nt):
    source_loader1 = torch.utils.data.DataLoader( source_dataset1, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    source_loader2 = torch.utils.data.DataLoader( source_dataset2, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    source_loader3 = torch.utils.data.DataLoader( source_dataset3, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    target_loader = torch.utils.data.DataLoader( target_dataset, 
                                            batch_size=batch_nt, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    return [source_loader1, source_loader2, source_loader3, target_loader]               

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

if __name__ == '__main__':
    # data loading
    # dataset_office: "digit_five", "visda", "office31", "office_caltech_10" 
    #                 "mnist", "svhn", "usps";
    # office: "amazon", "webcam", "dslr", "caltech"
    # "digit_five": "mnist", "usps", "svhn", "mnist-m", "synthetic_digits" 

    dataset_name = "office_caltech"
    class_num = 10
    target_list = ["dslr", "caltech"] # "amazon", "webcam", 
    dataset_list = ["amazon", "webcam", "dslr", "caltech"]

    for target in target_list:
        # for Ts in [10, 20, 50, 100, 200, 400, 800]:        
        for Ts in [100]:
            # Setup output
            data = time.strftime("%Y%m%d-%H-%M", time.localtime())
            exp = data + 'Average'
            # log_file = 'noTCA_no_source_init_random_subset_{}_{}.txt'.format(target, exp)
            log_file = '{}_Ts={}_selected_source_and_RFMMD_source_init_all_clients_{}_{}.txt'.format(dataset_name, Ts, target, exp)

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
            sigma = 5 # based on office-10 history
            kernel = 'rbf'
            feature_dim = 2048 # 25088
            highest_acc = []
            highest_ave_acc = []
            trade_off1 = 1
            trade_off2 = 1
            mu_list = [10]

            num_classes = 10
            class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

            # training parameters on office-caltech
            epoches = 20
            iter_num = 20
            pretrain_epoch = 2
            target_self_adapt_epoch = 20
            exp_repeated_num = 1 if target == "dslr" else 2

            log("epoches = {}, iter_num = {}, and Ts = {}.".format(epoches, iter_num, Ts))
            log("Ts classifier aggragation cycle.")
            log("pretrain_epoch = {}, source init means that when epoch < {}, loss = cls_loss in order to init source model, and when epoch >= {}, loss = cls_loss + source_mmd_loss + target_mmd_loss".format(pretrain_epoch, pretrain_epoch, pretrain_epoch))
            log("only RF-based MMD participants in MMD Loss calculation.")

            for currently_i in range(exp_repeated_num):
                batch_ns = 128
                batch_nt = batch_ns
        
                source_dataset_list = dataset_list.copy()
                source_dataset_list.remove(target)
                
                # loading datasets                        
                if dataset_name == "digit_five":
                    source_dataset1 = digit_five_dataset_federated(source_dataset_list[0])
                    source_dataset2 = digit_five_dataset_federated(source_dataset_list[1])
                    source_dataset3 = digit_five_dataset_federated(source_dataset_list[2])
                    source_dataset4 = digit_five_dataset_federated(source_dataset_list[3])
                    target_dataset = digit_five_dataset_federated(target)
                    ns_1, ns_2, ns_3, ns_4, nt = len(source_dataset1), len(source_dataset2), len(source_dataset3), len(source_dataset4), len(target_dataset)
                    ns_sum = ns_1 + ns_2 + ns_3 + ns_4
                    ns_list = [ns_1, ns_2, ns_3, ns_4, nt]
                
                    # DataLoader and ForeverDataIterator
                    loader_list = digit_five_ForeverDataIterator(source_dataset1, source_dataset2, source_dataset3, source_dataset4, target_dataset, batch_ns, batch_nt)
                    train_source_iter1 = ForeverDataIterator(loader_list[0])
                    train_source_iter2 = ForeverDataIterator(loader_list[1])
                    train_source_iter3 = ForeverDataIterator(loader_list[2])
                    train_source_iter4 = ForeverDataIterator(loader_list[3])
                    train_target_iter = ForeverDataIterator(loader_list[4])

                elif dataset_name == "office_caltech":
                    source_dataset1 = office_10_dataset_federated(source_dataset_list[0])
                    source_dataset2 = office_10_dataset_federated(source_dataset_list[1])
                    source_dataset3 = office_10_dataset_federated(source_dataset_list[2])
                    target_dataset = office_10_dataset_federated(target)
                    ns_1, ns_2, ns_3, nt = len(source_dataset1), len(source_dataset2), len(source_dataset3), len(target_dataset)
                    ns_sum = ns_1 + ns_2 + ns_3
                    ns_list = [ns_1, ns_2, ns_3, nt]
                    
                    # DataLoader and ForeverDataIterator
                    loader_list = office_caltech_ForeverDataIterator(source_dataset1, source_dataset2, source_dataset3, target_dataset, batch_ns, batch_nt)
                    train_source_iter1 = ForeverDataIterator(loader_list[0])
                    train_source_iter2 = ForeverDataIterator(loader_list[1])
                    train_source_iter3 = ForeverDataIterator(loader_list[2])
                    train_target_iter = ForeverDataIterator(loader_list[3])



                log("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                log('currently_i = {}'.format(currently_i))
                log('Ts = {}'.format(Ts))
                log("currently_i = {}".format(currently_i))
                log("Experiments for Robustness (Fig 5 in the paper) on communication cycle num Ts = {}:".format(Ts))
                log('dataset = {}, \t source1 = {}, \t source2 = {}, \t source3 = {},\t target = {}'.format(dataset_name, source_dataset_list[0], source_dataset_list[1], source_dataset_list[2], target)) 
                log("sigma = {}, kernel = {}, feature_dim = {}, n_features=(feature_dim // 2)".format(sigma, kernel, feature_dim))               

                # model setting
                # classifier = notca_federated_source_and_target_model(dataset=dataset_name, feature_dim=feature_dim,class_num=class_num).to(device)
                classifier = twice_notca_federated_source_and_target_model_office_caltech(feature_dim=feature_dim, n_features=(feature_dim//2), sigma=sigma, kernel='rbf', class_num=class_num).to(device)
                value_loss = nn.CrossEntropyLoss()
                learning_rate = 2e-3


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
                        x_s2, labels_s2 = next(train_source_iter2)
                        x_s3, labels_s3 = next(train_source_iter3)
                        x_t, labels_t = next(train_target_iter)
                        x_s1 = x_s1.to(device)
                        x_s2 = x_s2.to(device)
                        x_s3 = x_s3.to(device)
                        x_t = x_t.to(device)

                        labels_s1 = labels_s1.to(torch.int64).to(device)
                        labels_s2 = labels_s2.to(torch.int64).to(device)
                        labels_s3 = labels_s3.to(torch.int64).to(device)
                        labels_t = labels_t.to(torch.int64).to(device)

                        if dataset_name == "digit_fice":
                            x_s4, labels_s4 = next(train_source_iter4)
                            x_s4 = x_s4.to(device)
                            labels_s4 = labels_s4.to(torch.int64).to(device)

                            x_s1, x_s2, x_s3, x_s4, x_t, ys1, ys2, ys3, ys4, yt = classifier(x_s1, x_s2, x_s3, x_s4, x_t)

                        elif dataset_name == "office_caltech":
                            x_s1, x_s2, x_s3, x_t, ys1, ys2, ys3, yt = classifier(x_s1, x_s2, x_s3, x_t)

                        cls_loss1 = value_loss(ys1, torch.flatten(labels_s1, start_dim=0))
                        cls_loss2 = value_loss(ys2, torch.flatten(labels_s2, start_dim=0))
                        cls_loss3 = value_loss(ys3, torch.flatten(labels_s3, start_dim=0))
                        
                        # get a random subset of 4 source domains
                        passing_num1 = torch.tensor(random.randint(0, 3)).to(device) 
                        # passing_num2 = torch.tensor(random.randint(0, passing_num1)).to(device)
                        random_i_list = random.sample(range(3), passing_num1)
                        all_i_list = list(range(3))
                        # random_i_list2 = random.sample(random_i_list, passing_num2)
                        selected_list = all_i_list

                        # mmd_loss calculating
                        source_mmd_loss = torch.tensor(0.0)
                        target_mmd_loss = torch.tensor(0.0)
                        temp_target_Sigma_yi = x_t.detach()
                        temp_source1_Sigma_yi = x_s1.detach()
                        temp_source2_Sigma_yi = x_s2.detach()
                        temp_source3_Sigma_yi = x_s3.detach()
                        all_temp_source_Sigma_yi_list = [temp_source1_Sigma_yi, temp_source2_Sigma_yi, temp_source3_Sigma_yi]
                        
                        mmd_loss_source1 = classifier.mmd_loss(x_s1, temp_target_Sigma_yi) 
                        mmd_loss_source2 = classifier.mmd_loss(x_s2, temp_target_Sigma_yi)
                        mmd_loss_source3 = classifier.mmd_loss(x_s3, temp_target_Sigma_yi)
                        source_mmd_loss_list = [mmd_loss_source1, mmd_loss_source2, mmd_loss_source3]
                        source_mmd_loss = sum([source_mmd_loss_list[random_i] for random_i in all_i_list])

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
                        mmd_loss_target1 = classifier.mmd_loss(temp_source1_Sigma_yi, x_t)
                        mmd_loss_target2 = classifier.mmd_loss(temp_source2_Sigma_yi, x_t)
                        mmd_loss_target3 = classifier.mmd_loss(temp_source3_Sigma_yi, x_t)
                        target_mmd_loss_list = [mmd_loss_target1, mmd_loss_target2, mmd_loss_target3]
                        target_mmd_loss = sum([target_mmd_loss_list[random_i] for random_i in selected_list])

                        # loss function
                        cls_loss = cls_loss1 + cls_loss2 + cls_loss3
                        target_mmd_loss = trade_off2 * target_mmd_loss

                        if epoch < pretrain_epoch:
                            loss = cls_loss
                        else:
                            loss = cls_loss + source_mmd_loss + target_mmd_loss 

                        # model backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()                

                        # model evaluation
                        cls_acc1 = accuracy(ys1, labels_s1)
                        cls_acc2 = accuracy(ys2, labels_s2)
                        cls_acc3 = accuracy(ys3, labels_s3)
                        trans_cls_acc, _, _cls_str = wyj_check_y_label(yt, labels_t, class_names)

                        tar_acc = tar_acc + trans_cls_acc
                        if trans_cls_acc > max_acc:
                            max_acc = trans_cls_acc
                        """"""
                        if random_i_list != []:
                            # update source classifiers
                            if epoch >= pretrain_epoch:
                                # if (epoch * iter_num + iter_i) % Ts == 0:
                                classifier.FedAvg_classifier(selected_list) # random_i_list 

                    ave_trans_acc = tar_acc / iter_num
                    if ave_trans_acc > max_ace_acc:
                        max_ace_acc = ave_trans_acc

                    # full evaluate on target dataset
                    all_acc_point, mean_class_acc, cls_acc_str = wyj_check_y_label(yt, labels_t, class_names)

                    if epoch < pretrain_epoch:  
                        log("$$$ epoch={} < pretrain_epoch={}, only source model initialization, no mmd_loss participants in loss.backward()".format(epoch, pretrain_epoch))
                    if epoch >= target_self_adapt_epoch:
                        log("$$$ epoch={} < target_self_adapt_epoch={}, only source target trains, loss = los_IM".format(epoch, target_self_adapt_epoch))
                    log('epoch = {} , \t cls_acc1({}) = {:.5}, cls_acc2({}) = {:.5}, cls_acc3({}) = {:.5}'.format(epoch, source_dataset_list[0], cls_acc1.item(), source_dataset_list[1], cls_acc2.item(), source_dataset_list[2], cls_acc3.item()))
                    log('\t TARGET: mean acc={:.3%}, mean class acc={:.3%}'.format(all_acc_point, mean_class_acc))
                    log('\t \t: this_iteration_avg_ace_acc={}'.format(ave_trans_acc))
                    log('\t \t: max_avg_ace_acc={}'.format(max_ace_acc))
                    log('\t   per class:  {}'.format(cls_acc_str))     
                    log('selected_list = {}'.format(selected_list))  
                    # log(' source_mmd_loss1 = {}, \t source_mmd_loss2 = {}, \t source_mmd_loss3 = {}, \t source_mmd_loss4 = {}, \t target_mmd_loss = {} '.format(mmd_loss_source1.item(), mmd_loss_source2.item(), mmd_loss_source3.item(), mmd_loss_source4.item(), mmd_loss_target.item()))                 

                highest_acc.append(max_acc.item())
                highest_ave_acc.append(max_ace_acc.item())
                log("Ts = {}".format(Ts))
                log('highest_acc = {}, \t highest_ave_acc = {}'.format(highest_acc, highest_ave_acc))

                