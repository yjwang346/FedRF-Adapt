from random import sample
from turtle import Turtle, forward
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from torchvision import models, transforms as T
import torch
from TorchRandomF import RFF_perso
from linear_classifier import bottleneck
import collections
import math

"""
    Based on Horizontal Setting of Federated Learning, which means that 
    Source data and Target data are in the same featre space, with a domian shift between them.
"""

def notca_federated_source_and_target_model(dataset, feature_dim, class_num=31):
    if dataset == "digit_five":
        return notca_federated_source_and_target_model_digit_five(feature_dim, class_num)
    elif dataset == "office_caltech":
        return notca_federated_source_and_target_model_office_caltech(feature_dim, class_num)

class notca_federated_source_and_target_model_digit_five(nn.Module):
    def __init__(self, feature_dim, class_num=31):
        """
        Source forward process:
            X -> conv1 -> conv2 -> classifier
        Input:
            feature_dim: faeture dimention of the output of feature extractor;
            m: final feature dimention of transfer feature, top m eigenvectors of Equation;
        """
        super(notca_federated_source_and_target_model_digit_five, self).__init__()
        # feature extractor
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv_layers1 = torch.nn.Sequential(*(list(resnet50.children())[0:7])).requires_grad_(False)
        self.source1_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source2_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source3_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source4_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.target_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        
        # classifier
        self.source1_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source2_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source3_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source4_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.target_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(False)

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_t):
        """"""
        x_s1 = self.conv_layers1(x_s1)
        x_s2 = self.conv_layers1(x_s2)
        x_s3 = self.conv_layers1(x_s3)
        x_s4 = self.conv_layers1(x_s4)
        x_t = self.conv_layers1(x_t)

        x_s1 = self.source1_conv_layers2(x_s1)
        x_s2 = self.source2_conv_layers2(x_s2)
        x_s3 = self.source3_conv_layers2(x_s3)
        x_s4 = self.source4_conv_layers2(x_s4)
        x_t = self.target_conv_layers2(x_t)

        x_s1 = torch.flatten(x_s1, start_dim=1)
        x_s2 = torch.flatten(x_s2, start_dim=1)
        x_s3 = torch.flatten(x_s3, start_dim=1)
        x_s4 = torch.flatten(x_s4, start_dim=1)
        x_t = torch.flatten(x_t, start_dim=1)

        # each party runs independently
        x_s1 = self.sample_norm(x_s1)
        x_s2 = self.sample_norm(x_s2)
        x_s3 = self.sample_norm(x_s3)
        x_s4 = self.sample_norm(x_s4)
        x_t = self.sample_norm(x_t)

        # generate new transfer features
        ys1 = self.source1_classifier(x_s1)        
        ys2 = self.source2_classifier(x_s2)
        ys3 = self.source3_classifier(x_s3)
        ys4 = self.source3_classifier(x_s4)
        yt = self.target_classifier(x_t)

        return x_s1, x_s2, x_s3, x_s4, x_t, ys1, ys2, ys3, ys4, yt

    # CLassifier Layer Aggregation
    def FedAvg_classifier(self, subset_list=None):
        models_num = 4 
        models_list = [self.source1_classifier, self.source2_classifier, self.source3_classifier, self.source4_classifier]        
        if subset_list != None:
            models_num = len(subset_list)
            models_list = [models_list[subset_i] for subset_i in subset_list]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + models_state_dict[i][key]
            fed_state_dict[key] = key_sum / models_num
        # update fed weights to fed model
        self.source1_classifier.load_state_dict(fed_state_dict)
        self.source2_classifier.load_state_dict(fed_state_dict)
        self.source3_classifier.load_state_dict(fed_state_dict)
        self.source4_classifier.load_state_dict(fed_state_dict)
        self.target_classifier.load_state_dict(fed_state_dict)  

    def mmd_loss(self, x_s, x_t):
        batch_ns = x_s.size(0)
        batch_nt = x_t.size(0)
        x_s_ys = torch.sum(x_s, dim=0) / batch_ns
        x_t_yt = torch.sum(x_t, dim=0) / batch_nt
        Sigma = x_s_ys - x_t_yt
        Sigma = Sigma.unsqueeze(0)
        # Sigma = self.Wrf(Sigma)
        return torch.mm(Sigma, Sigma.T)
        
    def sample_norm(self, x_new):
        x_new = x_new.T
        x_new = x_new / (torch.linalg.norm(x_new, dim=0) + 1e-5)
        x_new = x_new.T
        return x_new
    
class digit_five_only_RF_MMD_notca_federated_source_and_target_model(nn.Module):
    def __init__(self, feature_dim, n_features, sigma, kernel='rbf', class_num=31):
        """
        Source forward process:
            X -> conv1 -> conv2 -> classifier
        Input:
            feature_dim: faeture dimention of the output of feature extractor;
            m: final feature dimention of transfer feature, top m eigenvectors of Equation;
        """
        super(digit_five_only_RF_MMD_notca_federated_source_and_target_model, self).__init__()
        # feature extractor
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv_layers1 = torch.nn.Sequential(*(list(resnet50.children())[0:7])).requires_grad_(False)
        self.source1_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source2_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source3_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source4_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.target_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        
        # random feature map part, with unified random weights
        self.rf_map = federated_rf_map(n_features=n_features, sigma=sigma, feature_dim=feature_dim, kernel=kernel, domain='source')
        
        # classifier
        self.source1_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source2_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source3_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source4_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.target_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(False)

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_t):
        """"""
        x_s1 = self.conv_layers1(x_s1)
        x_s2 = self.conv_layers1(x_s2)
        x_s3 = self.conv_layers1(x_s3)
        x_s4 = self.conv_layers1(x_s4)
        x_t = self.conv_layers1(x_t)

        x_s1 = self.source1_conv_layers2(x_s1)
        x_s2 = self.source2_conv_layers2(x_s2)
        x_s3 = self.source3_conv_layers2(x_s3)
        x_s4 = self.source4_conv_layers2(x_s4)
        x_t = self.target_conv_layers2(x_t)

        x_s1 = torch.flatten(x_s1, start_dim=1)
        x_s2 = torch.flatten(x_s2, start_dim=1)
        x_s3 = torch.flatten(x_s3, start_dim=1)
        x_s4 = torch.flatten(x_s4, start_dim=1)
        x_t = torch.flatten(x_t, start_dim=1)

        # each party runs independently
        x_s1 = self.sample_norm(x_s1)
        x_s2 = self.sample_norm(x_s2)
        x_s3 = self.sample_norm(x_s3)
        x_s4 = self.sample_norm(x_s4)
        x_t = self.sample_norm(x_t)

        # RF-TCA message
        source1_Sigma_i, source1_Sigma_yi = self.rf_map(x_s1)
        source2_Sigma_i, source1_Sigma_yi = self.rf_map(x_s2)
        source3_Sigma_i, source1_Sigma_yi = self.rf_map(x_s3)
        source4_Sigma_i, source1_Sigma_yi = self.rf_map(x_s4)
        target_Sigma_i, target_Sigma_yi = self.rf_map(x_t)

        # generate new transfer features
        source1_Sigma_i = self.sample_norm(source1_Sigma_i.T)
        source2_Sigma_i = self.sample_norm(source2_Sigma_i.T)
        source3_Sigma_i = self.sample_norm(source3_Sigma_i.T)
        source4_Sigma_i = self.sample_norm(source4_Sigma_i.T)
        target_Sigma_i = self.sample_norm(target_Sigma_i.T)
        ys1 = self.source1_classifier(x_s1)        
        ys2 = self.source2_classifier(x_s2)
        ys3 = self.source3_classifier(x_s3)
        ys4 = self.source3_classifier(x_s4)
        yt = self.target_classifier(x_t)

        # transfer message
        # z_s1 = torch.add(x_s1, source1_Sigma_i)
        # z_s2 = torch.add(x_s2, source2_Sigma_i)
        # z_s3 = torch.add(x_s3, source3_Sigma_i)
        # z_s4 = torch.add(x_s4, source4_Sigma_i)
        # z_t = torch.add(x_t, target_Sigma_i)

        # return z_s1, z_s2, z_s3, z_s4, z_t, ys1, ys2, ys3, ys4, yt
        return source1_Sigma_i, source2_Sigma_i, source3_Sigma_i, source4_Sigma_i, target_Sigma_i, ys1, ys2, ys3, ys4, yt

    # CLassifier Layer Aggregation
    def FedAvg_classifier(self, subset_list=None):
        models_num = 4 
        models_list = [self.source1_classifier, self.source2_classifier, self.source3_classifier, self.source4_classifier]        
        if subset_list != None:
            models_num = len(subset_list)
            models_list = [models_list[subset_i] for subset_i in subset_list]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + models_state_dict[i][key]
            fed_state_dict[key] = key_sum / models_num
        # update fed weights to fed model
        self.source1_classifier.load_state_dict(fed_state_dict)
        self.source2_classifier.load_state_dict(fed_state_dict)
        self.source3_classifier.load_state_dict(fed_state_dict)
        self.source4_classifier.load_state_dict(fed_state_dict)
        self.target_classifier.load_state_dict(fed_state_dict)  

    def mmd_loss(self, x_s, x_t):
        batch_ns = x_s.size(0)
        batch_nt = x_t.size(0)
        x_s_ys = torch.sum(x_s, dim=0) / batch_ns
        x_t_yt = torch.sum(x_t, dim=0) / batch_nt
        Sigma = x_s_ys - x_t_yt
        Sigma = Sigma.unsqueeze(0)
        # Sigma = self.Wrf(Sigma)
        return torch.mm(Sigma, Sigma.T)
    
    def only_target_train(self):
        self.source1_conv_layers2.requires_grad_(False)
        self.source2_conv_layers2.requires_grad_(False)
        self.source3_conv_layers2.requires_grad_(False)
        self.source4_conv_layers2.requires_grad_(False)
        self.target_conv_layers2.requires_grad_(True)

        self.source1_classifier.requires_grad_(False)
        self.source2_classifier.requires_grad_(False)
        self.source3_classifier.requires_grad_(False)
        self.source4_classifier.requires_grad_(False)
        self.target_classifier.requires_grad_(False)
        
    def sample_norm(self, x_new):
        x_new = x_new.T
        x_new = x_new / (torch.linalg.norm(x_new, dim=0) + 1e-5)
        x_new = x_new.T
        return x_new
    


class notca_federated_source_and_target_model_office_caltech(nn.Module):
    def __init__(self, feature_dim, class_num=31):
        """
        Source forward process:
            X -> conv1 -> conv2 -> classifier
        Input:
            feature_dim: faeture dimention of the output of feature extractor;
            m: final feature dimention of transfer feature, top m eigenvectors of Equation;
        """
        super(notca_federated_source_and_target_model_office_caltech, self).__init__()
        # feature extractor
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv_layers1 = torch.nn.Sequential(*(list(resnet50.children())[0:7])).requires_grad_(False)
        self.source1_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source2_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source3_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.target_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        
        # classifier
        self.source1_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source2_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source3_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.target_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(False)

    def forward(self, x_s1, x_s2, x_s3, x_t):
        """"""
        x_s1 = self.conv_layers1(x_s1)
        x_s2 = self.conv_layers1(x_s2)
        x_s3 = self.conv_layers1(x_s3)
        x_t = self.conv_layers1(x_t)

        x_s1 = self.source1_conv_layers2(x_s1)
        x_s2 = self.source2_conv_layers2(x_s2)
        x_s3 = self.source3_conv_layers2(x_s3)
        x_t = self.target_conv_layers2(x_t)

        x_s1 = torch.flatten(x_s1, start_dim=1)
        x_s2 = torch.flatten(x_s2, start_dim=1)
        x_s3 = torch.flatten(x_s3, start_dim=1)
        x_t = torch.flatten(x_t, start_dim=1)

        # each party runs independently
        x_s1 = self.sample_norm(x_s1)
        x_s2 = self.sample_norm(x_s2)
        x_s3 = self.sample_norm(x_s3)
        x_t = self.sample_norm(x_t)

        # generate new transfer features
        ys1 = self.source1_classifier(x_s1)        
        ys2 = self.source2_classifier(x_s2)
        ys3 = self.source3_classifier(x_s3)
        yt = self.target_classifier(x_t)

        return x_s1, x_s2, x_s3, x_t, ys1, ys2, ys3, yt

    # CLassifier Layer Aggregation
    def FedAvg_classifier(self, subset_list=None):
        models_num = 3 
        models_list = [self.source1_classifier, self.source2_classifier, self.source3_classifier]        
        if subset_list != None:
            models_num = len(subset_list)
            models_list = [models_list[subset_i] for subset_i in subset_list]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + models_state_dict[i][key]
            fed_state_dict[key] = key_sum / models_num
        # update fed weights to fed model
        self.source1_classifier.load_state_dict(fed_state_dict)
        self.source2_classifier.load_state_dict(fed_state_dict)
        self.source3_classifier.load_state_dict(fed_state_dict)
        self.target_classifier.load_state_dict(fed_state_dict)  

    def mmd_loss(self, x_s, x_t):
        batch_ns = x_s.size(0)
        batch_nt = x_t.size(0)
        x_s_ys = torch.sum(x_s, dim=0) / batch_ns
        x_t_yt = torch.sum(x_t, dim=0) / batch_nt
        Sigma = x_s_ys - x_t_yt
        Sigma = Sigma.unsqueeze(0)
        # Sigma = self.Wrf(Sigma)
        return torch.mm(Sigma, Sigma.T)

    def sample_norm(self, x_new):
        x_new = x_new.T
        x_new = x_new / (torch.linalg.norm(x_new, dim=0) + 1e-5)
        x_new = x_new.T
        return x_new



class twice_notca_federated_source_and_target_model_office_caltech(nn.Module):
    def __init__(self, feature_dim, n_features, sigma, kernel='rbf', class_num=31):
        """
        Source forward process:
            X -> conv1 -> conv2 -> classifier
        Input:
            feature_dim: faeture dimention of the output of feature extractor;
            m: final feature dimention of transfer feature, top m eigenvectors of Equation;
        """
        super(twice_notca_federated_source_and_target_model_office_caltech, self).__init__()
        # feature extractor
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv_layers1 = torch.nn.Sequential(*(list(resnet50.children())[0:7])).requires_grad_(False)
        self.source1_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source2_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source3_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.target_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        
        
        # random feature map part, with unified random weights
        self.rf_map = federated_rf_map(n_features=n_features, sigma=sigma, feature_dim=feature_dim, kernel=kernel, domain='source')
        
        # classifier
        self.source1_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source2_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.source3_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.target_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(False)

    def forward(self, x_s1, x_s2, x_s3, x_t):
        """"""
        x_s1 = self.conv_layers1(x_s1)
        x_s2 = self.conv_layers1(x_s2)
        x_s3 = self.conv_layers1(x_s3)
        x_t = self.conv_layers1(x_t)

        x_s1 = self.source1_conv_layers2(x_s1)
        x_s2 = self.source2_conv_layers2(x_s2)
        x_s3 = self.source3_conv_layers2(x_s3)
        x_t = self.target_conv_layers2(x_t)

        x_s1 = torch.flatten(x_s1, start_dim=1)
        x_s2 = torch.flatten(x_s2, start_dim=1)
        x_s3 = torch.flatten(x_s3, start_dim=1)
        x_t = torch.flatten(x_t, start_dim=1)

        # each party runs independently
        x_s1 = self.sample_norm(x_s1)
        x_s2 = self.sample_norm(x_s2)
        x_s3 = self.sample_norm(x_s3)
        x_t = self.sample_norm(x_t)

        # RF-TCA message
        source1_Sigma_i, source1_Sigma_yi = self.rf_map(x_s1)
        source2_Sigma_i, source1_Sigma_yi = self.rf_map(x_s2)
        source3_Sigma_i, source1_Sigma_yi = self.rf_map(x_s3)
        target_Sigma_i, target_Sigma_yi = self.rf_map(x_t)

        # generate new transfer features
        source1_Sigma_i = self.sample_norm(source1_Sigma_i.T)
        source2_Sigma_i = self.sample_norm(source2_Sigma_i.T)
        source3_Sigma_i = self.sample_norm(source3_Sigma_i.T)
        target_Sigma_i = self.sample_norm(target_Sigma_i.T)
        ys1 = self.source1_classifier(x_s1)        
        ys2 = self.source2_classifier(x_s2)
        ys3 = self.source3_classifier(x_s3)
        yt = self.target_classifier(x_t)

        # transfer message
        # z_s1 = torch.add(x_s1, source1_Sigma_i)
        # z_s2 = torch.add(x_s2, source2_Sigma_i)
        # z_s3 = torch.add(x_s3, source3_Sigma_i)
        # z_t = torch.add(x_t, target_Sigma_i)

        # return z_s1, z_s2, z_s3, z_t, ys1, ys2, ys3, yt
        return source1_Sigma_i, source2_Sigma_i, source3_Sigma_i, target_Sigma_i, ys1, ys2, ys3, yt

    # CLassifier Layer Aggregation
    def FedAvg_classifier(self, subset_list=None):
        models_num = 3 
        models_list = [self.source1_classifier, self.source2_classifier, self.source3_classifier]        
        if subset_list != None:
            models_num = len(subset_list)
            models_list = [models_list[subset_i] for subset_i in subset_list]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + models_state_dict[i][key]
            fed_state_dict[key] = key_sum / models_num
        # update fed weights to fed model
        self.source1_classifier.load_state_dict(fed_state_dict)
        self.source2_classifier.load_state_dict(fed_state_dict)
        self.source3_classifier.load_state_dict(fed_state_dict)
        self.target_classifier.load_state_dict(fed_state_dict)  

    def mmd_loss(self, x_s, x_t):
        batch_ns = x_s.size(0)
        batch_nt = x_t.size(0)
        x_s_ys = torch.sum(x_s, dim=0) / batch_ns
        x_t_yt = torch.sum(x_t, dim=0) / batch_nt
        Sigma = x_s_ys - x_t_yt
        Sigma = Sigma.unsqueeze(0)
        # Sigma = self.Wrf(Sigma)
        return torch.mm(Sigma, Sigma.T)

    def only_target_train(self):
        self.source1_conv_layers2.requires_grad_(False)
        self.source2_conv_layers2.requires_grad_(False)
        self.source3_conv_layers2.requires_grad_(False)
        self.target_conv_layers2.requires_grad_(True)

        self.source1_classifier.requires_grad_(False)
        self.source2_classifier.requires_grad_(False)
        self.source3_classifier.requires_grad_(False)
        self.target_classifier.requires_grad_(False)

    def sample_norm(self, x_new):
        x_new = x_new.T
        x_new = x_new / (torch.linalg.norm(x_new, dim=0) + 1e-5)
        x_new = x_new.T
        return x_new



class visda_RFMMD_notca_federated_source_target_model(nn.Module):
    def __init__(self, feature_dim, n_features, sigma, kernel='rbf', class_num=31):
        """
        Source forward process:
            X -> conv1 -> conv2 -> classifier
        Input:
            feature_dim: faeture dimention of the output of feature extractor;
            m: final feature dimention of transfer feature, top m eigenvectors of Equation;
        """
        super(visda_RFMMD_notca_federated_source_target_model, self).__init__()
        # feature extractor
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv_layers1 = torch.nn.Sequential(*(list(resnet50.children())[0:7])).requires_grad_(False)
        self.source1_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.target_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        
        
        # random feature map part, with unified random weights
        self.rf_map = federated_rf_map(n_features=n_features, sigma=sigma, feature_dim=feature_dim, kernel=kernel, domain='source')
        
        # classifier
        self.source1_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(True)
        self.target_classifier = bottleneck(input_features_dim=feature_dim, output=class_num).requires_grad_(False)

    def forward(self, x_s1, x_t):
        """"""
        x_s1 = self.conv_layers1(x_s1)
        x_t = self.conv_layers1(x_t)

        x_s1 = self.source1_conv_layers2(x_s1)
        x_t = self.target_conv_layers2(x_t)

        x_s1 = torch.flatten(x_s1, start_dim=1)
        x_t = torch.flatten(x_t, start_dim=1)

        # each party runs independently
        x_s1 = self.sample_norm(x_s1)
        x_t = self.sample_norm(x_t)

        # RF-TCA message
        source1_Sigma_i, source1_Sigma_yi = self.rf_map(x_s1)
        target_Sigma_i, target_Sigma_yi = self.rf_map(x_t)

        # generate new transfer features
        source1_Sigma_i = self.sample_norm(source1_Sigma_i.T)
        target_Sigma_i = self.sample_norm(target_Sigma_i.T)
        ys1 = self.source1_classifier(x_s1)
        yt = self.target_classifier(x_t)

        # transfer message
        # z_s1 = torch.add(x_s1, source1_Sigma_i)
        # z_t = torch.add(x_t, target_Sigma_i)

        # return z_s1, z_t, ys1, yt
        return source1_Sigma_i, target_Sigma_i, ys1, yt

    # CLassifier Layer Aggregation
    def FedAvg_classifier(self, subset_list=None):
        models_num = 1
        models_list = [self.source1_classifier]        
        if subset_list != None:
            models_num = len(subset_list)
            models_list = [models_list[subset_i] for subset_i in subset_list]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + models_state_dict[i][key]
            fed_state_dict[key] = key_sum / models_num
        # update fed weights to fed model
        self.source1_classifier.load_state_dict(fed_state_dict)
        self.target_classifier.load_state_dict(fed_state_dict)  

    def mmd_loss(self, x_s, x_t):
        batch_ns = x_s.size(0)
        batch_nt = x_t.size(0)
        x_s_ys = torch.sum(x_s, dim=0) / batch_ns
        x_t_yt = torch.sum(x_t, dim=0) / batch_nt
        Sigma = x_s_ys - x_t_yt
        Sigma = Sigma.unsqueeze(0)
        return torch.mm(Sigma, Sigma.T)

    def only_target_train(self):
        # feature extractors
        self.source1_conv_layers2.requires_grad_(False)
        self.target_conv_layers2.requires_grad_(True)
        # classifiers
        self.source1_classifier.requires_grad_(False)
        self.target_classifier.requires_grad_(False)

    def sample_norm(self, x_new):
        x_new = x_new.T
        x_new = x_new / (torch.linalg.norm(x_new, dim=0) + 1e-5)
        x_new = x_new.T
        return x_new


class federated_rf_map(nn.Module):
    def __init__(self, n_features, sigma, kernel='rbf', feature_dim=2048, domain='source'):
        super(federated_rf_map, self).__init__()
        self.rf_map = RFF_perso(sigma, n_features, kernel=kernel)
        self.rf_map.fit(feature_dim)
        self.domain = domain

    def forward(self, X):
        Xdevice = X.device
        ns_i = X.size(dim=0)
        X = self.sample_norm(X)

        # compute Sigma_i on i-th iteration
        assert not torch.any(torch.isnan(X))
        Sigma_i = self.rf_map.transform(X).T # Sigma_i: (2*n_features, n)
        assert not torch.any(torch.isnan(Sigma_i))

        # init y_i vector
        if self.domain == 'source':
            yi = 1/ns_i * torch.ones(ns_i, 1)
        elif self.domain == 'target':
            yi = -1/ns_i * torch.ones(ns_i, 1)
        yi = yi.to(Xdevice)

        Sigmai_yi = torch.mm(Sigma_i, yi)
                
        return Sigma_i, Sigmai_yi

    def get_rf_map(self):
        return self.rf_map
    
    def update_rf_map(self, rf_map):
        self.rf_map = rf_map

    def sample_norm(self, x_new):
        x_new = x_new.T
        x_new = x_new / (torch.linalg.norm(x_new, dim=0) + 1e-5)
        x_new = x_new.T
        return x_new
