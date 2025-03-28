import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User

import pytest
from itertools import product

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import wide_resnet50_2

from laplace import Laplace, SubnetLaplace, FullSubnetLaplace, DiagSubnetLaplace
from laplace.baselaplace import DiagLaplace
from laplace.utils import (SubnetMask, RandomSubnetMask, LargestMagnitudeSubnetMask,
                           LargestVarianceDiagLaplaceSubnetMask, LargestVarianceSWAGSubnetMask,
                           ParamNameSubnetMask, ModuleNameSubnetMask, LastLayerSubnetMask)

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

# Implementation for FedAvg clients

class UserFedSI(User):
    def __init__(self, numeric_id, train_data, test_data, model, args):
        super().__init__(numeric_id, train_data, test_data, model[0], args)
        self.device = args.device
        self.loss = nn.CrossEntropyLoss()

        other_parameters = [p for p in self.model.parameters() if not any(p is q for q in self.model.linear.parameters())]
        if (args.optimizer == "Adam"):  # Cifar10
            print("Optimizer:", args.optimizer)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.optimizer_p = torch.optim.Adam(self.model.linear.parameters(), lr=self.learning_rate)
        else:                           # Mnist
            print("Optimizer:", args.optimizer)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            self.optimizer_p = torch.optim.SGD(self.model.linear.parameters(), lr=self.learning_rate)

        self.n_params = len(parameters_to_vector(self.model.parameters()).detach())
        self.sigma = torch.zeros(parameters_to_vector(self.model.parameters()).shape)
        self.subnetwork_rate = args.subnetwork_rate
        # Cifar10
        # self.prior_weight = 1
        # MNIST
        self.prior_weight = 1e-4
        print("UserFedSI: subnetwork_rate", self.subnetwork_rate)
        print("n_params", self.n_params)
        print("subnetwork_n_params", int(self.n_params * self.subnetwork_rate))

    def set_parameters(self, model, personalized = []):
        num_param = len(self.local_model)
        idx = 0
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            # if (len(personalized) != 0):
                # print(old_param.data.shape, personalized[idx])
            if (len(personalized) != 0 and personalized[idx] == 1):
                continue
            old_param.data, local_param.data, idx = new_param.data.clone(), new_param.data.clone(), idx + 1
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs, mark_personalized_module, only_train_personal=False):
        LOSS = 0
        self.model.train()

        global_params = []
        mark_list = []
        mark_personalized_module = 1 - torch.tensor(mark_personalized_module)
        for idx, param in enumerate(self.model.parameters()):
            global_params.append(param.data.clone())
            if (mark_personalized_module[idx] == 1):
                mark_list.append(torch.ones(global_params[-1].shape).to(self.device))
            else:
                mark_list.append(torch.zeros(global_params[-1].shape).to(self.device))
        global_params = parameters_to_vector(global_params)
        mark_list = parameters_to_vector(mark_list)

        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            X, y = self.get_next_train_batch()
            output = self.model(X)

            if (only_train_personal):
                loss = self.loss(output, y)

                self.optimizer_p.zero_grad()
                loss.backward()
                self.optimizer_p.step()
            else:
                current_parameters = parameters_to_vector(self.model.parameters())
                prior_loss = torch.sum(0.5 / (self.sigma) * mark_list * torch.pow(current_parameters - global_params, 2))
                # loss = self.loss(output, y) 
                loss = self.loss(output, y) + prior_loss * self.prior_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.clone_model_paramenter(self.model.parameters(), self.local_model)

        diag_la = Laplace(self.model, 'classification', 
                          subset_of_weights='all', 
                          hessian_structure='diag')
        self.subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(self.model, 
                                                                    n_params_subnet=int(self.n_params * self.subnetwork_rate), 
                                                                    diag_laplace_model = diag_la)
        subnetwork_indices = self.subnetwork_mask.select(self.trainloader)
        # print("subnetwork_indices", subnetwork_indices)   
        if (only_train_personal != True):
            la = Laplace(self.model, 'classification',
                        subset_of_weights='subnetwork',
                        hessian_structure='full',
                        subnetwork_indices=subnetwork_indices) 
            sigma = la.fit(self.trainloader)
            # print("sigma", sigma.mean())
            diag_sigma = torch.diag(sigma)
            # print("diag_sigma", diag_sigma.mean())
            self.sigma = torch.zeros(parameters_to_vector(self.model.parameters()).shape).to(self.device)
            self.sigma[subnetwork_indices] = diag_sigma

        return LOSS