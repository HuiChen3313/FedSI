import torch
import os

from FLAlgorithms.users.userFedSIFac import UserFedSIFac
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

from torch.nn.utils import parameters_to_vector

# Implementation for TCYB Server

class FedSIFac(Server):
    def __init__(self, model, times, args):
        super().__init__(model[0], times, args)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(-1)
        print("mark_personalized_module", self.mark_personalized_module)

        data = read_data(args.dataset, args.datasize)
        self.device = args.device
        self.subnetwork_rate = args.subnetwork_rate

        if (args.only_one_local):
            self.total_users = 1
        else:
            self.total_users = len(data[0])
        for i in range(self.total_users):
            id, train , test = read_user_data(i, data, args.dataset)
            user = UserFedSIFac(id, train, test, model, args)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        # MNIST 
        # self.lamda = 1e-4
        # Cifar10
        self.lamda = 1e-4
        self.n_params = len(parameters_to_vector(self.model.parameters()).detach())
        self.global_sigma = torch.zeros(parameters_to_vector(self.model.parameters()).shape).to(args.device)
        self.global_sigma += 1 / self.lamda
        print("Number of users / total users:", args.numusers, " / " ,self.total_users)
        print("Finished creating FedSI server.")

    def send_grads(self, AddNewClient = False):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def send_parameters(self, personalized = []):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model, personalized)
            user.sigma = self.global_sigma

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        self.global_sigma = torch.zeros(parameters_to_vector(self.model.parameters()).shape).to(self.device)
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            self.global_sigma += user.sigma * user.train_samples / total_train
        print("before_global_sigma", self.global_sigma.sum()) 
        # MNIST
        zero_elements = (torch.abs(self.global_sigma) <= 1e-10)
        self.global_sigma[zero_elements] = 1 / self.lamda
        inf_elements = (torch.abs(self.global_sigma) <= 1e-3) * (torch.abs(self.global_sigma) > 1e-10)
        self.global_sigma[inf_elements] = 1e-3
        # Cifar10
        # zero_elements = (torch.abs(self.global_sigma) <= 1e-10)
        # self.global_sigma[zero_elements] = 1 / self.lamda
        # inf_elements = (torch.abs(self.global_sigma) <= 1e-1) * (torch.abs(self.global_sigma) > 1e-10)
        # self.global_sigma[inf_elements] = 1e-1

        print("after_global_sigma", self.global_sigma.mean())

    def train(self, AddNewClient = False):
        if (AddNewClient):
            self.users_copy = self.users
            self.users = self.users[1:]

        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters(personalized = self.mark_personalized_module)
            # Evaluate model each interation
            self.evaluate()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs, self.mark_personalized_module) #* user.train_samples
            self.evaluate_personalized_model(hasPMB=False)
            self.aggregate_parameters()

        if (AddNewClient):
            loss = []
            self.users = self.users_copy[0:1]
            self.mark_personalized_module[-1] = self.mark_personalized_module[-2] = 1
            for glob_iter in range(self.num_glob_iters):
                print("-------------Add New Client Round number: ",glob_iter, " -------------")
                #loss_ = 0
                self.send_parameters(personalized = self.mark_personalized_module)
                # Evaluate model each interation
                self.evaluate()
                self.selected_users = self.select_users(glob_iter, self.num_users)
                for user in self.selected_users:
                    user.train(self.local_epochs, self.mark_personalized_module, only_train_personal=True) #* user.train_samples
                self.evaluate_personalized_model(hasPMB=False)
                self.aggregate_parameters()

        self.save_results()
        self.save_model()