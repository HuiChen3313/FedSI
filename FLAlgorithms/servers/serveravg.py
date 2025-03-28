import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, model, times, args):
        super().__init__(model[0], times, args)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(0)
        data = read_data(args.dataset, args.datasize)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, args.dataset)
            user = UserAVG(id, train, test, model, args)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",args.numusers, " / " ,total_users)
        print("Finished creating FedAvg server.")

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

    def train(self, add_new_client = False):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
            self.aggregate_parameters()

            
        self.save_results()
        self.save_model()