import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import copy

# Implementation for pFeMe clients

class UserFedBABU(User):
    def __init__(self, numeric_id, train_data, test_data, model, args):
        super().__init__(numeric_id, train_data, test_data, model[0], args)

        self.loss = nn.NLLLoss()
        self.K = args.K
        self.personal_learning_rate = args.personal_learning_rate
        other_parameters = [p for p in self.model.parameters() if not any(p is q for q in self.model.linear.parameters())]
        self.optimizer = torch.optim.Adam(other_parameters, lr=self.learning_rate)
        self.optimizer_p = torch.optim.Adam(self.model.linear.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs, only_train_personal=False):
        LOSS = 0

        self.model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update 
            self.model.train()
            X, y = self.get_next_train_batch()
            output = self.model(X)
            loss = self.loss(output, y)
            if (only_train_personal):
                self.optimizer_p.zero_grad()
                loss.backward()
                self.optimizer_p.step()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.clone_model_paramenter(self.model.parameters(), self.local_model)

        return LOSS