import copy
import torch.nn as nn
from torch.autograd import Variable
from FLAlgorithms.trainmodel.OModels import *
from FLAlgorithms.users.userbase import User
from torch import distributions as dist
from torch.distributions.kl import kl_divergence
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from torch.nn import functional as F

class UserFedSOUL(User):
    def __init__(self, numeric_id, train_data, test_data, model, args, output_dim=10):
        super().__init__(numeric_id, train_data, test_data, model[0], args, output_dim = output_dim)

        self.output_dim = output_dim
        self.batch_size = args.batch_size
        self.N_Batch = len(train_data) // args.batch_size

        self.loss = nn.NLLLoss()

        self.K = 5
        self.personal_learning_rate = args.personal_learning_rate
        self.optimizer1 = torch.optim.Adam(self.personal_model.parameters(), lr=self.personal_learning_rate)
        self.optimizer1_p = torch.optim.Adam(self.personal_model.classifier.linear.parameters(), lr=self.personal_learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    
    def train(self, only_train_personal=False):
        LOSS = 0
        zeta = 0.001
        self.model.train()
        self.personal_model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update

            X, y = self.get_next_train_batch()

            for i in range(self.K):
                output = self.personal_model(X)

                loss = self.loss(output, y)

                param1 = self.model.get_parameter()
                param2 = self.personal_model.get_parameter()
                for idx in range(len(param1)):
                    for i in range(0, len(param1[idx]), 2):
                        mu_1, sigma_1 = param1[idx][i].clone().detach(), F.softplus(param1[idx][i+1].clone().detach())
                        mu_2, sigma_2 = param2[idx][i], F.softplus(param2[idx][i+1])
                        loss += 1.0 / self.local_epochs * zeta * kl_divergence(
                            Normal(mu_1, sigma_1), Normal(mu_2, sigma_2)
                            ).sum()
                if (only_train_personal):
                    self.optimizer1_p.zero_grad()
                    loss.backward()
                    self.optimizer1_p.step()
                else:
                    self.optimizer1.zero_grad()
                    loss.backward()
                    self.optimizer1.step()

            self.update_parameters(self.personal_model.parameters())

        return LOSS
