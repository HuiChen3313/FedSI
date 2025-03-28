import copy
from torch.autograd import Variable
from FLAlgorithms.trainmodel.OModels import *
from FLAlgorithms.users.userbase import User


class UserpFedBayes(User):
    def __init__(self, numeric_id, train_data, test_data, model, args, output_dim=10):
        super().__init__(numeric_id, train_data, test_data, model[0], args, output_dim = output_dim)

        self.output_dim = output_dim
        self.batch_size = args.batch_size
        self.loss = nn.NLLLoss()
        self.zeta = args.zeta
        self.K = 5
        self.N_Batch = len(train_data) // args.batch_size
        self.personal_learning_rate = args.personal_learning_rate
        self.optimizer1 = torch.optim.Adam(self.personal_model.parameters(), lr=self.personal_learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        if (isinstance(self.model, pBNN)):
            N_Samples = 1
            Round = 5
            self.model.train()
            self.personal_model.train()

            for epoch in range(1, self.local_epochs + 1):

                X, Y = self.get_next_train_batch()
                batch_X = Variable(X.view(min(self.batch_size, Y.shape[0]), -1))
                batch_Y = Variable(Y.view(min(self.batch_size, Y.shape[0]), -1))
                label_one_hot = F.one_hot(batch_Y, num_classes=self.output_dim).squeeze(dim=1)

                for r in range(1, Round + 1):
                    ### personal model
                    epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
                    layer_params1 = self.personal_model.transform_gaussian_samples(
                        self.personal_model.mus, self.personal_model.rhos, epsilons)

                    personal_output = self.personal_model.net(batch_X, layer_params1)
                    # calculate the loss
                    personal_loss = self.personal_model.combined_loss_personal(
                        personal_output, label_one_hot, layer_params1,
                        self.personal_model.mus, self.personal_model.sigmas,
                        copy.deepcopy(self.model.mus),
                        [t.clone().detach() for t in self.model.sigmas], self.local_epochs)

                    self.optimizer1.zero_grad()
                    personal_loss.backward()
                    self.optimizer1.step()

                ### local model
                epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
                layer_params2 = self.model.transform_gaussian_samples(self.model.mus, self.model.rhos, epsilons)
                model_output = self.model.net(batch_X, layer_params2)

                model_loss = self.model.combined_loss_local(
                    [t.clone().detach() for t in layer_params1],
                    copy.deepcopy(self.personal_model.mus),
                    [t.clone().detach() for t in self.personal_model.sigmas],
                    self.model.mus, self.model.sigmas, self.local_epochs)

                self.optimizer2.zero_grad()
                model_loss.backward()
                self.optimizer2.step()
        else:
            zeta = self.zeta
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

                    self.optimizer1.zero_grad()
                    loss.backward()
                    self.optimizer1.step()

                # local model
                model_output = self.model(X)

                param1 = self.model.get_parameter()
                param2 = self.personal_model.get_parameter()
                model_loss = 0
                for idx in range(len(param1)):
                    for i in range(0, len(param1[idx]), 2):
                        mu_1, sigma_1 = param1[idx][i], F.softplus(param1[idx][i+1])
                        mu_2, sigma_2 = param2[idx][i].clone().detach(), F.softplus(param2[idx][i+1].clone().detach())
                        model_loss += 1.0 / self.local_epochs * zeta * kl_divergence(
                            Normal(mu_1, sigma_1), Normal(mu_2, sigma_2)
                            ).sum()

                # self.optimizer1.zero_grad()
                # model_loss.backward()
                # self.optimizer1.step()

                self.optimizer2.zero_grad()
                model_loss.backward()
                self.optimizer2.step()
            
            # self.update_parameters(self.personal_model.parameters())

        return LOSS
