import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from torch.autograd import Variable

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, id, train_data, test_data, model, args, output_dim = 10):

        self.device = args.device
        self.output_dim = output_dim
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader =  DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))

        self.personal_model = copy.deepcopy(model)
        self.local_global_model = copy.deepcopy(model)
        # self.local_model = copy.deepcopy(model)

        self.N_Batch = len(train_data) // args.batch_size
        self.data_size = len(train_data)
        data_dim = 784
        hidden_dim = 100
        total = (data_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (
                hidden_dim + 1) * 1
        L = 3
        a = np.log(total) + 0.1 * ((L + 1) * np.log(hidden_dim) + np.log(np.sqrt(self.data_size) * data_dim))
        lm = 1 / np.exp(a)
        self.phi_prior = torch.tensor(lm).to(self.device)
        self.temp = 0.5
    
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
    
    def set_parameters_pFed(self, model):
        for user_layer, server_layer in zip(self.model.layers, model.layers):
            for personal_param, local_param, new_param in zip(user_layer.personal.parameters(),
                                                              user_layer.local.parameters(),
                                                              server_layer.local.parameters()):
                personal_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        # print("-----------")
        for param , new_param in zip(self.model.parameters(), new_params):
            # print("param", param.shape)
            # print("new_param", new_param.shape)
            param.data = new_param.data.clone()
        # print("-----------")

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        output_list = torch.tensor([]).to(self.device)
        y_list = torch.tensor([]).to(self.device)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            output_list = torch.cat((output_list, output), dim = 0)
            y_list = torch.cat((y_list, y), dim = 0)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0], output_list, y_list 

    def testBayes(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            test_size = x.size()[0]
            test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
            test_Y = Variable(y.view(test_size, -1)).to(self.device)
            # output = self.model.forward(test_X, mode='MAP').data.argmax(axis=1)

            epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
            # compute softplus for variance
            sigmas = self.model.transform_rhos(self.model.rhos)
            # obtain a sample from q(w|theta) by transforming the epsilons
            layer_params = self.model.transform_gaussian_samples(self.model.mus, sigmas, epsilons)
            # forward-propagate the batch
            output = self.model.net(test_X, layer_params)
            output = F.softmax(output, dim=1).data.argmax(axis=1)
            y = test_Y.data.view(test_size)
            test_acc += (torch.sum(output == y)).item()

            # test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]
    
    def testpFedbayes(self):
        self.model.eval()
        test_acc_personal = 0
        test_acc_global = 0
        for x, y in self.testloaderfull:
            test_size = x.size()[0]
            test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
            test_Y = Variable(y.view(test_size, -1)).to(self.device)

            # personal model
            epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
            # obtain a sample from q(w|theta) by transforming the epsilons
            layer_params1 = self.personal_model.transform_gaussian_samples(self.personal_model.mus,
                                                                           self.personal_model.rhos, epsilons)
            # forward-propagate the batch
            output = self.personal_model.net(test_X, layer_params1)
            output = F.softmax(output, dim=1).data.argmax(axis=1)
            y = test_Y.data.view(test_size)
            test_acc_personal += (torch.sum(output == y)).item()

            # global model
            epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
            # obtain a sample from q(w|theta) by transforming the epsilons
            layer_params1 = self.model.transform_gaussian_samples(self.model.mus, self.model.rhos, epsilons)
            # forward-propagate the batch
            output = self.model.net(test_X, layer_params1)
            output = F.softmax(output, dim=1).data.argmax(axis=1)
            # y = test_Y.data.view(test_size)
            test_acc_global += (torch.sum(output == y)).item()

        return test_acc_personal, test_acc_global, y.shape[0]

    def testBayesV2(self):
        self.model.eval()
        test_acc_personal = 0
        test_acc_global = 0
        output_list = torch.tensor([]).to(self.device)
        y_list = torch.tensor([]).to(self.device)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)

            output = self.personal_model(x)
            output_list = torch.cat((output_list, output), dim = 0)
            y_list = torch.cat((y_list, y), dim = 0)
            output = F.softmax(output, dim=1).data.argmax(axis=1)
            test_acc_personal += (torch.sum(output == y)).item()

            # global model
            output = self.model(x)
            output = F.softmax(output, dim=1).data.argmax(axis=1)
            # y = test_Y.data.view(test_size)
            test_acc_global += (torch.sum(output == y)).item()

        return test_acc_personal, test_acc_global, y.shape[0], output_list, y_list 

    def testSparseBayes(self):
        # self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            test_size = x.size()[0]
            test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
            test_Y = Variable(y.view(test_size, -1)).to(self.device)
            output = self.model.forward(test_X, mode='MAP').data.argmax(axis=1)
            # loss, pred = self.model.sample_elbo(test_X, test_Y, 30, self.temp, self.phi_prior, self.N_Batch)
            # pred = pred.mean(dim=0)
            # output = pred.data.argmax(axis=1)
            y = test_Y.data.view(test_size)

            test_acc += (torch.sum(output == y)).item()

            # test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]

    def testpFedSbayes(self):
        # self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            test_size = x.size()[0]
            test_X = Variable(x.view(test_size, -1).type(torch.FloatTensor)).to(self.device)
            test_Y = Variable(y.view(test_size, -1)).to(self.device)
            # output = self.model.forward(test_X, mode='MAP').data.argmax(axis=1)
            loss, pred = self.model.sample_elbo(test_X, test_Y, 30, self.temp, self.phi_prior, self.N_Batch)
            pred = pred.mean(dim=0)
            output = pred.data.argmax(axis=1)
            y = test_Y.data.view(test_size)

            test_acc += (torch.sum(output == y)).item()

            # test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            # train_acc += (torch.sum(output.max(1)[1] == y)).item()
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        return train_acc, loss , self.train_samples

    def train_error_and_loss_bayes(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            size = x.size()[0]
            train_X = Variable(x.view(size, -1).type(torch.FloatTensor)).to(self.device)
            train_Y = Variable(y.view(size, -1)).to(self.device)

            label_one_hot = F.one_hot(train_Y, num_classes=self.output_dim).squeeze(dim=1)
            epsilons = self.model.sample_epsilons(self.model.layer_param_shapes)
            # compute softplus for variance
            sigmas = self.model.transform_rhos(self.model.rhos)
            # obtain a sample from q(w|theta) by transforming the epsilons
            layer_params = self.model.transform_gaussian_samples(self.model.mus, sigmas, epsilons)
            # forward-propagate the batch
            output = self.model.net(train_X, layer_params)
            # calculate the loss
            loss = self.model.combined_loss(output, label_one_hot, layer_params, self.model.mus, sigmas,
                                            self.local_epochs)

            output = F.softmax(output, dim=1).data.argmax(axis=1)
            y = train_Y.data.view(size)
            train_acc += (torch.sum(output == y)).item()
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        return train_acc, loss, self.train_samples

    def train_error_and_loss_cifar10(self):
        self.model.eval()
        correct_items = 0
        total_loss = 0
        total_samples = 0
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            # calculate the loss
            output = self.model(x)
            test = torch.sum(output.max(1)[1] == y)
            loss = self.loss(output, y)

            output = F.softmax(output, dim=1).data.argmax(axis=1)
            correct_items += (torch.sum(output == y)).item()

            total_loss += loss.item()
            total_samples += len(x)

        return correct_items, total_loss, total_samples
    
    def train_error_and_loss_pFedbayes(self):
        self.model.eval()
        correct_items = 0
        total_loss = 0
        total_samples = 0
        for x, y in self.trainloader:
            size = x.size()[0]
            train_X = Variable(x.view(size, -1).type(torch.FloatTensor)).to(self.device)
            train_Y = Variable(y.view(size, -1)).to(self.device)
            # print("output_dim", self.output_dim)
            label_one_hot = F.one_hot(train_Y, num_classes=self.output_dim).squeeze(dim=1)
            ### personal model
            epsilons = self.personal_model.sample_epsilons(self.model.layer_param_shapes)
            # obtain a sample from q(w|theta) by transforming the epsilons
            layer_params1 = self.personal_model.transform_gaussian_samples(self.personal_model.mus,
                                                                           self.personal_model.rhos, epsilons)
            # forward-propagate the batch
            output = self.personal_model.net(train_X, layer_params1)
            # calculate the loss
            loss = self.personal_model.combined_loss_personal(output, label_one_hot, layer_params1,
                                                              self.personal_model.mus, self.personal_model.sigmas,
                                                              self.model.mus, self.model.sigmas, self.local_epochs)
            output = F.softmax(output, dim=1).data.argmax(axis=1)
            y = train_Y.data.view(size)
            correct_items += (torch.sum(output == y)).item()
            total_loss += loss.item()
            total_samples += len(x)
            # output = self.model.forward(train_X, mode='MAP').data.argmax(axis=1)
            # y = train_Y.data.view(size)
            # train_acc += (torch.sum(output == y)).item()
            #
            # loss += self.loss.loss_fn(output, y, 1.0)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)

        return correct_items, total_loss, total_samples

    def train_error_and_loss_sparsebayes(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            size = x.size()[0]
            train_X = Variable(x.view(size, -1).type(torch.FloatTensor)).to(self.device)
            train_Y = Variable(y.view(size, -1)).to(self.device)
            output = self.model.forward(train_X, mode='MAP').data.argmax(axis=1)
            # loss_temp, pred = self.model.sample_elbo(train_X, train_Y, 30, self.temp, self.phi_prior, self.N_Batch)
            # pred = pred.mean(dim=0)
            # output = pred.data.argmax(axis=1)
            y = train_Y.data.view(size)
            train_acc += (torch.sum(output == y)).item()

            loss += self.loss.loss_fn(output, y, 1.0)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        return train_acc, loss, self.train_samples

    def train_error_and_loss_pFedSbayes(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            size = x.size()[0]
            train_X = Variable(x.view(size, -1).type(torch.FloatTensor)).to(self.device)
            train_Y = Variable(y.view(size, -1)).to(self.device)
            loss_temp, pred = self.model.sample_elbo(train_X, train_Y, 30, self.temp, self.phi_prior, self.N_Batch)
            pred = pred.mean(dim=0)
            output = pred.data.argmax(axis=1)
            y = train_Y.data.view(size)
            train_acc += (torch.sum(output == y)).item()

            loss += loss_temp
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        return train_acc, loss, self.train_samples

    def test_persionalized_model(self, hasPMB = False):
        self.model.eval()
        test_acc = 0
        output_list = torch.tensor([]).to(self.device)
        y_list = torch.tensor([]).to(self.device)
        if (hasPMB):
            self.update_parameters(self.persionalized_model_bar)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            output_list = torch.cat((output_list, output), dim = 0)
            y_list = torch.cat((y_list, y), dim = 0)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        if (hasPMB):
            self.update_parameters(self.local_model)
        return test_acc, y.shape[0], output_list, y_list 

    def train_error_and_loss_persionalized_model(self, hasPMB = False):
        self.model.eval()
        train_acc = 0
        loss = 0
        if (hasPMB):
            self.update_parameters(self.persionalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        if (hasPMB):
            self.update_parameters(self.local_model)
        return train_acc, loss , self.train_samples
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))