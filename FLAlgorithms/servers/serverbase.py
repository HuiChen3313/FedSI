import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy
from utils.loadresult_utils import get_file_path

class Server:
    def __init__(self, model, times, args):
        # Set up the main attributes
        self.max_acc = 0
        # self.addclient_iters = args.add_new_client
        # self.only_one_local = args.only_one_local
        self.output_list = []
        self.y_list = []
        self.device = args.device
        self.dataset = args.dataset
        self.datasize = args.datasize
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = args.numusers
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        # self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,                  self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_per_acc, self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], [], []
        self.times = times
        self.save_results_path = get_file_path(args, False, times)[0]
        [self.save_results_path_p, self.save_results_path_output, self.save_results_path_y] = get_file_path(args, True, times)
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self, personalized = []):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model, personalized)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        print("store persionalized value!")

        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File(self.save_results_path, 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()

        if (len(self.rs_glob_acc_per) != 0) :
            with h5py.File(self.save_results_path_p, 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                if (len(self.rs_train_acc_per)):
                    hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                if (len(self.rs_train_loss_per)):
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()
        self.output_list = torch.tensor(self.output_list)
        self.y_list = torch.tensor(self.y_list)
        np.save(self.save_results_path_output, self.output_list.cpu().detach().numpy())
        np.save(self.save_results_path_y, self.y_list.cpu().detach().numpy())

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        output_list = torch.tensor([]).to(self.device)
        y_list = torch.tensor([]).to(self.device)
        for c in self.users:
            ct, ns, output, y = c.test()
            output_list = torch.cat((output_list, output), dim=0)
            y_list = torch.cat((y_list, y), dim=0)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct, output_list, y_list

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self, hasPMB=False):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        output_list = torch.tensor([]).to(self.device)
        y_list = torch.tensor([]).to(self.device)
        for c in self.users:
            ct, ns, output, y = c.test_persionalized_model(hasPMB)
            output_list = torch.cat((output_list, output), dim=0)
            y_list = torch.cat((y_list, y), dim=0)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, output_list, y_list

    def train_error_and_loss_persionalized_model(self, hasPMB=False):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model(hasPMB=hasPMB) 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()  
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        if (self.max_acc < glob_acc):
            self.max_acc = glob_acc
            self.output_list = stats[-2]
            self.y_list = stats[-1]

        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)

    def evaluate_personalized_model(self, hasPMB=False):
        stats = self.test_persionalized_model(hasPMB=hasPMB)  
        stats_train = self.train_error_and_loss_persionalized_model(hasPMB=hasPMB)
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        if (self.max_acc < glob_acc):
            self.max_acc = glob_acc
            self.output_list = stats[-2]
            self.y_list = stats[-1]
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        if (self.max_acc < glob_acc):
            self.max_acc = glob_acc
            self.output_list = stats[-2]
            self.y_list = stats[-1]
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)

    def testBayes(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.testBayes()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def testpFedbayes(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct_p = []
        tot_correct_g = []
        losses = []
        for c in self.users:
            ct_p, ct_g, ns = c.testpFedbayes()
            tot_correct_p.append(ct_p * 1.0)
            tot_correct_g.append(ct_g * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct_p, tot_correct_g
    
    def testBayesV2(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct_p = []
        tot_correct_g = []
        losses = []
        output_list = torch.tensor([]).to(self.device)
        y_list = torch.tensor([]).to(self.device)
        for c in self.users:
            ct_p, ct_g, ns, output, y = c.testBayesV2()
            output_list = torch.cat((output_list, output), dim=0)
            y_list = torch.cat((y_list, y), dim=0)
            tot_correct_p.append(ct_p * 1.0)
            tot_correct_g.append(ct_g * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct_p, tot_correct_g, output_list, y_list

    def testSparseBayes(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.testSparseBayes()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def testpFedSbayes(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.testSparseBayes()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_bayes(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_bayes()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_cifar10(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_cifar10()
            tot_correct.append(ct)
            num_samples.append(ns)
            losses.append(cl)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_pFedbayes(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_pFedbayes()
            tot_correct.append(ct)
            num_samples.append(ns)
            losses.append(cl)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_sparsebayes(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_sparsebayes()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_pFedSbayes(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_sparsebayes()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses
