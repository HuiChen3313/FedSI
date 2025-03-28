import copy

def get_file_path(args, loadP = False, run_idx = 0):
    alg = args.dataset + "_" + args.datasize + "_" + args.algorithm
    if (loadP):
        alg += "_p"
    alg = alg + "_" + str(args.learning_rate) + "_" + str(args.beta) + "_" + str(args.lamda) + "_" + str(args.numusers) + "u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    if (args.algorithm == "pFedMe" or args.algorithm == "pFedMe_p"):
        alg = alg + "_" + str(args.K) + "_" + str(args.personal_learning_rate)
    alg = alg + "_" + str(run_idx)
    if (args.algorithm == "FedSI"):
        alg = alg + "_" + str(args.subnetwork_rate)
    if (args.only_one_local):
        alg = alg + "_only_one_local"
    if (args.add_new_client):
        alg = alg + "addclient" + str(args.add_new_client)

    return ["./results/"+'{}.h5'.format(alg, args.local_epochs),
            "./results/"+'{}_output.npy'.format(alg, args.local_epochs),
            "./results/"+'{}_y.npy'.format(alg, args.local_epochs)]

def change_avg(args_pre):
    args = copy.deepcopy(args_pre)
    if (args.algorithm == "FedSI" or args.algorithm == "FedSIFac"):
        # Old Version
        args.lamda = 15
        # args.lamda = 0.0001
        args.batch_size = 50
        if (args.dataset == "Cifar10"):
            args.learning_rate = 0.01
            args.subnetwork_rate = 0.07
        else:
            args.learning_rate = 0.1
            args.subnetwork_rate = 0.05
    if (args.algorithm == "FedAvg" or args.algorithm == "LocalOnly" or args.algorithm == "FedAvgFT"):
        if (args.dataset == "Cifar10"):
            args.learning_rate = 0.01
        else:
            args.learning_rate = 0.001
    if (args.algorithm == 'PerAvg'):
        args.beta = 0.1
    if (args.algorithm == 'pFedMe'):
        args.learning_rate=0.01 
        args.personal_learning_rate=0.01
    if (args.algorithm == 'pFedBayes'):
        args.batch_size=100 
    if (args.algorithm == 'BPFedPD'):
        args.batch_size=100 
    if (args.algorithm == 'FedSOUL'):
        args.beta = 1.0
    return args
