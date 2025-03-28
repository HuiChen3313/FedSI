#!/usr/bin/env python
from utils.main_utils import server_select, model_select, load_hypermater
from FLAlgorithms.trainmodel.OModels import *
from FLAlgorithms.trainmodel.FedSIModel import *
from utils.plot_utils import *
import torch
import time
import psutil

torch.manual_seed(0)

def main(args):
    # Get device status: Check GPU or CPU
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    # Initialize time and memory tracking
    total_training_time = 0
    max_memory_allocated = 0

    for i in range(args.times):
        print("---------------Running time:", i, "------------")

        # Start time measurement
        start_time = time.time()

        # Generate model
        model = model_select(args)
        server = server_select(model, i, args) 

        # Train the model
        server.train(args.add_new_client)

        # Measure training time
        training_time = time.time() - start_time
        total_training_time += training_time
        print(f"Training time for iteration {i}: {training_time:.2f} seconds")

        # Measure memory usage
        if args.device.type == 'cuda':
            memory_allocated = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)  # Convert to MB
            torch.cuda.reset_max_memory_allocated(args.device)  # Reset for next iteration
        else:
            memory_allocated = psutil.Process().memory_info().rss / (1024 ** 2)  # Convert to MB
        max_memory_allocated = max(max_memory_allocated, memory_allocated)
        print(f"Memory used for iteration {i}: {memory_allocated:.2f} MB")

        # Test the model
        if isinstance(model[0], pBNN):
            server.testpFedbayes()
        else:
            server.test()

    # Print average results
    print(f"Average training time of each training: {total_training_time / args.times:.2f} seconds")
    print(f"Maximum memory used: {max_memory_allocated:.2f} MB")

if __name__ == "__main__":
    args = load_hypermater()
    main(args)
