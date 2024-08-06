import os
import logging
import argparse
import torch.optim as optim 
import torch
import random
import yaml
import numpy as np
from model import VITAL
from dataloader import load_data
from evaluation import inference
from utils import get_log, train, set_seed
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser()

    # Dataset info.
    parser.add_argument('--dataset_name', default='CUB', type=str, 
                        choices=['CUB', 'Scene-15', 'WIKI', 'NUS-WIDE', 'Deep Animal', 'Deep Caltech-101', 'MNIST-USPS', 'NoisyMNIST'],
                        help='The name of the dataset.')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in the dataset.')
    parser.add_argument('--aligned_rate', default=0.5, type=float, help='PVP setting aligned rate.')
    parser.add_argument('--train_time', default=5, type=int, help='Number of training iterations.')
    parser.add_argument('--dataset_path', default='./datasets/', type=str, help='Path to the dataset.')
    parser.add_argument('--log_path', default='./log/', type=str, help='Path to save logs.')
    parser.add_argument('--config_path', default='./config/', type=str, 
                        help='Path to the configuration files.')

    # Network architecture
    parser.add_argument('--recognition_model_dims', 
                        default=[[1024, 1024, 1024, 1024, 256], [300, 1024, 1024, 1024, 256]], type=list,
                        help='Recognition model layer dimensions in the form of a list of lists.')
    parser.add_argument('--generative_model_dims', 
                        default=[[128, 1024, 1024, 1024, 300], [128, 1024, 1024, 1024, 1024]], type=list,
                        help='Generative model layer dimensions in the form of a list of lists.')
    parser.add_argument('--activation', default='none', type=str, 
                        choices=['none', 'relu', 'tanh'], 
                        help='Activation function used in the adaption layer of the recognition model.')
    parser.add_argument('--use_dropout', default=True, type=bool, help='Whether to use dropout in the model.')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='Dropout rate if dropout is used.')
    parser.add_argument('--temperature', default=0.4, type=float, help='Temperature parameter for model training.')

    # Optimizer parameters
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size for training.')
    parser.add_argument('--vcl_epochs', default=100, type=int, help='Number of epochs for VCL training.')
    parser.add_argument('--vcl_dr_epochs', default=110, type=int, help='Number of epochs for total training.')
    parser.add_argument('--vcl_lr', default=2e-3, type=float, help='Learning rate for VCL training.')
    parser.add_argument('--vcl_dr_lr', default=1e-4, type=float, help='Learning rate for VCL-DR training.')

    # Other parameters
    parser.add_argument('--data_norm', default='min-max', type=str, 
                        choices=['min-max', 'standard', 'l2-norm'], 
                        help='Dataset preprocessing normalization method.')
    parser.add_argument('--fitting_type', default='loss', type=str, 
                        choices=['loss', 'sim'], 
                        help='Fitting type for GMM model input.')
    parser.add_argument('--mask_scheme', default='vital', type=str, choices=['gmm', 'vital'], help='Mask scheme for the model.')
    parser.add_argument('--init_alpha', default=0.1, type=float, help='Initial value of alpha parameter.')
    parser.add_argument('--fix_alpha', default=False, type=bool, help='Whether to fix alpha during training.')
    parser.add_argument('--feats_norm', default=True, type=bool, help='Whether to normalize features.')

    # GPU parameters
    parser.add_argument('--gpu', default='0', type=str, help='GPU device number to use.')

    return parser.parse_args()


def main():
    args = get_args_parser()

    # Load configurations from YAML file
    with open(os.path.join(args.config_path, args.dataset_name) + '.yaml') as f:
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())

    # Merge command line arguments and YAML configurations
    args = vars(args)
    args.update(configs)
    args = argparse.Namespace(**args)

    # Set the CUDA device
    torch.cuda.set_device(f"cuda:{args.gpu}")
    get_log(args)

    acc_list, nmi_list, ari_list = [], [], []
    seed_list = [random.randint(1, 10000) for _ in range(args.train_time)]

    # Begin Training
    for t in range(1, args.train_time + 1):
        set_seed(seed_list[t - 1])

        train_loader, test_loader, num_samples, num_classes, num_views, dims_list = load_data(args)

        # Validate dataset and model dimensions
        if dims_list[0] != args.recognition_model_dims[0][0] or dims_list[1] != args.recognition_model_dims[1][0]:
            raise ValueError("Model dimensions do not match dataset dimensions!")
        if num_classes != args.num_classes:
            raise ValueError("Wrong number of classes!")

        # Log dataset information on the first iteration
        if t == 1:
            logging.info(f'''Dataset info.: 
                dataset name: {args.dataset_name}, 
                number of views: {num_views}, 
                feature dimensions: {dims_list}, 
                number of samples: {num_samples}, 
                number of classes: {num_classes},
                aligned rate: {args.aligned_rate}''')

        # Initialize model
        model = VITAL(args).cuda()

        if t == 1:
            logging.info("Model info.:")
            logging.info(model)

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.vcl_lr)

        for epoch in range(args.vcl_dr_epochs + 1):
            if epoch == 0:
                with torch.no_grad():
                    train(train_loader, model, optimizer, epoch)
            else:
                train(train_loader, model, optimizer, epoch)

            # Adjust learning rate for VCL-DR stage
            if epoch == args.vcl_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.vcl_dr_lr

        # Begin Testing
        acc, nmi, ari = inference(model, test_loader, args)

        if t == 1:
            logging.info("Performance:")
        logging.info("Round %d: ACC: %.2f  NMI: %.2f  ARI: %.2f", 
                     t, round(acc, 2), round(nmi, 2), round(ari, 2))

        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)

    # Log the average results and standard deviation if multiple training iterations
    if args.train_time != 1:
        logging.info("   Mean: ACC: %.2f  NMI: %.2f  ARI: %.2f", 
                     round(np.mean(acc_list), 2), round(np.mean(nmi_list), 2), round(np.mean(ari_list), 2))
        logging.info("    std: ACC: %.2f   NMI: %.2f   ARI: %.2f", 
                     round(np.std(acc_list), 2), round(np.std(nmi_list), 2), round(np.std(ari_list), 2))
        

if __name__ == '__main__':
    main()
