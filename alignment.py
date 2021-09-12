import torch
import matplotlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from model import AlignmentNet
from train import Trainer
from helper import compute_mean_var_within_class, test_model, get_means

parser = argparse.ArgumentParser(description='CDTNCA example')
parser.add_argument("--exp_name", type=str,
                    default="10_shapes",
                    help='experiment name')
parser.add_argument('--tess_size', type=int, default=[32],
                    help="CPA velocity field partition")
parser.add_argument('--zero_boundary', type=bool, default=True,
                    help="zero boundary constraint")
parser.add_argument('--circularity', type=bool, default=True,
                    help="circularity  constraint")
parser.add_argument('--epochs', type=int, default=100,
                    help="number of epochs")
parser.add_argument('--batch_size', type=int, default=32,
                    help="batch size")
parser.add_argument('--lr', type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--data_path", type=str,
                    default="datasets/align_num_classes_10_num_samples_100_tess_size_16_circularity_True_K_4.npy",
                    choices=["datasets/align_num_classes_10_num_samples_100_tess_size_16_circularity_True_K_4.npy",
                             "datasets/align_num_classes_10_num_samples_100_tess_size_16_circularity_True_K_8.npy",
                             "datasets/align_num_classes_10_num_samples_100_tess_size_16_circularity_True_K_16.npy",
                             "datasets/align_num_classes_10_num_samples_100_tess_size_16_circularity_True_sigma_0.1.npy",
                             "datasets/align_num_classes_10_num_samples_100_tess_size_16_circularity_True_sigma_0.5.npy",
                             "datasets/align_num_classes_10_num_samples_100_tess_size_16_circularity_True_sigma_1.npy"],
                    help='dataset')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
args = parser.parse_args()

# get GPU/CPU as device
if torch.cuda.device_count() > 0:
    device = torch.device('cuda')
    cpab_device = 'gpu'
else:
    device = torch.device('cpu')
    cpab_device = 'cpu'

args.device = device

# Set the seed of PRNG manually for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

dataset = np.load(args.data_path, allow_pickle=True)
X, y, base_shapes, _ = dataset

# k stands for the dimension of each larndmark and n stands for the number of landmarks
k, n = base_shapes.shape[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test,
                                                random_state=args.seed)

base_train_var = compute_mean_var_within_class(X_train, y_train)
base_test_var = compute_mean_var_within_class(X_test, y_test)

if __name__ == '__main__':
    model = AlignmentNet(n, k, args.tess_size, args.circularity, args.zero_boundary, cpab_device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# parameters:", pytorch_total_params)

    use_cpab = True
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=args.batch_size,
                            shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=args.batch_size,
                             shuffle=True)

    trainer = Trainer(args=args, model=model, train_loader=train_loader, val_loader=val_loader, optimizer=opt,
                      device=device)
    model = trainer.train()

    X_train_aligned, X_train_loss = test_model(X_train, y_train, model)
    X_test_aligned, X_test_loss = test_model(X_test, y_test, model)
    print(
        f'Base variance train:{base_train_var}, base variance test:{base_test_var}, aligned train variance:{X_train_loss}, aligned test variance:{X_test_loss}')

    # Plot results
    num_shapes = base_shapes.shape[0]
    train_means = get_means(X_train, y_train)
    train_means_aligned = get_means(X_train_aligned, y_train)
    test_means = get_means(X_test, y_test)
    test_means_aligned = get_means(X_test_aligned, y_test)

    cols = 3
    rows = 2

    for i in range(num_shapes):
        matplotlib.style.use('seaborn')
        fig, ax = plt.subplots(rows, cols, figsize=(18, 12))

        ax[0, 0].set_title("Original shape", fontsize=40)
        ax[0, 1].set_title("Train mean", fontsize=40)
        ax[0, 2].set_title("Test mean", fontsize=40)

        ax[1, 0].set_title("Original shape", fontsize=40)
        ax[1, 1].set_title("Aligned train mean", fontsize=40)
        ax[1, 2].set_title("Aligned test mean", fontsize=40)
        base_shape = base_shapes[i]
        train_shape = train_means[i]
        train_shape_alig = train_means_aligned[i]

        test_shape = test_means[i]
        test_shape_alig = test_means_aligned[i]

        ax[0, 0].plot(base_shape[0], base_shape[1], marker='o', markersize=10, color='red')
        ax[0, 1].plot(train_shape[0], train_shape[1], marker='o', markersize=10, color='red')
        ax[0, 2].plot(test_shape[0], test_shape[1], marker='o', markersize=10, color='red')

        ax[1, 0].plot(base_shape[0], base_shape[1], marker='o', markersize=10, color='red')
        ax[1, 1].plot(train_shape_alig[0], train_shape_alig[1], marker='o', markersize=10, color='red')
        ax[1, 2].plot(test_shape_alig[0], test_shape_alig[1], marker='o', markersize=10, color='red')
        plt.show()
