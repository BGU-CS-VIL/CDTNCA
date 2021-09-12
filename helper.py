import torch
import numpy as np

def compute_mean_var_within_class(X, y):
    '''

    :param X: [samples, channels, timesteps]
    :param y: labels in numerical format [samples,1]
    :return: mean variance of all classes
    '''
    # measure variance within class
    class_names = np.unique(y)
    n_classes = len(class_names)
    class_var = np.zeros(n_classes)
    for idx, class_num in enumerate(class_names):
        class_idx = y == class_num
        class_var[idx] = np.var(X[class_idx], axis=0).mean(
            axis=1).mean()  # variance between signals along each time-step
    return np.mean(class_var)

def test_model(X, y, model):
    X_aligned = model(torch.from_numpy(X).float().to('cuda'))[0].detach().cpu().numpy()
    loss = compute_mean_var_within_class(X_aligned, y)
    return X_aligned, loss

def get_means(X, y):
    X_means_aligned = []
    for i in np.unique(y):
        X_within_class = X[np.where(y == i)]
        X_means_aligned.append(np.mean(X_within_class, axis=0))
    means = np.array(X_means_aligned)
    return means