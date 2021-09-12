#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:51:31 2019

@author: nsde
"""

#%%
from cpabTests.libcpab20.libcpab import Cpab
from cpabTests.libcpab20.libcpab import CpabAligner
from cpabTests.libcpab20.libcpab.core.utility import get_dir

import numpy as np
import matplotlib.pyplot as plt
import argparse

#%%
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='tensorflow',
                        choices=['numpy', 'tensorflow', 'pytorch'],
                        help='backend to run demo with')
    parser.add_argument('--device', default='gpu',
                        choices=['cpu', 'gpu'],
                        help='device to run demo on')
    parser.add_argument('--alignment_type', default='sampling',
                        choices=['sampling', 'gradient'],
                        help='how to align samples')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='number of iteration in alignment algorithm')
    return parser.parse_args()


#%%
if __name__ == "__main__":
    # Input arguments
    args = argparser()


    # Load some data
    data = plt.imread(get_dir(__file__) + '/../data/cat.jpg') / 255
    data = np.expand_dims(data, 0)  # create batch effect

    # Create transformer class
    T = Cpab([1, 1], backend=args.backend, device=args.device, zero_boundary=True,
             volume_perservation=False, override=False)

    # Sample random transformation
    theta = 0.5*T.sample_transformation(1)

    # Convert data to the backend format
    data = T.backend.to(data, device=args.device)

    # Pytorch have other data format than tensorflow and numpy, color information
    # is the second dim. We need to correct this before and after
    data = data.permute(0, 3, 1, 2) if args.backend == 'pytorch' else data

    # Transform the images
    transformed_data = T.transform_data(data, theta, outsize=(350, 350))

    # Now lets see if we can esimate the transformation we just used, by
    # iteratively trying to transform the data
    A = CpabAligner(T)
    
    # Do by sampling, work for all backends
    if args.alignment_type == 'sampling':
        theta_est = A.alignment_by_sampling(
            data, transformed_data, maxiter=args.maxiter)
    # Or do it by gradient descend (only tensorflow and pytorch)
    else:
        theta_est = A.alignment_by_gradient(
            data, transformed_data, maxiter=args.maxiter)

    # Lets see what we converged to
    trans_est = T.transform_data(data, theta_est, outsize=(350, 350))
    
    # Revert pytorch format
    if args.backend == 'pytorch':
        data = data.permute(0, 2, 3, 1)
        transformed_data = transformed_data.permute(0, 2, 3, 1)
        trans_est = trans_est.permute(0, 2, 3, 1)

    # Show the results
    data = T.backend.tonumpy(data)
    transformed_data = T.backend.tonumpy(transformed_data)
    trans_est = T.backend.to(trans_est)
    
    print('Theta:    ', T.backend.tonumpy(theta))
    print('Theta est:', T.backend.tonumpy(theta_est))
    
    plt.subplots(1, 3)
    plt.subplot(1, 3, 1)
    plt.imshow(data[0])
    plt.axis('off')
    plt.title('Source')
    plt.subplot(1, 3, 2)
    plt.imshow(transformed_data[0])
    plt.axis('off')
    plt.title('Target')
    plt.subplot(1, 3, 3)
    plt.imshow(trans_est[0])
    plt.axis('off')
    plt.title('Estimate')
    plt.show()
    
    
