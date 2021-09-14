# CDTNCA
Repository for our <b>ICIP 2021</b> paper, titled [Cyclic Diffeomorphic Transformer Nets for Contour Alignment](https://ieeexplore.ieee.org/abstract/document/9506570) co-authored by: Ilya Kaufman, Ron Shapira Weber and Oren Freifeld.
<img src="/figures/intro.png" alt="CDTNCA alignmnet.">
## Model Architecture
<img src="/figures/model.png" alt="CDTNCA architecture.">

### libcpab
licpab [2] is a python package supporting the CPAB transformations [1] in Numpy, Tensorflow and Pytorch.
Our code uses a modified version of libcpab which allows for circular warps.
<img src="/figures/warps.png" alt="Warp with different constraints.">
## Author of this software 
Ilya Kafuman (email: ilyakau@post.bgu.ac.il)

## Requirements
- Standard Python(>=3.6) packages: numpy, matplotlib, tqdm
- PyTorch >= 1.4
- For Nvidia GPU iimplementation: CUDA==11.0 + appropriate cuDNN as well. You can follow the instructions [here](https://pytorch.org/get-started/locally/).

### Usage
Alignment of a subset of shapes generated from the 2D Shape Structure dataset [3].
For the entire archive, please visit:
[http://2dshapesstructure.github.io/](http://2dshapesstructure.github.io/)

```
python alignment.py [args]
```
* tess_size: list, with the number of cells in each dimension
* zero_boundary: bool, determines if the velocity at the boundary is zero 
* circularity: bool, allows for circular warps

## References
```
[1] @article{freifeld2017transformations,
  title={Transformations Based on Continuous Piecewise-Affine Velocity Fields},
  author={Freifeld, Oren and Hauberg, Soren and Batmanghelich, Kayhan and Fisher, John W},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017},
  publisher={IEEE}
}

[2] @misc{detlefsen2018,
  author = {Detlefsen, Nicki S.},
  title = {libcpab},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SkafteNicki/libcpab}},
}
[3] @article{Carlier:CG:20162:2DShapes,
  title={The 2d shape structure dataset: A user annotated open access database},
  author={Carlier, Axel and Leonard, Kathryn and Hahmann, Stefanie and Morin, Geraldine and Collins, Misha},
  journal={Computers \& Graphics},
  volume={58},
  pages={23--30},
  year={2016},
  publisher={Elsevier}
}
```
## License
This software is released under the MIT License (included with the software). Note, however, that if you are using this code (and/or the results of running it) to support any form of publication (e.g., a book, a journal paper, a conference paper, a patent application, etc.) then we request you will cite our paper:
```
@INPROCEEDINGS{kaufman2021contouralign,
  author={Kaufman, Ilya and Weber, Ron Shapira and Freifeld, Oren},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={Cyclic Diffeomorphic Transformer Nets For Contour Alignment}, 
  year={2021},
  pages={349-353}
}

