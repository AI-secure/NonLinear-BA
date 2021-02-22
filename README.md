This is the code accompanying the AISTATS2021 paper: _Nonlinear Projection Based Gradient Estimation for Query EfficientBlackbox Attacks_.
This README explains the dependencies required as well as the detailed instructions to run the experiments either with the provided pretrained models, or from scratch.

## Environment Requirements ##
* GPU access (The pretrained models are on CUDA. Need extra modification to the code if no CUDA is available.)
* Python 3
* PyTorch 1.3.0
* Numpy 1.15.2
* Access to public image Datasets: ImageNet, CelebA, CIFAR10 and MNIST
* Face++ API access: please register an account at [Face++ website](https://www.faceplusplus.com/) to get the key and secret for access. Remember to update the `YOUR_KEY` and `YOUR_SECRET` attributes in `./src/foolbox/models/api_faceplusplus.py`.

### Datasets ###
Currently the code supports four datasets: ImageNet, CelebA, CIFAR10 and MNIST.
Please update `RAW_DATA_PATH` and `ROOT_DATA_PATH` in `./src/constants.py` to the directory names of the raw and preprocessed datasets in order to run the attacks.

## Attack with Pretrained Gradient Estimator ##
### Pretrained Models ###
We include some of the pretrained models that are necessary to reproduce a subset of the experimental results in the code repository, including the pretrained gradient estimators in `./src/gen_models` and the pretrained target models in `./src/class_models`.

Note that the pretrained models provided are only a subset of all the pretrained models since some of them are too large (e.g., the generator of a GAN model takes about 78 MB space each).

### Offline Tasks ###
To run the offline attack experiments on ImageNet, CelebA, CIFAR10 or MNIST with pretrained models, example command:
`python3 main_pytorch_multi.py --use_gpu --TASK CelebA --pgen AE9408 --N_img 50`
or `python3 main_pytorch_multi.py --use_gpu --TASK mnist --pgen VAE9408 --N_img 50 --mnist_img_size 224`
etc.

### Online API Experiments ###
We stored the randomly sampled pair IDs in `api_results/src_tgt_pairs.npy`
 so that if you want to reproduce the experimental results reported in the paper, run
`python3 main_api_attack.py --threshold 0.5 --use_gpu --pgen AE9408` and replace '`AE9408`' with other methods like '`VAE9408`' etc. for other experiments.
Or you can leverage the code in helper.py to run the attacks sequentially to avoid manually running the commands one by one.

If you instead want to try a different set of source-target image pairs, then comment out line 43 and line 46 in `./src/main_api_attack.py` and uncomment line 45. Then run the above commands.


## Train from scratch ##
If you don't want to use our pretrained estimators but instead want to train them from scratch yourself. Please follow the steps below before running the attacks.

For simplicity define command suffix `cmd_suf` as `--TASK imagenet` or `--TASK celeba` or `--TASK mnist --mnist_img_size 224` or `--TASK cifar10 --cifar10_img_size 224`. The choice depends on which image dataset is running.

### Preprocess Raw ImageNet Data ###
Run `preprocess_data.py` using command:
`python3 preprocess_data.py --TASK imagenet_preprocess`.

### Train Target and Reference Models ###
First we need both target and reference models.
* CelebA:
`python3 train_celeba_model.py --do_train --do_test`
* CIFAR10: `python3 train_cifar10_model.py --cifar10_img_size 224 --do_train`
* MNIST: `python3 train_mnist_model.py --mnist_img_size 224 --do_train`

### Gradient Data Preparation ###
`python3 gradient_generate.py` with cmd_suf.

`python3 preprocess_data.py` with cmd_suf.

### Train Estimators ###
`python3 train_pca_generator.py` with cmd_suf.

`python3 train_vae_generator.py` with cmd_suf.

`python3 train_ae_generator.py` with cmd_suf.

`python3 train_gan_generator.py` with cmd_suf.

## Citations ##
* If you find our code useful, please cite our paper: _Nonlinear Projection Based Gradient Estimation for Query EfficientBlackbox Attacks_.
* You may also find it useful to take a look at another work on this topic by us: _QEBA: Query-Efficient Boundary-Based Blackbox Attack_.


