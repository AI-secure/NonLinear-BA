import numpy as np
import foolbox
import torch

from models import ResizeGenerator
from models import DCTGenerator
from models import PCAGenerator
from models import AEGenerator
from models import GANGenerator
from models import VAEGenerator

from models import Cifar10AEGenerator

from models import OldAEGenerator

from models import ExpCosGenerator

from models import MNISTAEGenerator
from models import MNIST224VAEGenerator
# from models import MNISTResizeAEGenerator

from models import Cifar10DCGenerator
from models import MNISTDCGenerator

import constants
import model_settings


def load_pgen(task, pgen_type, args):
    if task == 'imagenet' or task == 'celeba' or task == 'celebaid' or task == 'celeba2':
        if pgen_type == 'naive':
            p_gen = None

        elif pgen_type == 'resize9408':
            p_gen = ResizeGenerator(factor=4.0)

        elif pgen_type == 'DCT2352':
            p_gen = DCTGenerator(factor=8.0)
        elif pgen_type == 'DCT4107':
            p_gen = DCTGenerator(factor=6.0)
        elif pgen_type == 'DCT9408':
            p_gen = DCTGenerator(factor=4.0)
        elif pgen_type == 'DCT16428':
            p_gen = DCTGenerator(factor=3.0)

        elif pgen_type == 'PCA9408':
            p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
            # p_gen.load('./gen_models/pca_gen_%s_%d.npy' % (task, 9408))
            p_gen.load('./gen_models/pca_gen_%s_%d.npy' % ('imagenet', 9408))

        elif pgen_type == 'AE128':
            p_gen = AEGenerator(n_channels=3, gpu=args.use_gpu, N_Z=128)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_ae_%d_generator.model' %(task, 128)))
        elif pgen_type == 'AE9408':
            p_gen = AEGenerator(n_channels=3, gpu=args.use_gpu, N_Z=9408)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_ae_%d_generator.model' %(task, 9408)))

        elif pgen_type == 'VAE9408':
            p_gen = VAEGenerator(n_channels=3, gpu=args.use_gpu)
            if args.TASK == 'celeba' and args.smooth:
                p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_smooth%s_generator.model' % (task, 9408, args.smooth_suffix)))
            else:
                p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_generator.model' %(task, 9408)))

        elif pgen_type == 'GAN128':
            p_gen = GANGenerator(n_z=128, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_gan_%d_generator.model' % (task, 128)))
        elif pgen_type == 'GAN9408':
            p_gen = GANGenerator(n_z=9408, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_gan_%d_generator.model' % (task, 9408)))

        elif pgen_type == 'oldAE9408':
            p_gen = OldAEGenerator(n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_old_ae_%d_generator.model' %(task, 9408)))

        elif pgen_type == 'expcos9408':
            p_gen = ExpCosGenerator(n_channels=3, gpu=args.use_gpu, N_Z=9408, lmbd=args.lmbd)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_expcos_%d_generator.model' % (task, 9408)))

    elif task == 'cifar10':
        model_file_name = model_settings.get_model_file_name("cifar10", args)
        n_channels = 3
        if args.cifar10_img_size==32:
            if pgen_type == 'naive':
                p_gen = None

            elif pgen_type == 'resize192':
                p_gen = ResizeGenerator(factor=4.0)

            elif pgen_type == 'DCT192':
                p_gen = DCTGenerator(factor=4.0)

            elif pgen_type == 'AE192':
                p_gen = Cifar10AEGenerator(gpu=args.use_gpu)
                p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_ae_%d_generator.model' % (task, 192)))

        elif args.cifar10_img_size == 224:
            if pgen_type == 'naive':
                p_gen = None

            elif pgen_type == 'resize9408':
                p_gen = ResizeGenerator(factor=4.0)

            elif pgen_type == 'DCT9408':
                p_gen = DCTGenerator(factor=4.0)

            elif pgen_type == 'PCA9408':
                N_b = 9408
                approx = True
                p_gen = PCAGenerator(N_b=N_b, approx=approx, basis_only=True)
                # p_gen.load('./gen_models/pca_gen_%s_%d.npy' % (model_file_name, 9408))
                p_gen.load('./gen_models/pca_gen_%s_%d.npy' % ('imagenet', 9408))

            elif pgen_type.startswith('AE'):
                N_Z = int(pgen_type[2:])
                p_gen = AEGenerator(n_channels=3, preprocess=None, gpu=args.use_gpu, N_Z=N_Z)
                p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_ae_%d_generator.model' % (model_file_name, N_Z)))

            elif pgen_type.startswith('GAN'):
                N_Z = int(pgen_type[3:])
                p_gen = GANGenerator(n_z=N_Z, n_channels=n_channels, gpu=args.use_gpu)
                p_gen.load_state_dict(torch.load(
                    './gen_models/%s_gradient_gan_%d_generator.model' % (model_file_name, N_Z)))

            elif pgen_type=='DCGAN':
                p_gen = Cifar10DCGenerator(ngpu=1).eval()
                p_gen.load_state_dict(torch.load('./models/weights/cifar10_netG_epoch_199.pth'), strict=False)

            elif pgen_type.startswith('DCGAN_finetune'):
                n_epoch = int(pgen_type[-1])
                p_gen = Cifar10DCGenerator(ngpu=1).eval()
                # p_gen.load_state_dict(torch.load('./models/weights/cifar10_224_netG_100_epoch%d.pth'%(n_epoch)), strict=False)
                p_gen.load_state_dict(torch.load('./gen_models/cifar10_224_netG_100_epoch%d.pth'%(n_epoch)), strict=False)

            elif pgen_type.startswith('VAE'):
                N_Z = int(pgen_type[3:])
                assert N_Z == 9408
                p_gen = VAEGenerator(n_channels=n_channels, gpu=args.use_gpu)
                if args.smooth:
                    p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_smooth%s_generator.model'
                                                     % (model_file_name, int(pgen_type[3:]), args.smooth_suffix)))
                else:
                    p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_generator.model'
                                                 % (model_file_name, int(pgen_type[3:]))))
        else:
            print('cifar10', args.cifar10_img_size, pgen_type, "Not implemented")
            assert 0

    elif task == 'mnist':
        # model_file_name, output_file_name = model_settings.get_model_file_name("mnist", args)
        model_file_name = model_settings.get_model_file_name("mnist", args)
        n_channels = 1
        if args.mnist_img_size == 28:
            if pgen_type == 'naive':
                p_gen = None

            elif pgen_type == 'resize':
                p_gen = ResizeGenerator(factor=4.0)

            elif pgen_type == 'DCT':
                p_gen = DCTGenerator(factor=4.0)

            elif pgen_type.startswith('AE'):
                p_gen = MNISTAEGenerator(gpu=args.use_gpu, N_Z=int(pgen_type[2:]))
                p_gen.load_state_dict(torch.load('./gen_models/%s_%d_gradient_ae_%d_generator.model' % (task, args.mnist_img_size, int(pgen_type[2:]))))

            elif pgen_type.startswith('GAN'):
                print("Not implemented yet")
                assert 0

            elif pgen_type.startswith('VAE'):
                print("Not implemented yet")
                assert 0

        elif args.mnist_img_size == 224:
            if pgen_type == 'naive':
                p_gen = None

            elif pgen_type == 'resize9408':
                p_gen = ResizeGenerator(factor=4.0)

            elif pgen_type == 'DCT9408':
                p_gen = DCTGenerator(factor=4.0)

            elif pgen_type == 'PCA9408':
                N_b = 9408
                approx = True
                p_gen = PCAGenerator(N_b=N_b, approx=approx, basis_only=True)
                # p_gen.load('./gen_models/pca_gen_%s_%d.npy' % (model_file_name, 9408))
                p_gen.load('./gen_models/pca_gen_%s_%d.npy' % ('imagenet', 9408))

            elif pgen_type.startswith('AE'):
                if pgen_type.endswith('9408'):
                    p_gen = AEGenerator(n_channels=n_channels, preprocess=None, gpu=args.use_gpu, N_Z=9408)
                else:
                    print("Not implemented yet")
                    assert 0
                p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_ae_%d_generator.model' % (
                model_file_name, int(pgen_type[2:]))))

            elif pgen_type.startswith('GAN'):
                assert int(pgen_type[3:]) == 9408
                p_gen = GANGenerator(n_z=int(pgen_type[3:]), n_channels=n_channels, gpu=args.use_gpu)
                p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_gan_%d_generator.model' % (
                model_file_name, int(pgen_type[3:]))))

            elif pgen_type.startswith('DCGAN'):
                p_gen = MNISTDCGenerator(ngpu=1).eval()
                p_gen.load_state_dict(torch.load('./models/weights/MNIST_netG_epoch_99.pth'), strict=False)

            elif pgen_type.startswith('VAE'):
                if int(pgen_type[3:]) == 9408:
                    p_gen = VAEGenerator(n_channels=n_channels, gpu=args.use_gpu)
                    if args.smooth:
                        p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_smooth%s_generator.model'
                                                         % (model_file_name, 9408, args.smooth_suffix)))
                    else:
                        p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_generator.model'
                                                     % (model_file_name, 9408)))
                elif int(pgen_type[3:]) == 3136:
                    p_gen = MNIST224VAEGenerator(n_channels=n_channels, gpu=args.use_gpu)
                    p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_generator.model'
                                                     % (model_file_name, 3136)))

        else:
            print('mnist', args.mnist_img_size, pgen_type, "Not implemented")
            assert 0

    elif task == 'dogcat2':
        if pgen_type == 'naive':
            p_gen = None

        elif pgen_type == 'resize9408':
            p_gen = ResizeGenerator(factor=4.0)

        elif pgen_type == 'DCT9408':
            p_gen = DCTGenerator(factor=4.0)

        elif pgen_type == 'AE9408':
            p_gen = AEGenerator(n_channels=3, gpu=args.use_gpu, N_Z=9408)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_ae_%d_generator.model' % (task, 9408)))

        elif pgen_type == 'GAN9408':
            p_gen = GANGenerator(n_z=9408, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_gan_%d_generator.model' % (task, 9408)))

        elif pgen_type == 'VAE9408':
            p_gen = VAEGenerator(n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('./gen_models/%s_gradient_vae_%d_generator.model' %(task, 9408)))

    return p_gen


def pgen_input_size(task, pgen_type, args):
    size = None
    if task == 'imagenet' or task == 'celeba' or task == 'celebaid' or task == 'dogcat2' or task == 'celeba2':
        if pgen_type == 'AE128':
            size = (8, 4, 4)
        elif pgen_type == 'AE9408':
            size = (48, 14, 14)
        elif pgen_type == 'VAE9408':
            size = (48, 14, 14)
        elif pgen_type == 'GAN128':
            size = (128, )
        elif pgen_type == 'GAN9408':
            size = (9408, )
        elif pgen_type == 'oldAE9408':
            size = (9408, )
        # elif pgen_type == 'naive':
        elif pgen_type == 'expcos9408':
            size = (48, 14, 14)
        else:
            size = (3, 224, 224)

    elif task == 'cifar10':
        if args.cifar10_img_size == 32:
            if pgen_type == 'AE192':
                size = (12, 4, 4)
            else:
                size = (3, 32, 32)

        elif args.cifar10_img_size == 224:
            if pgen_type == 'AE9408':
                size = (48, 14, 14)
            elif pgen_type == 'VAE9408':
                size = (48, 14, 14)
            elif pgen_type == 'GAN9408':
                size = (9408,)
            else:
                size = (3, 224, 224)

    elif task == 'mnist':
        if args.mnist_img_size == 28:
            if pgen_type.startswith('AE'):
                size = (int(pgen_type[2:]), )
            else:
                size = (28, 28)

        elif args.mnist_img_size == 224:
            if pgen_type == 'AE9408':
                size = (48, 14, 14)
            elif pgen_type == 'VAE9408':
                size = (48, 14, 14)
            elif pgen_type == 'VAE3136':
                size = (16, 14, 14)
            elif pgen_type == 'GAN9408':
                size = (9408,)
            else:
                size = (1, 224, 224)
    return size


def load_imagenet_img(path):
    from PIL import Image
    image = Image.open(path).convert('RGB')
    tmp = np.array(image)
    image = image.resize((224, 224))
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    ### for pytorch ###
    image = image / 255
    image = image.transpose(2, 0, 1)
    return image


def imagenet_attack(args, N_img):
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True).eval()  # for CPU, remove cuda()
    if args.use_gpu:
        resnet18.cuda()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std),
                                         discretize=args.model_discretize)

    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    while (len(src_images) < N_img):
        sid = np.random.randint(280000, 300000)
        tid = np.random.randint(280000, 300000)
        src_image = load_imagenet_img('%s/imagenet/%d.JPEG' % (constants.ROOT_DATA_PATH, sid))
        tgt_image = load_imagenet_img('%s/imagenet/%d.JPEG' % (constants.ROOT_DATA_PATH, tid))
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if (src_label != tgt_label):
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_labels.append(src_label)
            tgt_labels.append(tgt_label)

    mask = None
    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask


def celeba_attack(args, N_img, mounted=False):
    attr_o_i = 'Mouth_Slightly_Open'

    from models import CelebADNN
    model = CelebADNN(model_type='res18', pretrained=False, gpu=args.use_gpu).eval()
    model.load_state_dict(torch.load('./class_models/celeba_%s_%s.model'%(attr_o_i, 'res18')))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=2,
                                         preprocessing=(mean.reshape((3, 1, 1)), std.reshape((3, 1, 1))))

    from datasets import CelebAAttributeDataset
    import torchvision.transforms as transforms
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    list_attr_path = '%s/celeba/list_attr_celeba.txt' %(constants.ROOT_DATA_PATH)
    img_data_path = '%s/celeba/img_align_celeba' %(constants.ROOT_DATA_PATH)
    dataset = CelebAAttributeDataset(attr_o_i, list_attr_path, img_data_path, data_split='test', transform=transform)
    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    used_ids = set()
    while (len(src_images) < N_img):
        sid = np.random.randint(len(dataset))
        tid = np.random.randint(len(dataset))
        if (sid, tid) in used_ids:
            continue
        used_ids.add((sid, tid))
        src_image, src_y = dataset[sid]
        tgt_image, tgt_y = dataset[tid]
        src_image, tgt_image = src_image.numpy(), tgt_image.numpy()
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if (src_label != tgt_label) and (src_y == src_label) and (tgt_y == tgt_label): # predictions should match gt
            # if (True):
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_labels.append(src_label)
            tgt_labels.append(tgt_label)
    mask = None

    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask


def celebaid_attack(args, N_img, num_class, mounted=False):
    from models import CelebAIDDNN
    model = CelebAIDDNN(model_type='res18', num_class=num_class, pretrained=False, gpu=args.use_gpu).eval()
    model.load_state_dict(torch.load('../class_models/celeba_id_%s.model' % ('res18')))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=num_class,
                                         preprocessing=(mean.reshape((3, 1, 1)), std.reshape((3, 1, 1))))

    from datasets import CelebAIDDataset
    import torchvision.transforms as transforms
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    root_dir = constants.ROOT_DATA_PATH
    dataset = CelebAIDDataset(root_dir=root_dir, is_train=False, transform=transform, preprocess=False, random_sample=False, n_id=num_class)
    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    used_ids = set()
    while (len(src_images) < N_img):
        sid = np.random.randint(len(dataset))
        tid = np.random.randint(len(dataset))
        if (sid, tid) in used_ids:
            continue
        used_ids.add((sid, tid))
        src_image, src_y = dataset[sid]
        tgt_image, tgt_y = dataset[tid]
        src_image, tgt_image = src_image.numpy(), tgt_image.numpy()
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if (src_label != tgt_label) and (src_y == src_label) and (tgt_y == tgt_label):  # predictions should match gt
            # if (True):
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_labels.append(src_label)
            tgt_labels.append(tgt_label)
    mask = None

    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask


def cifar10_attack(args, N_img):
    img_size = args.cifar10_img_size
    model_file_name = model_settings.get_model_file_name("cifar10", args)

    from models import CifarDNN
    model = CifarDNN(model_type='res18', pretrained=False, gpu=args.use_gpu, img_size=img_size).eval()
    model.load_state_dict(torch.load('../class_models/%s_res18.model'%(model_file_name)))

    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean, std = constants.get_mean_std('cifar10')
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10,
                                         preprocessing=(mean.reshape((3, 1, 1)), std.reshape((3, 1, 1))))

    import torchvision.datasets as datasets

    transform = model_settings.get_data_transformation_without_normalization('cifar10', args)

    dataset = datasets.CIFAR10(root=f"{constants.ROOT_DATA_PATH}", train=False, download=True, transform=transform)
    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    used_ids = set()
    while (len(src_images) < N_img):
        sid = np.random.randint(len(dataset))
        tid = np.random.randint(len(dataset))
        if (sid, tid) in used_ids:
            continue
        used_ids.add((sid, tid))
        src_image, _ = dataset[sid]
        tgt_image, _ = dataset[tid]
        src_image, tgt_image = src_image.numpy(), tgt_image.numpy()
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if (src_label != tgt_label):
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_labels.append(src_label)
            tgt_labels.append(tgt_label)
    mask = None

    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask


def mnist_attack(args, N_img, num_class, mounted=False):
    img_size = args.mnist_img_size
    model_file_name = model_settings.get_model_file_name("mnist", args)

    from models import MNISTDNN
    model = MNISTDNN(model_type='res18', gpu=args.use_gpu, n_class=num_class, img_size=img_size).eval()
    model.load_state_dict(torch.load('../class_models/%s_%s.model'%(model_file_name, 'res18')))
    mean, std = constants.get_mean_std('mnist')
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=num_class,
                                         preprocessing=(mean, std))

    import torchvision.transforms as transforms
    transform = model_settings.get_data_transformation_without_normalization('mnist', args)

    import torchvision
    dataset = torchvision.datasets.MNIST(root=f"{constants.ROOT_DATA_PATH}", train=False, download=True, transform=transform)
    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    used_ids = set()
    while (len(src_images) < N_img):
        sid = np.random.randint(len(dataset))
        tid = np.random.randint(len(dataset))
        if (sid, tid) in used_ids:
            continue
        used_ids.add((sid, tid))
        src_image, src_y = dataset[sid]
        tgt_image, tgt_y = dataset[tid]
        src_image, tgt_image = src_image.numpy(), tgt_image.numpy()
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if (src_label != tgt_label) and (src_y == src_label) and (tgt_y == tgt_label):  # predictions should match gt
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_labels.append(src_label)
            tgt_labels.append(tgt_label)
    mask = None
    print("MNIST attack, %d src imgs, %d tgt imgs" %(len(src_images), len(tgt_images)))

    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask


def nclass_attack(args, N_img, task, num_class, mounted=False):
    from models import NClassDNN
    model = NClassDNN(model_type='res18', pretrained=False, gpu=args.use_gpu, n_class=num_class).eval()
    model.load_state_dict(torch.load('../class_models/%s%d_%s.model' % (task, num_class, 'res18')))
    mean, std = constants.get_mean_std('%s%d' % (task, num_class))
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=num_class,
                                         preprocessing=(mean.reshape((3, 1, 1)), std.reshape((3, 1, 1))))
    if task == 'dogcat':
        from datasets import DogCatDataset
        import torchvision.transforms as transforms
        image_size = 224
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        dataset = DogCatDataset(root_dir=f'{constants.ROOT_DATA_PATH}/dogcat', mode='test', transform=transform)
    elif task == 'celeba':
        from datasets import CelebABinIDDataset
        import torchvision.transforms as transforms
        image_size = 224
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        root_dir = constants.ROOT_DATA_PATH
        dataset = CelebABinIDDataset(poi=args.celeba_poi, root_dir=root_dir, is_train=False, transform=transform,
                                     get_data=False, random_sample=False, n_id=10) # TODO: change back to test set
    else:
        assert 0
    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    used_ids = set()

    if task == 'celeba':
        all_pos = 0
        pos_imgs = []
        neg_imgs = []
        for _i in range(len(dataset)):
            _img, _lbl = dataset[_i]
            all_pos += _lbl
            _pred = np.argmax(fmodel.forward_one(_img.numpy()))
            if _pred != _lbl:
                continue
            if _lbl == 1:
                pos_imgs.append(_img)
            else:
                neg_imgs.append(_img)
        print("A total of %d true pos and %d true neg" %(len(pos_imgs), len(neg_imgs)))
        print('all pos %d' %(all_pos))
        assert 0
        for p_id in range(len(pos_imgs)):
            for _j in range(int(N_img / len(pos_imgs)) + 1):
                n_id = np.random.randint(len(neg_imgs))
                while (p_id, n_id) in used_ids or (n_id, p_id) in used_ids:
                    n_id = np.random.randint(len(neg_imgs))
                if _j % 2 == 0:
                    used_ids.add((p_id, n_id))
                    src_images.append(pos_imgs[p_id])
                    src_labels.append(1)
                    tgt_images.append(neg_imgs[n_id])
                    tgt_labels.append(0)
                else:
                    used_ids.add((n_id, p_id))
                    tgt_images.append(pos_imgs[p_id])
                    tgt_labels.append(1)
                    src_images.append(neg_imgs[n_id])
                    src_labels.append(0)
    else:
        while (len(src_images) < N_img):
            sid = np.random.randint(len(dataset))
            tid = np.random.randint(len(dataset))
            if (sid, tid) in used_ids:
                continue
            used_ids.add((sid, tid))
            src_image, src_y = dataset[sid]
            tgt_image, tgt_y = dataset[tid]
            src_image, tgt_image = src_image.numpy(), tgt_image.numpy()
            src_label = np.argmax(fmodel.forward_one(src_image))
            tgt_label = np.argmax(fmodel.forward_one(tgt_image))
            if (src_label != tgt_label) and (src_y == src_label) and (tgt_y == tgt_label):  # predictions should match gt
                # if (True):
                src_images.append(src_image)
                tgt_images.append(tgt_image)
                src_labels.append(src_label)
                tgt_labels.append(tgt_label)
    mask = None

    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask


