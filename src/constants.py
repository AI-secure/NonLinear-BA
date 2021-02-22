import numpy as np

def get_mean_std(TASK):
    if TASK == 'dogcat2':
        return np.array([0, 0, 0]), np.array([1, 1, 1])
    elif TASK == 'celeba2':
        return np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    elif TASK.startswith('mnist'):
        return np.array([0.1307,]), np.array([0.3081,])
    elif TASK.startswith('cifar10'):
        return np.array([0.485, 0.456, 0.406]), np.array(([0.229, 0.224, 0.225]))
    else:
        assert 0


ROOT_DATA_PATH = "../data"
RAW_DATA_PATH = "../raw"



