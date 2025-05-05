
import numpy as np
import torch
import os

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_dhfm.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    if not model_dir or not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' doesn't exist.")
        return None
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_dhfm.pt')
    if not os.path.exists(file_name):
        print(f"Model file '{file_name}' doesn't exist.")
        return None
    try:
        model = torch.load(file_name)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def MAPE(v, v_, axis=None):
    mape = (np.abs(v_ - v) / (np.abs(v)+1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)

def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)

def RMSE(v, v_, axis=None):
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MSE(v, v_, axis=None):
    return np.mean((v_ - v) ** 2, axis).astype(np.float64)


def evaluate(y, y_hat, by_step=False, by_node=False):

    if not by_step and not by_node:
        return MSE(y, y_hat, axis=None), MAE(y, y_hat, axis=None)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))
