
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader import Dataset_ECG, Dataset_CSI, Dataset_CSI_old
from model.MODEL import DSTSGNN
import time
import os
import numpy as np
from utils.utils import save_model, load_model, evaluate
import random
import torch.nn.functional as F

fix_seed = 77777
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='patch and fourier graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='PEMS03', help='data set')
parser.add_argument('--feature_size', type=int, default='358', help='feature size')
parser.add_argument('--seq_length', type=int, default=336, help='inout length')
parser.add_argument('--pre_length', type=int, default=96, help='predict length')
parser.add_argument('--patch_len', type=int, default=48, help='patch_len')
parser.add_argument('--d_model', type=int, default=128, help='patch_len')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=1024, help='hidden dimensions')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=2, help='input data batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.1)
parser.add_argument('--stride', type=float, default=0)

parser.add_argument('--device', type=str, default='cuda:0', help='device')
args = parser.parse_args()
print(f'Training configs: {args}')


result_train_file = os.path.join('output_best', args.data, 'train')

result_test_file = os.path.join('output_best', args.data, 'test')

if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)

if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)


data_parser = {  
    'stock_open': {'root_path': 'data/stock_processed/stock_open.csv', 'type': '1'},
    'stock_close': {'root_path': 'data/stock_processed/stock_close.csv', 'type': '1'},
    'stock_high': {'root_path': 'data/stock_processed/stock_high.csv', 'type': '1'},
    'stock_low': {'root_path': 'data/stock_processed/stock_low.csv', 'type': '1'},
    'stock_volume': {'root_path': 'data/stock_processed/stock_volume.csv', 'type': '1'},
    'ETTm1': {'root_path': 'data/ETTm1.csv', 'type': '1'},
    'ETTm2': {'root_path': 'data/ETTm2.csv', 'type': '1'},
    'ECG': {'root_path': 'data/ECG.csv', 'type': '1'},
    'FC1-test': {'root_path': 'data/FC1-test.csv', 'type': '1'},
    'electricity': {'root_path': 'data/electricity.csv', 'type': '1'},
    'ETTh2': {'root_path': 'data/ETTh2.csv', 'type': '1'},
    'ETTh1': {'root_path': 'data/ETTh1.csv', 'type': '1'},
    'exchange_rate': {'root_path': 'data/exchange_rate.csv', 'type': '1'},
    'national_illness': {'root_path': 'data/national_illness.csv', 'type': '1'},
    'CSI300': {'root_path': 'data/CSI300.csv', 'type': '1'},
    'PEMS07': {'root_path': 'data/PEMS07.csv', 'type': '1'},
    'PEMS04': {'root_path': 'data/PEMS04.csv', 'type': '1'},
    'PEMS03': {'root_path': 'data/PEMS03.csv', 'type': '1'},
    'weather': {'root_path': 'data/weather.csv', 'type': '1'},
    'traffic': {'root_path': 'data/traffic.csv', 'type': '1'},
    'electricity': {'root_path': 'data/electricity.csv', 'type': '1'},
    'Solar': {'root_path': 'data/Solar.csv', 'type': '1'},
    'METR-LA': {'root_path': 'data/METR-LA.csv', 'type': '1'},

}


if args.data in data_parser.keys():
    data_info = data_parser[args.data]

data_dict = {  
    'stock_open': Dataset_CSI,
    'stock_close': Dataset_CSI,
    'stock_low': Dataset_CSI,
    'stock_high': Dataset_CSI,
    'stock_volume': Dataset_CSI,
    'ETTm1': Dataset_CSI,
    'ETTm2': Dataset_CSI,
    'ETTh1': Dataset_CSI,
    'ETTh2': Dataset_CSI,
    'ECG': Dataset_CSI,
    'FC1-test': Dataset_CSI,
    'electricity': Dataset_CSI,  
    'exchange_rate': Dataset_CSI,
    'national_illness': Dataset_CSI,
    'CSI300': Dataset_CSI,
    'PEMS07': Dataset_CSI,
    'PEMS04': Dataset_CSI,
    'PEMS03': Dataset_CSI,
    'weather': Dataset_CSI,
    'traffic': Dataset_CSI,
    'electricity': Dataset_CSI,
    'Solar': Dataset_CSI,
    'METR-LA': Dataset_CSI
}

Data = data_dict[args.data]



train_set = Data(root_path=data_info['root_path'], flag='train', seq_len=args.seq_length, pre_len=args.pre_length,
                 type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)

test_set = Data(root_path=data_info['root_path'], flag='test', seq_len=args.seq_length, pre_len=args.pre_length,
                type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)

val_set = Data(root_path=data_info['root_path'], flag='val', seq_len=args.seq_length, pre_len=args.pre_length,
               type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)

train_dataloader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

test_dataloader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

val_dataloader = DataLoader(
    val_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DSTSGNN(stride=args.stride, pre_length=args.pre_length, embed_size=args.embed_size,
                   feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size,
                   patch_len=args.patch_len, d_model=args.d_model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
my_optim_large_model = torch.optim.Adagrad(params=model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)


my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim_large_model, gamma=args.decay_rate)


forecast_loss = nn.L1Loss(reduction='mean').to(device)


def validate(model, vali_loader): 
    model.eval()

    cnt = 0  
    loss_total = 0  
    preds = []  
    trues = []  
    for index, (x, y) in enumerate(vali_loader):  
        cnt += 1
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")

        Y = model(x)
        y = y.permute(0, 2, 1).contiguous()
        loss = forecast_loss(Y.float(), y.float())

        loss_total += float(loss)  
        forecast = Y.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        preds.append(forecast)
        trues.append(y)


    preds = np.array(preds, dtype=object)
    trues = np.array(trues, dtype=object)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)


    score = evaluate(trues, preds)

    print(f'NORM : MSE {score[0]:7.9f}; MAE {score[1]:7.9f}.')

    model.train()
    return loss_total / cnt



def test():
    result_test_file = 'output_best/' + args.data + '/train'

    model = load_model(result_test_file, 'best_model')  

    model.eval()  
    preds = []  
    trues = []  
    sne = []
    for index, (x, y) in enumerate(test_dataloader): 
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")

        Y = model(x)
        y = y.permute(0, 2, 1).contiguous()
        forecast = Y.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        preds.append(forecast)
        trues.append(y)
    preds = np.array(preds, dtype=object)
    trues = np.array(trues, dtype=object)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    score = evaluate(trues, preds)

    print(f'NORM : MSE {score[0]:7.9f}; MAE {score[1]:7.9f}.')



if __name__ == '__main__':

    lowest_mae = float('inf')  
    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0

        all_weights = []

        for index, (x, y) in enumerate(train_dataloader):
            cnt += 1
            y = y.float().to("cuda:0")
            x = x.float().to("cuda:0")

            Y = model(x)
            y = y.permute(0, 2, 1).contiguous()

            loss = forecast_loss(Y.float(), y.float())

            my_optim_large_model.zero_grad()
            loss.backward(retain_graph=True) 
            my_optim_large_model.step()


            loss_total += float(loss)

        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()

        if (epoch + 1) % args.validate_freq == 0:
            val_loss = validate(model, val_dataloader)

            if val_loss < lowest_mae:
                lowest_mae = val_loss

                save_model(model, result_train_file, 'best_model')
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | val_loss {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))

        save_model(model, result_train_file, epoch)
        
    start_time = time.time()
    test()
    end_time = time.time()  
    run_time = end_time - start_time  
    print(f"Inference time of the code: {run_time} seconds")
