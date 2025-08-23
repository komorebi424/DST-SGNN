import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader_L import Dataset_ECG, Dataset_CSI, Dataset_CSI_old
from model.MODEL_L import LDSGNN
import time
import os
import numpy as np
from utils.utils_L import save_model, load_model, evaluate
import random
import torch.nn.functional as F

fix_seed = 77777
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='patch and fourier graph network for multivariate time series forecasting')

parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--data', type=str, default='ETTh1_test', help='data set')
parser.add_argument('--data_pretrain', type=str, default='ETTh1_pretrain', help='data set')
parser.add_argument('--seq_length', type=int, default=96, help='inout length')
parser.add_argument('--pre_length', type=int, default=192, help='predict length')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='input data batch size')
parser.add_argument('--feature_size', type=int, default=7, help='feature size')
parser.add_argument('--out_N', type=int, default=7, help='real size')

parser.add_argument('--patch_len', type=int, default=48, help='patch_len')
parser.add_argument('--d_model', type=int, default=24, help='patch_len')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=1024, help='hidden dimensions')
parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--petrain_train', type=float, default=0.8)
parser.add_argument('--petrain_val', type=float, default=0.2)
parser.add_argument('--test_train', type=float, default=0)
parser.add_argument('--test_val', type=float, default=0)
parser.add_argument('--stride', type=float, default=0)

parser.add_argument('--device', type=str, default='cuda:0', help='device')
args = parser.parse_args()
print(f'Training configs: {args}')

result_train_file = os.path.join('output_best_0', args.data_pretrain, 'train')

if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)

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
    'combined': {'root_path': 'data/combined.csv', 'type': '1'},
    'combined_it': {'root_path': 'data/combined_it.csv', 'type': '1'},
    'padded_ETTh1': {'root_path': 'data/padded_ETTh1.csv', 'type': '1'},
    'ETTm1_pretrain': {'root_path': 'data/ETTm1_pretrain.csv', 'type': '1'},
    'ETTm2_pretrain': {'root_path': 'data/ETTm2_pretrain.csv', 'type': '1'},
    'ETTh2_pretrain': {'root_path': 'data/ETTh2_pretrain.csv', 'type': '1'},
    'ETTh1_pretrain': {'root_path': 'data/ETTh1_pretrain.csv', 'type': '1'},
    'ETTm1_test': {'root_path': 'data/ETTm1_test.csv', 'type': '1'},
    'ETTm2_test': {'root_path': 'data/ETTm2_test.csv', 'type': '1'},
    'ETTh2_test': {'root_path': 'data/ETTh2_test.csv', 'type': '1'},
    'ETTh1_test': {'root_path': 'data/ETTh1_test.csv', 'type': '1'},
    'PEMS03_pretrain': {'root_path': 'data/PEMS03_pretrain.csv', 'type': '1'},
    'PEMS04_pretrain': {'root_path': 'data/PEMS04_pretrain.csv', 'type': '1'},
    'PEMS07_pretrain': {'root_path': 'data/PEMS07_pretrain.csv', 'type': '1'},
    'PEMS08_pretrain': {'root_path': 'data/PEMS08_pretrain.csv', 'type': '1'},
    'PEMS03_test': {'root_path': 'data/PEMS03_test.csv', 'type': '1'},
    'PEMS04_test': {'root_path': 'data/PEMS04_test.csv', 'type': '1'},
    'PEMS07_test': {'root_path': 'data/PEMS07_test.csv', 'type': '1'},
    'PEMS08_test': {'root_path': 'data/PEMS08_test.csv', 'type': '1'},
    'CSI300_pretrain': {'root_path': 'data/CSI300_pretrain.csv', 'type': '1'},
    'NASDAQ100_pretrain': {'root_path': 'data/NASDAQ100_pretrain.csv', 'type': '1'},
    'CSI300_test': {'root_path': 'data/CSI300_test.csv', 'type': '1'},
    'NASDAQ100_test': {'root_path': 'data/NASDAQ100_test.csv', 'type': '1'},

}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
if args.data_pretrain in data_parser.keys():
    data_info_pretrain = data_parser[args.data_pretrain]

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
    'ETTm1_pretrain': Dataset_CSI,
    'ETTm2_pretrain': Dataset_CSI,
    'ETTh1_pretrain': Dataset_CSI,
    'ETTh2_pretrain': Dataset_CSI,
    'ETTm1_test': Dataset_CSI,
    'ETTm2_test': Dataset_CSI,
    'ETTh1_test': Dataset_CSI,
    'ETTh2_test': Dataset_CSI,
    'ECG': Dataset_CSI,
    'FC1-test': Dataset_CSI,
    'electricity': Dataset_CSI,
    'exchange_rate': Dataset_CSI,
    'national_illness': Dataset_CSI,
    'CSI300_pretrain': Dataset_CSI,
    'CSI300_test': Dataset_CSI,
    'NASDAQ100_pretrain': Dataset_CSI,
    'NASDAQ100_test': Dataset_CSI,
    'PEMS07_pretrain': Dataset_CSI,
    'PEMS04_pretrain': Dataset_CSI,
    'PEMS03_pretrain': Dataset_CSI,
    'PEMS08_pretrain': Dataset_CSI,
    'PEMS07_test': Dataset_CSI,
    'PEMS04_test': Dataset_CSI,
    'PEMS03_test': Dataset_CSI,
    'PEMS08_test': Dataset_CSI,
    'weather': Dataset_CSI,
    'traffic': Dataset_CSI,
    'electricity': Dataset_CSI,
    'Solar': Dataset_CSI,
    'combined': Dataset_CSI,
    'combined_it': Dataset_CSI,
    'padded_ETTh1': Dataset_CSI,
    'METR-LA': Dataset_CSI
}

Data = data_dict[args.data]

train_set = Data(root_path=data_info_pretrain['root_path'], flag='train', seq_len=args.seq_length,
                 pre_len=args.pre_length,
                 type=data_info_pretrain['type'], train_ratio=args.petrain_train, val_ratio=args.petrain_val)

test_set = Data(root_path=data_info['root_path'], flag='test', seq_len=args.seq_length, pre_len=args.pre_length,
                type=data_info['type'], train_ratio=args.test_train, val_ratio=args.test_val)

val_set = Data(root_path=data_info_pretrain['root_path'], flag='val', seq_len=args.seq_length, pre_len=args.pre_length,
               type=data_info_pretrain['type'], train_ratio=args.petrain_train, val_ratio=args.petrain_val)

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
model = LDSGNN(stride=args.stride, pre_length=args.pre_length, embed_size=args.embed_size,
               feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size,
               patch_len=args.patch_len, d_model=args.d_model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
my_optim_large_model = torch.optim.Adagrad(params=model.parameters(), lr=args.learning_rate, lr_decay=0, weight_decay=0,
                                           initial_accumulator_value=0)

my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim_large_model, gamma=args.decay_rate)

forecast_loss = nn.L1Loss(reduction='mean').to(device)


def validate(model, vali_loader):
    model.eval()

    cnt = 0
    loss_total = 0.0
    total_mae, total_mse, total_smape = 0.0, 0.0, 0.0

    with torch.no_grad():
        for index, (x, y) in enumerate(vali_loader):
            cnt += 1
            y = y.float().to("cuda:0")
            x = x.float().to("cuda:0")

            Y = model(x)
            y = y.permute(0, 2, 1).contiguous()
            loss = forecast_loss(Y.float(), y.float())

            loss_total += float(loss)

            pred = Y.detach().cpu().numpy()
            true = y.detach().cpu().numpy()

            mae, mse, smape = evaluate(true, pred)
            total_mae += mae
            total_mse += mse
            total_smape += smape

            del x, y, Y
            torch.cuda.empty_cache()

    avg_mae = total_mae / cnt
    avg_mse = total_mse / cnt
    avg_smape = total_smape / cnt

    print(f'NORM: MSE {avg_mse:7.9f}; MAE {avg_mae:7.9f}.')

    model.train()
    return loss_total / cnt


def test():
    result_test_file = 'output_best_0/' + args.data_pretrain + '/train'
    model = load_model(result_test_file, f'best_model_pre{args.pre_length}')
    model.eval()

    total_mae_1, total_mse_1, total_smape_1 = 0.0, 0.0, 0.0

    count = 0

    with torch.no_grad():
        for index, (x, y) in enumerate(test_dataloader):
            y = y.float().to("cuda:0")
            x = x.float().to("cuda:0")

            Y = model(x)
            y = y.permute(0, 2, 1).contiguous()

            pred = Y.detach().cpu().numpy()
            true = y.detach().cpu().numpy()

            bs = x.shape[0]

            pred_1 = pred[:, :args.out_N, :]
            true_1 = true[:, :args.out_N, :]
            mae_1, mse_1, smape_1 = evaluate(true_1, pred_1)
            total_mae_1 += mae_1 * bs
            total_mse_1 += mse_1 * bs
            total_smape_1 += smape_1 * bs


            count += bs

            del x, y, Y
            torch.cuda.empty_cache()

    print(f'NORM: MAE {total_mae_1 / count:.9f}; MSE {total_mse_1 / count:.9f}; '
          f'RMSE {np.sqrt(total_mse_1 / count):.9f}; sMAPE {total_smape_1 / count:.9f}')



if __name__ == '__main__':
    if args.is_training:
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
                # my_lr_scheduler1.step()
                # my_lr_scheduler2.step()

            if (epoch + 1) % args.validate_freq == 0:
                val_loss = validate(model, val_dataloader)

                if val_loss < lowest_mae:
                    lowest_mae = val_loss

                    save_model(model, result_train_file, f'best_model_pre{args.pre_length}')
            print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | val_loss {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))

            save_model(model, result_train_file, epoch)

    start_time = time.time()
    test()
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
