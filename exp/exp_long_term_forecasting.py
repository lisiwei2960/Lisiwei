from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(
            self.args,
            use_se=self.args.use_se if hasattr(self.args, 'use_se') else False,
            use_hybrid=self.args.use_hybrid if hasattr(self.args, 'use_hybrid') else False
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_losses = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len:, -1 if self.args.features=='MS' else 0:]
                        loss = criterion(outputs, batch_y[:, -self.args.pred_len:, -1 if self.args.features=='MS' else 0:].to(self.device))
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, -1 if self.args.features=='MS' else 0:]
                    loss = criterion(outputs, batch_y[:, -self.args.pred_len:, -1 if self.args.features=='MS' else 0:].to(self.device))
                    loss.backward()
                    optimizer.step()
                train_losses.append(loss.item())

                if (i+1) % 100 == 0:
                    speed = (time.time()-time_now)/100
                    print(f"iters: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f} | speed: {speed:.4f}s/iter")
                    time_now = time.time()

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.2f}")
            train_loss = np.average(train_losses)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch+1, self.args)

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')
        if test:
            print('loading model...')
            self.model.load_state_dict(torch.load(self.args.checkpoints))
        preds, trues = [], []
        folder = f"./test_results/{setting}/"
        os.makedirs(folder, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:,...].cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:,...].cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)
                if i % 20 == 0:
                    inp = batch_x.cpu().numpy()[0]
                    gt = np.concatenate((inp[:, -1], batch_y[0,:, -1]))
                    pd = np.concatenate((inp[:, -1], outputs[0,:, -1]))
                    visual(gt, pd, f"{folder}{i}.png")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        dtw_val = 'Not calculated'
        if self.args.use_dtw:
            dtw_list = []
            for k in range(preds.shape[0]):
                d, *_ = accelerated_dtw(preds[k].reshape(-1,1), trues[k].reshape(-1,1), dist=lambda x,y: abs(x-y))
                dtw_list.append(d)
            dtw_val = np.mean(dtw_list)
        print(f"mse:{mse}, mae:{mae}, dtw:{dtw_val}")
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(f"{setting}mse:{mse}, mae:{mae}, dtw:{dtw_val}")

        # **新增：** 确保保存目录存在
        folder_path = f"./results/{setting}/"
        os.makedirs(folder_path, exist_ok=True)

        np.save(f"./results/{setting}/metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(f"./results/{setting}/pred.npy", preds)
        np.save(f"./results/{setting}/true.npy", trues)