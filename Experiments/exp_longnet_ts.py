import os
import time
import torch
from torch import optim, nn
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import numpy as np
from data_provider.data_factory import data_provider
from dilated_attention_pytorch.long_net import LongNetTS

# @Author: Junchi Ma
# @Description: The experiement for longnet_ts, this is adapted from S-D-Mamba
class Exp_LongNet(object):
    def __init__(self, args):
        self.args = args

        self.dtype = torch.float16
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        # the longnet for the time series data config
        model = LongNetTS(
            num_features=7,  # Number of features in the time series data
            d_model=128,  # the dimension for the model (embedding dimension)
            nhead=8,  # number of attention heads
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=512,
            segment_lengths=[
                96,
                192,
                384,
                768,
                1536,
            ],  # Example segment lengths adjusted for the input sequence length
            dilation_rates=[1, 2, 4, 6, 12],  # Example dilation rates
            dropout=0.1,
            activation="relu",
            layer_norm_eps=1e-5,
            pred_len=self.args.pred_len,  # Prediction length
            device=self.device,
            dtype=self.dtype,
        )
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # get device (cpu/gpu)
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu if not self.args.use_multi_gpu else self.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    # get the dataset and dataloader using data_provider
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        print("loaded")
        return data_set, data_loader

    # self-defined optimizer
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # self-defined loss function
    def _select_loss(self):
        loss = nn.MSELoss()
        return loss

    # training function-
    def train(self, setting):
        train_data, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")
        test_data, test_loader = self._get_data("test")

        # checkpoint path
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # define training essential ingredients
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        loss = self._select_loss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # training loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # set model to be training mode
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()

                # move data to device
                # batch_x.shape: torch.Size([32, 96, 7])
                # batch_y.shape: torch.Size([32, 144, 7])
                batch_x = batch_x.half().to(self.device)
                batch_y = batch_y.half().to(self.device)

                print("Forward Pass")
                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # select the last feature if we are using multiple features to predict the single feature
                f_dim = -1 if self.args.features == "MS" else 0

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                loss_val = loss(outputs, batch_y)
                train_loss.append(loss_val.item())

                # Backward pass
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, loss)
            test_loss = self.vali(test_data, test_loader, loss)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # In training vali function for monitoring the training
    def vali(self, vali_data, vali_loader, loss):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.half().to(self.device)
                batch_y = batch_y.half()

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # Select the last pred_len steps
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                # Compute loss
                loss_value = loss(outputs, batch_y)
                total_loss.append(loss_value.item())

        total_loss = np.average(total_loss)
        self.model.train()  # Set the model back to training mode
        return total_loss

    # Test function, do it after the training
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.half().to(self.device)
                batch_y = batch_y.half().to(self.device)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # Select the last pred_len steps
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # Optionally apply inverse transform
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(
                        shape
                    )
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(
                        shape
                    )

                preds.append(outputs)
                trues.append(batch_y)

                # Optionally visualize results
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(
                            shape
                        )
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # Calculate and save metrics
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        with open("result_long_term_forecast.txt", "a") as f:
            f.write(setting + "  \n")
            f.write("mse:{}, mae:{}".format(mse, mae))
            f.write("\n")
            f.write("\n")

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return

    # The prediction function for longnet in time series data
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.half().to(self.device)
                batch_y = batch_y.half().to(self.device)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # Select the last pred_len steps
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(
                        shape
                    )
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # Save the predictions
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return
