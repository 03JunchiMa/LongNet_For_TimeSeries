import torch
import numpy as np
import argparse
import torch.multiprocessing
from Experiments.exp_longnet_ts import Exp_LongNet

# @Author: Junchi Ma
# @Description: This content is adapted from S-D-Mamba
# @Note: One could run the following script by copy and paste this: python -u run.py --is_training 1 --data ETTh1 --train_epochs 1 --model_id longnet_test --model LongNet
if __name__ == "__main__":

    # create a parser for using timseries data
    parser = argparse.ArgumentParser(description="LongNet_Time_Series")

    # basic config
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="training status"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        default="Test",
        help="model id for path that save the file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="LongNet",
        help="model name, options=[LongNet] (to be added)",
    )

    # data loader
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        default="custom",
        help="the data to be selected from data dict (refer to data provider)",
    )

    parser.add_argument(
        "--root_path",
        type=str,
        default="./dataset/ETT-small/",
        help="root path of the data file",
    )

    parser.add_argument(
        "--data_path", type=str, default="ETTm1.csv", help="data csv file"
    )

    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=str, default=248, help="input sequence length") # 96
    parser.add_argument( # 48
        "--label_len", 
        type=str,
        default=124,
        help="label sequence length (used for during training)",
    )
    parser.add_argument( # 96
        "--pred_len",
        type=str,
        default=248,
        help="prediction sequence length (used for during inference)",
    )

    # here we use the default configuration which is defined in torchscale/architecture/config.py, so we won't include the add_argument for the model config
    parser.add_argument(
        "--enc_in",
        type=int,
        default=7,
        help="encoder input size (this is num_features in our data, and in multivariate predict multivariate case, this dimension will be projected into d_model by tokenlayer)",
    )
    parser.add_argument(
        "--dec_in",
        type=int,
        default=7,
        help="decoder input size (this is num_features in our data, and in multivariate predict multivariate case, this dimension will be projected into d_model by tokenlayer)",
    )
    parser.add_argument(
        "--c_out",
        type=int,
        default=7,
        help="in the end, we will do a project from the d_model to c_out, in the mutlivariate predict multivariate case, the c_out will be equal to num_features, since we want to do the prediction",
    )

    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to predict unseen future data",
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options=:[timeF, fixed, learned]",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=8, help="data loader num workers"
    )
    parser.add_argument("--exp_itr", type=int, default=1, help="experiments times")
    parser.add_argument(
        "--train_epochs", type=int, required=True, default=10, help="train epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size for input training data"
    )
    parser.add_argument("--patience", type=int, default=2, help="early stopping rate")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="optimizer learning rate"
    )
    parser.add_argument(
        "--des", type=str, default="test", help="experiement description"
    )
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default='type1', help='adjust learning rate')
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed presicion training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use_gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device id for multiple gpus"
    )

    # handle the input arguments
    args = parser.parse_args()

    # handle gpu
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(_id) for _id in device_ids]
        args.gpu = device_ids[0]

    # Experiement
    print("Args in experiment:")
    print(args)

    # set up multiprocessing for multi-workers
    torch.multiprocessing.set_sharing_strategy("file_system")

    # set up experiements
    Exp = Exp_LongNet
    exp = Exp(args)

    # Experiment: training
    if args.is_training == 1:
        for ii in range(args.exp_itr):
            # setting record for experiemnts (joined with checkpoints)
            setting = "id->{}_dt->{}_epochs->{}_ft->{}_ii{}".format(
                args.model_id, args.data, args.train_epochs, args.features, ii
            )

            # start training
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            # start testing
            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)

            # prediction
            if args.do_predict:
                print(
                    ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                exp.predict(setting)

            torch.cuda.empty_cache()
    elif args.is_training == 2:
        ii = 0
        # setting record for experiemnts (joined with checkpoints)
        setting = "id->{}_dt->{}_epochs->{}_ft->{}_ii{}".format(
            args.model_id, args.data, args.train_epochs, args.features, ii
        )

        #  testing
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
