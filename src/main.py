import os
import argparse

# import internal libs
from data import prepare_dataset
from model import prepare_model
from train import train
from utils import set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from config import DATE, MOMENT, SRC_PATH

def add_args() -> argparse.Namespace:
    """get arguments from the program.
    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple baseline")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the random seed.")
    # parser.add_argument("--load", default='D:\\2021_Summer\\VE450\\models\\wind-power-forecast\\kdd\\model_weight_20000.pth', type=str,
    #                     help="load checkpoint.")
    parser.add_argument("--load", default='..\kdd\windsp.pth', type=str,
                        help="load checkpoint.")
    # parser.add_argument("--load", default=None, type=str,
    #                     help="load checkpoint.")
    parser.add_argument("--save", default='./output/', type=str,
                        help="save checkpoint path.")
    parser.add_argument("--s_iter", default=1000, type=int,
                        help="save checkpoint per s_iter iters.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    # parser.add_argument("--dataset", default="kddcup", type=str,
    #                     help='the dataset name.')
    parser.add_argument("--dataset", default="windsp", type=str,
                        help='the dataset name.')
    # parser.add_argument('--data_path', default='D:\\2021_Summer\\VE450\\models\\wind-power-forecast\\kdd\\kdd_train.csv', type=str,
    #                     help='the dataset path.')
    # parser.add_argument('--data_path_test', default='D:\\2021_Summer\\VE450\\models\\wind-power-forecast\\kdd\\kdd_test.csv', type=str,
    #                     help='the test dataset path.')
    parser.add_argument('--data_path', default='D:\\2021_Summer\\VE450\\models\\wind-power-forecast\\kdd\\save_upload_table(2).xlsx', type=str,
                        help='the dataset path.')
    parser.add_argument('--data_path_test', default='D:\\2021_Summer\\VE450\\models\\wind-power-forecast\\kdd\\windsp_test_new.xlsx', type=str,
                        help='the test dataset path.')
    # parser.add_argument("--step", default=256, type=int,
    #                     help="step")
    parser.add_argument("--step", default=15, type=int,
                        help="step")
    # parser.add_argument("--out_step", default=144, type=int,
    #                     help="set the random seed.")
    parser.add_argument("--out_step", default=1, type=int,
                        help="set the random seed.")
    # parser.add_argument("--model", default='attention_mlp', type=str,
    #                     help='the model name.')

    parser.add_argument("--model", default='lstm_turbine', type=str,
                        help='the model name.')                   
    parser.add_argument("--bs", default=1, type=int,
                        help="set the batch size")
    parser.add_argument("--lr", default=0.000001, type=float,
                        help="set the learning rate")
    parser.add_argument("--epochs", default=60, type=int,
                        help="set the number of epochs")
    parser.add_argument("--ev_only", default=True, type=int,
                        help="set the number of epochs")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([DATE, 
                         MOMENT,
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"bs{args.bs}",
                         f"lr{args.lr}",
                         f"epochs{args.epochs}"])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)

    # show the args.
    logger.info("#########parameters settings....")
    import config
    log_settings(args, config.__dict__)

    # save the current src
    save_current_src(save_path = args.save_path, 
                     src_path = SRC_PATH)

    # prepare the dataset
    logger.info("#########preparing dataset....")
    train_set, val_set = prepare_dataset(args.dataset, args.data_path, args.step, args.out_step)
    test_set, _ = prepare_dataset(args.dataset, args.data_path_test, args.step, args.out_step)

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.device, args.model, args.dataset, args.step, args.out_step)
    
    # train the model
    logger.info("#########training model....")
    train(device = args.device,
          model = model,
          train_set = train_set,
          val_set = test_set,
          batch_size = args.bs,
          lr = args.lr, 
          epochs = args.epochs,
          load = args.load,
          save = args.save,
          s_iter = args.s_iter,
          ev_only = args.ev_only,
          )


if __name__ == "__main__":
    main()