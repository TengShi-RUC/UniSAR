import argparse
import datetime
import logging
import os

from models import *
from utils import *
from utils import const, utils


def parse_global_args(parser: argparse.ArgumentParser):
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--random_seed', type=int, default=20230601)
    parser.add_argument('--time', type=str, default='none')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--test_path', type=str, default="")

    parser.add_argument('--data', type=str, default='KuaiSAR')
    parser.add_argument('--model', type=str, default='UniSAR')
    return parser


if __name__ == '__main__':
    global_start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = UniSAR.parse_model_args(parser)
    parser = SarRunner.parse_runner_args(parser)
    args, extras = parser.parse_known_args()

    if args.gpu == 'cpu':
        args.device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if args.gpu != '' and torch.cuda.is_available():
            args.device = torch.device('cuda')

    if args.data == 'KuaiSAR':
        const.init_setting_KuaiSAR()
    elif args.data == 'Amazon':
        const.init_setting_Amazon()
    else:
        raise ValueError('Dataset Error')

    utils.setup_seed(args.random_seed)

    if args.time == 'none':
        cur_time = datetime.datetime.now()
        args.time = cur_time.strftime(r"%Y%m%d-%H%M%S")

    args.model_path = "output/{}/{}/checkpoints/{}".format(
        args.data, args.model, args.time)

    utils.load_hyperparam(args)
    utils.set_logging(args)
    print(args)
    for flag, value in sorted(args.__dict__.items(), key=lambda x: x[0]):
        logging.info('{}: {}'.format(flag, value))

    model: BaseModel = UniSAR(args)
    runner: BaseRunner = SarRunner(args)

    num_parameters = model.count_variables()
    logging.info("num model parameters:{}".format(num_parameters))

    if args.train == 0:
        model.load_model(model_path=args.test_path)
        test_result, _ = runner.evaluate(model, 'test')
        logging.info("Test Result:")
        logging.info(utils.format_metric(test_result))
    else:
        runner.train(model)

    global_end_time = datetime.datetime.now()
    logging.info("running used time:{}".format(global_end_time -
                                               global_start_time))
