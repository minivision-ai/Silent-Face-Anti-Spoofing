# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:39
# @Author : zhuying
# @Company : Minivision
# @File : train.py
# @Software : PyCharm
import argparse
import os
from src.default_config import get_default_config, update_config
from train_fas import TrainFAS


def parse_args():
    """parsing and configuration"""
    desc = "Silence-FAS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_ids", type=str, default="1", help="which gpu id, 0123")
    parser.add_argument("--train_set", type=str, default="fake-OS-ft-200605-train-data", help=" ")
    parser.add_argument("--num_classes", type=int, default=3, help="the classes of the model:[2 / 3 / 4]")
    parser.add_argument("--patch_info", type=str, default="1_0_0_80x80",
                        help="[org_1_80x60 / 1_0_0_80x80 / 2.7_0_0_80x80 / 4_0_0_80x80]")
    parser.add_argument("--model", type=str, default="MultiFTNet", help=" ")
    args = parser.parse_args()
    cuda_devices = [int(elem) for elem in args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    args.devices = [x for x in range(len(cuda_devices))]
    return args


if __name__ == "__main__":
    args = parse_args()
    conf = get_default_config()

    file_name = 'LiveBody-P-DA_{}_{}_{}-c3.py'.format(args.patch_info, args.train_set, args.model)
    print(file_name)
    conf = update_config(args, conf, file_name)
    trainer = TrainFAS(conf)
    trainer.train_model()

