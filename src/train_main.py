# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_main.py
# @Software : PyCharm

import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.utility import get_time
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_loader import get_train_loader


class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)

    def train_model(self):
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        self.model.train()
        running_loss = 0.
        running_acc = 0.
        running_loss_cls = 0.
        running_loss_ft = 0.
        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False
            print('epoch {} started'.format(e))
            print("lr: ", self.schedule_lr.get_lr())

            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]
                labels = target

                loss, acc, loss_cls, loss_ft = self._train_batch_data(imgs, labels)
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss += loss
                running_acc += acc

                self.step += 1

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss', loss_board, self.step)
                    acc_board = running_acc / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Acc', acc_board, self.step)
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar(
                        'Training/Learning_rate', lr, self.step)
                    loss_cls_board = running_loss_cls / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss_cls', loss_cls_board, self.step)
                    loss_ft_board = running_loss_ft / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss_ft', loss_ft_board, self.step)

                    running_loss = 0.
                    running_acc = 0.
                    running_loss_cls = 0.
                    running_loss_ft = 0.
                if self.step % self.save_every == 0 and self.step != 0:
                    time_stamp = get_time()
                    self._save_state(time_stamp, extra=self.conf.job_name)
            self.schedule_lr.step()

        time_stamp = get_time()
        self._save_state(time_stamp, extra=self.conf.job_name)
        self.writer.close()

    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_fea.item()

    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size}

        model = MultiFTNet(**param).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        torch.save(self.model.state_dict(), save_path + '/' +
                   ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step)))
