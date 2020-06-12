# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_fas.py
# @Software : PyCharm
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch import optim
import torch.nn.init as init
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from utils.utility import get_time
from models.MultiFTNet import MultiFTNet
from utils.dataset_loader import get_train_loader


class TrainFAS():
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.init_test_method()
        self.init_train_loader()

    def train_model(self):
        self.init_model_param()
        self.train_stage()

    def init_test_method(self):
        pass
        #self.roc_evaluate = ROCEvaluate(self.conf)

    def init_train_loader(self):
        self.train_loader = get_train_loader(self.conf)

    def init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self.define_network()
        self.get_optimizer()
        self.schedule_lr = self.get_lr_scheduler(self.optimizer)


    def train_stage(self):
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

            for info in tqdm(iter(self.train_loader)):
                imgs = info[0:2]
                labels = info[-1]

                loss, acc, loss_cls, loss_ft = self.train_batch_data(imgs, labels)
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft

                running_loss += loss
                running_acc += acc

                self.step += 1

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('Training/Loss', loss_board, self.step)
                    acc_board = running_acc / self.board_loss_every
                    self.writer.add_scalar('Training/Acc', acc_board, self.step)
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Training/Learning_rate', lr, self.step)

                    loss_cls_board = running_loss_cls/self.board_loss_every
                    self.writer.add_scalar('Training/Loss_cls', loss_cls_board, self.step)
                    loss_depth_board = running_loss_ft/self.board_loss_every
                    self.writer.add_scalar('Training/Loss_ft', loss_depth_board, self.step)
                    running_loss = 0.
                    running_acc = 0.
                    running_loss_cls = 0.
                    running_loss_ft = 0.
                if self.step % self.save_every == 0 and self.step != 0:
                    time_stamp = get_time()
                    self.save_state(time_stamp, extra=self.conf.job_name)
            self.schedule_lr.step()

        time_stamp = get_time()
        self.save_state(time_stamp, extra=self.conf.job_name)
        self.writer.close()

    def train_batch_data(self, imgs, labels):

        labels = labels.to(self.conf.device)
        self.optimizer.zero_grad()
        embeddings, depth_ = self.model.forward(imgs[0].to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_depth = self.ft_criterion(depth_, imgs[1].to(self.conf.device))
        loss = loss_cls + 5e-5 * loss_depth
        acc = self.get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_depth.item()

    def define_network(self):
        param = {'num_classes': self.conf.num_classes, 'img_channel': self.conf.input_channel,
                 'embedding_size': self.conf.embedding_size, 'conv6_kernel': self.conf.kernel_size}

        model = MultiFTNet(**param).to(self.conf.device)
        # model.apply(self.weight_init)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def get_optimizer(self):

        depth_params = list(map(id, self.model.module.FTGenerator.parameters()))
        base_params = filter(lambda p: id(p) not in depth_params,
                             self.model.module.parameters())
        self.optimizer = optim.SGD([
            {'params': base_params},
            {'params': self.model.module.FTGenerator.parameters(), 'lr': self.conf.lr * 100},
        ], lr=self.conf.lr, weight_decay=5e-4, momentum=self.conf.momentum)

    def get_lr_scheduler(self, optimizer, start_epoch=0):

        if self.conf.schedule_lr_type == "MSTEP":
            print("lr: ", self.conf.lr)
            print("epochs: ", self.conf.epochs)
            print("milestones: ", self.conf.milestones)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self.conf.milestones, self.conf.gamma, start_epoch - 1)
        else:
            lr_scheduler = None

            print('expected scheduler type should be MultiStepLR'
                  'but got {}, please implementing corresponding method'.format(self.conf.schedule_lr_type))
        return lr_scheduler

    def get_accuracy(self, output, target, topk=(1,)):
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

    def save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        torch.save(
            self.model.state_dict(), save_path +'/'+ ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step)))