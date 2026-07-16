import argparse
import os

import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.utility import get_time
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_loader import get_train_loader
from src.default_config import get_default_config, update_config


class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)

        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)
        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        if conf.resume:
            self._load_checkpoint(conf.resume)

    def train(self):
        self.model.train()
        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

        running_loss = 0.
        running_acc = 0.
        running_loss_cls = 0.
        running_loss_ft = 0.
        writer = SummaryWriter(self.conf.log_path)

        for e in range(self.start_epoch, self.conf.epochs):
            print('epoch {} started'.format(e))
            print("lr: ", self.schedule_lr.get_last_lr())

            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                loss, acc, loss_cls, loss_ft = self._train_batch(sample, ft_sample, target)
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss += loss
                running_acc += acc

                self.step += 1

                if self.step % self.board_loss_every == 0:
                    writer.add_scalar('Training/Loss', running_loss / self.board_loss_every, self.step)
                    writer.add_scalar('Training/Acc', running_acc / self.board_loss_every, self.step)
                    writer.add_scalar('Training/Learning_rate', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('Training/Loss_cls', running_loss_cls / self.board_loss_every, self.step)
                    writer.add_scalar('Training/Loss_ft', running_loss_ft / self.board_loss_every, self.step)
                    running_loss = running_acc = running_loss_cls = running_loss_ft = 0.

                if self.step % self.save_every == 0:
                    self._save_checkpoint(e)

            self.schedule_lr.step()

        self._save_checkpoint(self.conf.epochs, final=True)
        writer.close()

    def _train_batch(self, sample, ft_sample, target):
        self.optimizer.zero_grad()
        sample = sample.to(self.conf.device)
        ft_sample = ft_sample.to(self.conf.device)
        target = target.to(self.conf.device)

        embeddings, feature_map = self.model.forward(sample)

        loss_cls = self.cls_criterion(embeddings, target)
        loss_fea = self.ft_criterion(feature_map, ft_sample)
        loss = 0.5 * loss_cls + 0.5 * loss_fea

        loss.backward()
        self.optimizer.step()

        acc = self._accuracy(embeddings, target)
        return loss.item(), acc, loss_cls.item(), loss_fea.item()

    def _define_network(self):
        model = MultiFTNet(
            num_classes=self.conf.num_classes,
            img_channel=self.conf.input_channel,
            embedding_size=self.conf.embedding_size,
            conv6_kernel=self.conf.kernel_size,
        ).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)
        return model

    @staticmethod
    def _accuracy(output, target):
        pred = output.argmax(dim=1)
        return pred.eq(target).float().mean().item()

    def _save_checkpoint(self, epoch, final=False):
        state = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.schedule_lr.state_dict(),
        }
        tag = 'final' if final else 'iter-{}'.format(self.step)
        save_path = os.path.join(self.conf.model_path, '{}_{}_{}.pth'.format(
            get_time(), self.conf.job_name, tag))
        torch.save(state, save_path)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.conf.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.schedule_lr.load_state_dict(checkpoint['scheduler_state_dict'])
            self.step = checkpoint['step']
            self.start_epoch = checkpoint['epoch']
            print("Resumed training from '{}' at epoch {}, step {}".format(
                checkpoint_path, self.start_epoch, self.step))
        else:
            # bare state_dict (e.g. one of the shipped inference models) - use its
            # weights as a starting point and train from epoch 0
            state_dict = checkpoint
            if next(iter(state_dict)).startswith('module.'):
                self.model.load_state_dict(state_dict)
            else:
                self.model.module.load_state_dict(state_dict)
            print("Initialized model weights from '{}'".format(checkpoint_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Silence-FAS")
    parser.add_argument("--device_ids", type=str, default="1", help="which gpu id, 0123")
    parser.add_argument("--patch_info", type=str, default="1_80x80",
                        help="[org_1_80x60 / 1_80x80 / 2.7_80x80 / 4_80x80]")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a checkpoint (or bare model weights) to resume/fine-tune from")
    args = parser.parse_args()
    cuda_devices = [int(elem) for elem in args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    args.devices = [x for x in range(len(cuda_devices))]
    return args


if __name__ == "__main__":
    args = parse_args()
    conf = get_default_config()
    conf = update_config(args, conf)
    Trainer(conf).train()
