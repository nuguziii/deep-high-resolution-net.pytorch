# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy, accuracy_classification, accuracy_landmark
from utils.vis import save_result_images, save_debug_images, save_images_landmark


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_classifier = AverageMeter()
    loss_landmark = AverageMeter()
    acc = AverageMeter()
    acc_cls = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        classification, landmark = model(input)

        #target2 = meta["visible"].type(torch.FloatTensor).cuda(non_blocking=True).view(classification.size(0),-1)
        target = meta["visible"].type(torch.FloatTensor).cuda(non_blocking=True)
        classloss = criterion[0](classification, target)

        target2 = meta["joints"].reshape(-1,64).type(torch.FloatTensor).cuda(non_blocking=True)
        lmloss = criterion[1](landmark, target2)

        #loss = config.TRAIN.LOSS_WEIGHT[0]*classloss + config.TRAIN.LOSS_WEIGHT[1]*lmloss
        loss = config.TRAIN.LOSS_WEIGHT[1] * lmloss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        loss_classifier.update(classloss.item(), input.size(0))
        loss_landmark.update(lmloss.item(), input.size(0))

        avg_acc, cnt= accuracy_landmark(landmark.detach().cpu().numpy(),
                                                    target2.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        avg_acc, cnt = accuracy_classification(classification.detach().cpu().numpy(),
                                                   target.detach().cpu().numpy())
        acc_cls.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f}) ({classific.avg: .5f}+{lm.avg: .5f})\t' \
                  'Accuracy(landmark) {acc.val:.3f} ({acc.avg:.3f})\t'\
                   'Accuracy(classification) {acc_cls.val:.3f} ({acc_cls.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time,
                      loss=losses, classific=loss_classifier, lm=loss_landmark,
                      acc=acc, acc_cls=acc_cls)
            logger.info(msg)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            classification, landmark = model(input)

            target = meta["visible"].type(torch.FloatTensor).cuda(non_blocking=True)
            classloss = criterion[0](classification, target)

            target2 = meta["joints"].reshape(-1, 64).type(torch.FloatTensor).cuda(non_blocking=True)
            lmloss = criterion[1](landmark, target2)

            #loss = config.TRAIN.LOSS_WEIGHT[0]*classloss + config.TRAIN.LOSS_WEIGHT[1]*lmloss
            loss = config.TRAIN.LOSS_WEIGHT[1] * lmloss

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            avg_acc, cnt = accuracy_landmark(landmark.detach().cpu().numpy(),
                                             target2.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

    return acc.avg

def test(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_cls = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            classification, landmark = model(input)

            target = meta["visible"].type(torch.FloatTensor).cuda(non_blocking=True)
            classloss = criterion[0](classification, target)

            target2 = meta["joints"].reshape(-1, 64).type(torch.FloatTensor).cuda(non_blocking=True)
            lmloss = criterion[1](landmark, target2)

            #loss = config.TRAIN.LOSS_WEIGHT[0] * classloss + config.TRAIN.LOSS_WEIGHT[1] * lmloss
            loss = config.TRAIN.LOSS_WEIGHT[1] * lmloss

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            avg_acc, cnt = accuracy_landmark(landmark.detach().cpu().numpy(),
                                             target2.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            avg_acc, cnt = accuracy_classification(classification.detach().cpu().numpy(),
                                                   target.detach().cpu().numpy())
            acc_cls.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 1 == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'\
                      'Accuracy {acc2.val:.3f} ({acc2.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc, acc2=acc_cls)
                logger.info(msg)

                prefix = os.path.join(output_dir, 'result')

                save_images_landmark(meta, landmark.detach().cpu().numpy(), classification.detach().cpu().numpy(), prefix, i)

    return 0


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
