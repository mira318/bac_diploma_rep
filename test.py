import time
import yaml
import sys, os
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torchvision.datasets import VisionDataset

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler


class rvl_cdip_test_dataset(VisionDataset):
    def __init__(self, data_path, transform, target_transform=None):
        # Трансформации изображения надо задавать отдельно.
        # Трансформации результата известны и делаются прямо в классе.
        super().__init__(data_path, transform=transform,
                         target_transform=target_transform)
        self.data_path = data_path
        self.lines = open(data_path + '/labels/test.txt').readlines()
        self.len = len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        name, target = line.split(' ')
        input_file_name = self.data_path + '/images/' + name
        img = Image.open(input_file_name).convert('RGB')  # аналог загрузки через loader
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(int(target)).long()

    def __len__(self):
        return self.len

def count_test(model, loader, loss_fn, args, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()
            if args['channels_last']:
                input = input.contiguous(memory_format=torch.channels_last)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args['tta']
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args['distributed']:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args['local_rank'] == 0 and (last_batch or batch_idx % args['log_interval'] == 0):
                log_name = 'Test' + log_suffix
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics


def main():
    checkpoint_path = sys.argv[1]
    args = yaml.load(open(checkpoint_path + '/args.yaml'))
    args['distributed'] = False

    model = create_model(
        args['model'],
        pretrained=args['pretrained'],
        num_classes=args['num_classes'],
        drop_rate=args['drop'],
        drop_connect_rate=args['drop_connect'],  # DEPRECATED, use drop_path
        drop_path_rate=args['drop_path'],
        drop_block_rate=args['drop_block'],
        global_pool=args['gp'],
        bn_momentum=args['bn_momentum'],
        bn_eps=args['bn_eps'],
        scriptable=args['torchscript'],
        checkpoint_path=args['initial_checkpoint'])
    train_info = torch.load(checkpoint_path + '/model_best.pth.tar')
    model.load_state_dict(train_info['state_dict'])
    model.eval()

    data_config = resolve_data_config(args, model=model, verbose=args['local_rank'] == 0)
    dataset_test = rvl_cdip_test_dataset(args['data_dir'], None)
    loader_test = create_loader(
        dataset_test,
        input_size=data_config['input_size'],
        batch_size=args['batch_size'],
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args['workers'],
        distributed=args['distributed'],
        crop_pct=data_config['crop_pct'],
        pin_memory=args['pin_mem'],
    )

    loss_fn = nn.CrossEntropyLoss().cuda()
    model = model.cuda()
    metrics = count_test(model, loader_test, loss_fn, args)
    print('metrics:', metrics)

if __name__ == '__main__':
    main()
