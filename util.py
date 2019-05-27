import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import time
import shutil
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.image as mpimg

class ProtestDataset(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, img_dir, transform = None):
        """
        Args:
            txt_file: Path to txt file with annotation
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.label_frame)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        image = pil_loader(imgpath)

        protest = self.label_frame.iloc[idx, 1:2].values.astype('float')
        violence = self.label_frame.iloc[idx, 2:3].values.astype('float')
        visattr = self.label_frame.iloc[idx, 3:].values.astype('float')
        label = {'protest':protest, 'violence':violence, 'visattr':visattr}

        sample = {"image":image, "label":label}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

class ProtestDatasetEval(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, img_dir):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                                transforms.Resize(125),
                                transforms.CenterCrop(100),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
        self.img_list = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
        image = pil_loader(imgpath)
        # we need this variable to check if the image is protest or not)
        sample = {"imgpath":imgpath, "image":image}
        sample["image"] = self.transform(sample["image"])
        return sample

class FinalLayer(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.fc = nn.Linear(2048, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def modified_resnet():
    # load pretrained resnet with a modified last fully connected layer
    model = models.resnet50(pretrained = True)
    model.fc = FinalLayer()
    return model

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
        if self.count != 0:
            self.avg = self.sum / self.count

class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/recognition/dataset/minc.py
    """
    
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()            .mul(alpha.view(1, 3).expand(3, 3))            .mul(self.eigval.view(1, 3).expand(3, 3))            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

# for indexing output of the model
protest_idx = Variable(torch.LongTensor([0]))
violence_idx = Variable(torch.LongTensor([1]))
visattr_idx = Variable(torch.LongTensor(range(2,12)))
best_loss = float("inf")

def calculate_loss(output, target, criterions, weights = [1, 10, 5]):
    """Calculate loss"""
    # number of protest images
    N_protest = int(target['protest'].data.sum())
    batch_size = len(target['protest'])

    if N_protest == 0:
        # if no protest image in target
        outputs = [None]
        # protest output
        outputs[0] = output.index_select(1, protest_idx)
        targets = [None]
        # protest target
        targets[0] = target['protest'].float()
        losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(1)]
        scores = {}
        scores['protest_acc'] = accuracy_score((outputs[0]).data.round(), targets[0].data)
        scores['violence_mse'] = 0
        scores['visattr_acc'] = 0
        return losses, scores, N_protest

    # used for filling 0 for non-protest images
    not_protest_mask = (1 - target['protest']).byte()

    outputs = [None] * 4
    # protest output
    outputs[0] = output.index_select(1, protest_idx)
    # violence output
    outputs[1] = output.index_select(1, violence_idx)
    outputs[1].masked_fill_(not_protest_mask, 0)
    # visual attribute output
    outputs[2] = output.index_select(1, visattr_idx)
    outputs[2].masked_fill_(not_protest_mask.repeat(1, 10),0)


    targets = [None] * 4

    targets[0] = target['protest'].float()
    targets[1] = target['violence'].float()
    targets[2] = target['visattr'].float()

    scores = {}
    # protest accuracy for this batch
    scores['protest_acc'] = accuracy_score(outputs[0].data.round(), targets[0].data)
    # violence MSE for this batch
    scores['violence_mse'] = ((outputs[1].data - targets[1].data).pow(2)).sum() / float(N_protest)
    # mean accuracy for visual attribute for this batch
    comparison = (outputs[2].data.round() == targets[2].data)
    comparison.masked_fill_(not_protest_mask.repeat(1, 10).data,0)
    n_right = comparison.float().sum()
    mean_acc = n_right / float(N_protest*10)
    scores['visattr_acc'] = mean_acc

    # return weighted loss
    losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(len(criterions))]

    return losses, scores, N_protest



def train(train_loader, model, criterions, optimizer, epoch):
    """training the model"""

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_protest = AverageMeter()
    loss_v = AverageMeter()
    protest_acc = AverageMeter()
    violence_mse = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(train_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        input_var = Variable(input)
        output = model(input_var)

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)

        optimizer.zero_grad()
        loss = 0
        for l in losses:
            loss += l
        # back prop
        loss.backward()
        optimizer.step()
        if N_protest:
            loss_protest.update(losses[0].data, input.size(0))
            loss_v.update(loss.data - losses[0].data, N_protest)
        else:
            # when there is no protest image in the batch
            loss_protest.update(losses[0].data, input.size(0))
        loss_history.append(loss.data)
        protest_acc.update(scores['protest_acc'], input.size(0))
        violence_mse.update(scores['violence_mse'], N_protest)
        visattr_acc.update(scores['visattr_acc'], N_protest)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'Protest {protest_acc.val:.3f} ({protest_acc.avg:.3f})  '
                  'Violence {violence_mse.val:.5f} ({violence_mse.avg:.5f})  '
                  'Vis Attr {visattr_acc.val:.3f} ({visattr_acc.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   loss_val=loss_protest.val + loss_v.val,
                   loss_avg = loss_protest.avg + loss_v.avg,
                   protest_acc = protest_acc, violence_mse = violence_mse,
                   visattr_acc = visattr_acc))

    return loss_history

def validate(val_loader, model, criterions, epoch):
    """Validating"""
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_protest = AverageMeter()
    loss_v = AverageMeter()
    protest_acc = AverageMeter()
    violence_mse = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(val_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        input_var = Variable(input)

        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        output = model(input_var)

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)
        loss = 0
        for l in losses:
            loss += l

        if N_protest:
            loss_protest.update(losses[0].data, input.size(0))
            loss_v.update(loss.data - losses[0].data, N_protest)
        else:
            # when no protest images
            loss_protest.update(losses[0].data, input.size(0))
        loss_history.append(loss.data)
        protest_acc.update(scores['protest_acc'], input.size(0))
        violence_mse.update(scores['violence_mse'], N_protest)
        visattr_acc.update(scores['visattr_acc'], N_protest)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'Protest Acc {protest_acc.val:.3f} ({protest_acc.avg:.3f})  '
                  'Violence MSE {violence_mse.val:.5f} ({violence_mse.avg:.5f})  '
                  'Vis Attr Acc {visattr_acc.val:.3f} ({visattr_acc.avg:.3f})'
                  .format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   loss_val =loss_protest.val + loss_v.val,
                   loss_avg = loss_protest.avg + loss_v.avg,
                   protest_acc = protest_acc,
                   violence_mse = violence_mse, visattr_acc = visattr_acc))

    print(' * Loss {loss_avg:.3f} Protest Acc {protest_acc.avg:.3f} '
          'Violence MSE {violence_mse.avg:.5f} '
          'Vis Attr Acc {visattr_acc.avg:.3f} '
          .format(loss_avg = loss_protest.avg + loss_v.avg,
                  protest_acc = protest_acc,
                  violence_mse = violence_mse, visattr_acc = visattr_acc))
    return loss_protest.avg + loss_v.avg, loss_history

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every 5 epochs"""
    lr = args.lr * (0.4 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoints"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    global best_loss
    loss_history_train = []
    loss_history_val = []
    data_dir = args.data_dir
    img_dir_train = os.path.join(data_dir, "train")
    img_dir_val = os.path.join(data_dir, "test")
    txt_file_train = os.path.join(data_dir, "annot_train.txt")
    txt_file_val = os.path.join(data_dir, "annot_test.txt")

    # load pretrained resnet50 with a modified last fully connected layer
    model = modified_resnet()

    # we need three different criterion for training
    criterion_protest = nn.BCELoss()
    criterion_violence = nn.MSELoss()
    criterion_visattr = nn.BCELoss()
    criterions = [criterion_protest, criterion_violence, criterion_visattr]

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU Found")
    if args.cuda:
        model = model.cuda()
        criterions = [criterion.cuda() for criterion in criterions]
    # we are not training the frozen layers
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
                        parameters, args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay
                        )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            loss_history_train = checkpoint['loss_history_train']
            loss_history_val = checkpoint['loss_history_val']
            if args.change_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948,  0.4203]])

    train_dataset = ProtestDataset(
                        txt_file = txt_file_train,
                        img_dir = img_dir_train,
                        transform = transforms.Compose([
                                transforms.RandomResizedCrop(100),
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(
                                    brightness = 0.4,
                                    contrast = 0.4,
                                    saturation = 0.4,
                                    ),
                                transforms.ToTensor(),
                                Lighting(0.1, eigval, eigvec),
                                normalize,
                        ]))
    val_dataset = ProtestDataset(
                    txt_file = txt_file_val,
                    img_dir = img_dir_val,
                    transform = transforms.Compose([
                        transforms.Resize(125),
                        transforms.CenterCrop(100),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    train_loader = DataLoader(
                    train_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size,
                    shuffle = True
                    )
    val_loader = DataLoader(
                    val_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        loss_history_train_this = train(train_loader, model, criterions,
                                        optimizer, epoch)
        loss_val, loss_history_val_this = validate(val_loader, model,
                                                   criterions, epoch)
        loss_history_train.append(loss_history_train_this)
        loss_history_val.append(loss_history_val_this)

        # loss = loss_val.avg

        is_best = loss_val < best_loss
        if is_best:
            print('best model!!')
        best_loss = min(loss_val, best_loss)


        save_checkpoint({
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict(),
            'best_loss' : best_loss,
            'optimizer' : optimizer.state_dict(),
            'loss_history_train': loss_history_train,
            'loss_history_val': loss_history_val
        }, is_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "",
                        help = "directory path to dataset",
                        )
    parser.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 8,
                        help = "batch size",
                        )
    parser.add_argument("--epochs",
                        type = int,
                        default = 10,
                        help = "number of epochs",
                        )
    parser.add_argument("--weight_decay",
                        type = float,
                        default = 1e-4,
                        help = "weight decay",
                        )
    parser.add_argument("--lr",
                        type = float,
                        default = 0.01,
                        help = "learning rate",
                        )
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "momentum",
                        )
    parser.add_argument("--print_freq",
                        type = int,
                        default = 10,
                        help = "print frequency",
                        )
    parser.add_argument('--resume',
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--change_lr',
                        action = "store_true",
                        help = "Use this if you want to \
                        change learning rate when resuming")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    args, unknown = parser.parse_known_args()

    if args.cuda:
        protest_idx = protest_idx.cuda()
        violence_idx = violence_idx.cuda()
        visattr_idx = visattr_idx.cuda()


    main()

