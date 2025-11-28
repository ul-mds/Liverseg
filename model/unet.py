from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from collections import OrderedDict

from utilities.metrics import dice_coef, iou_score
import utilities.losses as losses
from utilities.utils import str2bool

import numpy as np

from tqdm import tqdm


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self):
        super(UNet, self).__init__()
        in_ch = 3
        out_ch = 2
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out

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
        self.avg = self.sum / self.count


def train(net, trainloader, validloader, optimizer, criterion, device, epoch):
    #losses = AverageMeter()
    #ious = AverageMeter()
    #dices_livers = AverageMeter()
    #dices_tumors = AverageMeter()
    final_loss = np.inf
    final_ious = 0.0
    final_dice_liver = 0.0
    final_dice_tumour = 0.0
        
    losses = AverageMeter()
    ious = AverageMeter()
    dices_livers = AverageMeter()
    dices_tumours = AverageMeter()
    net.train()
    for i, (input, target) in tqdm(enumerate(trainloader), total=len(trainloader), disable=True):
        input = input.to(device)
        target = target.to(device)
        net.to(device)
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice_liver = dice_coef(output, target)[0]
        dice_tumour = dice_coef(output, target)[1]

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices_livers.update(torch.tensor(dice_liver), input.size(0))
        dices_tumours.update(torch.tensor(dice_tumour), input.size(0))

        # compute gradient and do optimizing step
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val_loss, val_log = validate(net, validloader, criterion, device)

    if val_loss < final_loss:
        print("epoch: ", epoch)
        print("train ious: ", ious.avg)
        print("train dice liver: ", dices_livers.avg)
        print("train dice tumour: ", dices_tumours.avg)
        #torch.save(net.state_dict(), 'model.pth')
        print("validation ious: ", val_log['iou'])
        print("validation dice liver: ", val_log['dice_liver'])
        print("validation dice tumour: ", val_log['dice_tumour'])
        final_loss = val_loss
        print("=> new best model: ",epoch)
        
    torch.cuda.empty_cache()

    log = OrderedDict([
        ('iou', ious.avg),
        ('dice_liver', dices_livers.avg),
        ('dice_tumor', dices_tumours.avg),
        ('val_iou', val_log['iou']),
        ('val_dice_liver', val_log['dice_liver']),
        ('val_dice_tumour', val_log['dice_tumour'])
    ])
    
    #net.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))
    #print("fim do train")

    return losses.avg, log

def validate(model, val_loader, criterion, device):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_livers = AverageMeter()
    dices_tumours = AverageMeter()

    # switch to evaluate mode
    model.eval()
    print("Entrei no validation 1 3")

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader), disable=True):
            input = input.to(device)
            target = target.to(device)
            model.to(device)
            # compute output
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice_liver = dice_coef(output, target)[0]
            dice_tumour = dice_coef(output, target)[1]
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            
            dices_livers.update(torch.tensor(dice_liver), input.size(0))
            dices_tumours.update(torch.tensor(dice_tumour), input.size(0))
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_liver', dices_livers.avg),
        ('dice_tumour', dices_tumours.avg)
    ])
    
    return losses.avg, log


def test(net, testloaders, criterion, device):
    
    test_individual_losses = []
    test_individual_ious = []
    test_individual_dices_livers = []
    test_individual_dices_tumors = []
    print("Entrei no test")

    for index in range(len(testloaders)):
        test_individual_losses.append(AverageMeter())
        test_individual_ious.append(AverageMeter())
        test_individual_dices_livers.append(AverageMeter())
        test_individual_dices_tumors.append(AverageMeter())
    # switch to evaluate mode
    net.eval()

    with torch.no_grad():
        for i,testloader in enumerate(testloaders):
            
            for j, (input, target) in tqdm(enumerate(testloader), total=len(testloader),disable=True):
                input = input.to(device)
                target = target.to(device)
                net.to(device)
                
                output = net(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_liver = dice_coef(output, target)[0]
                dice_tumor = dice_coef(output, target)[1]

                test_individual_losses[i].update(loss.item(), input.size(0))
                test_individual_ious[i].update(iou, input.size(0))
                test_individual_dices_livers[i].update(torch.tensor(dice_liver), input.size(0))
                test_individual_dices_tumors[i].update(torch.tensor(dice_tumor), input.size(0))
        print("Meio do test")

        log = OrderedDict()
        #individual test sets evaluation
        for index in range(len(test_individual_ious)-1):
            
            log['ind_iou_'+str(index)] = test_individual_ious[index].avg
            log['ind_dice_liver_'+str(index)] = test_individual_dices_livers[index].avg
            log['ind_dice_tumour_'+str(index)] = test_individual_dices_tumors[index].avg

        #total test sets evaluation
        log['all_iou'] = test_individual_ious[-1].avg
        log['all_dice_liver'] = test_individual_dices_livers[-1].avg
        log['all_dice_tumour'] = test_individual_dices_tumors[-1].avg
        print("Fim do test")

    return test_individual_losses[-1].avg, log

def model_to_parameters(model):
    from flwr.common.parameter import ndarrays_to_parameters
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Extracted model parameters!")
    return parameters
