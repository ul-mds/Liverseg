import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict
from tqdm import tqdm

from utilities.metrics import dice_coef, iou_score

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, expand):
        super(InvertedResidual, self).__init__()
        self.expand=expand
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, 1, 0, dilation=expand, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, 1, bias=False),
        )

    def forward(self, x):
        x_pad = fixed_padding(x, 3, self.expand)
        y= self.conv(x_pad)
        return y

class block_down(nn.Module):

    def __init__(self, inp_channel, out_channel, expand):
        super(block_down, self).__init__()
        self.deepwise1 = InvertedResidual(inp_channel, inp_channel, expand)
        self.deepwise2 = InvertedResidual(inp_channel, out_channel, expand)
        self.resnet= nn.Conv2d(inp_channel, out_channel, 1, 1, 0, 1, bias=False)

    def forward(self, input):
        resnet=self.resnet(input)
        x = self.deepwise1(input)
        x= self.deepwise2(x)
        out=torch.add(resnet,x)
        return out


class block_up(nn.Module):

    def __init__(self, inp_channel, out_channel, expand):
        super(block_up, self).__init__()
        self.up = nn.ConvTranspose2d(inp_channel, out_channel, 2, stride=2)
        self.deepwise1 = InvertedResidual(inp_channel, inp_channel, expand)
        self.deepwise2 = InvertedResidual(inp_channel, out_channel, expand)
        self.resnet = nn.Conv2d(inp_channel, out_channel, 1, 1, 0, 1, bias=False)

    def forward(self, x, y):
        x = self.up(x)
        x1 = torch.cat([x, y], dim=1)
        x = self.deepwise1(x1)
        x = self.deepwise2(x)
        resnet=self.resnet(x1)
        out=torch.add(resnet,x)

        return out


class U_net(nn.Module):

    def __init__(self):
        super(U_net, self).__init__()
        class_num = 2
        self.inp = nn.Conv2d(3, 64, 1)
        self.block2 = block_down(64, 128, expand=1)
        self.block3 = block_down(128, 256, expand=2)
        self.block4 = block_down(256, 512, expand=2)
        self.block5 = block_down(512, 1024, expand=1)
        self.block6 = block_up(1024, 512, expand=1)
        self.block7 = block_up(512, 256, expand=1)
        self.block8 = block_up(256, 128, expand=2)
        self.block9 = block_up(128, 64, expand=2)
        self.out = nn.Conv2d(64, class_num, 1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x1_use = self.inp(x)
        x1 = self.maxpool(x1_use)
        x2_use = self.block2(x1)
        x2 = self.maxpool(x2_use)
        x3_use = self.block3(x2)
        x3 = self.maxpool(x3_use)
        x4_use = self.block4(x3)
        x4 = self.maxpool(x4_use)
        x5 = self.block5(x4)

        x6 = self.block6(x5, x4_use)
        x7 = self.block7(x6, x3_use)
        x8 = self.block8(x7, x2_use)
        x9 = self.block9(x8, x1_use)
        out= self.out(x9)
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

def train(net, trainloader, optimizer, criterion, device):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_livers = AverageMeter()
    dices_tumors = AverageMeter()
    net.train()
    for i, (input, target) in tqdm(enumerate(trainloader), total=len(trainloader)):
        input = input.to(device)
        target = target.to(device)
        output = net(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice_liver = dice_coef(output, target)[0]
        dice_tumour = dice_coef(output, target)[1]

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices_livers.update(torch.tensor(dice_liver), input.size(0))
        dices_tumors.update(torch.tensor(dice_tumour), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('iou', ious.avg),
        ('dice_liver', dices_livers.avg),
        ('dice_tumor', dices_tumors.avg)
    ])

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
