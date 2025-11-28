import torch.nn as nn
import torch
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

from utilities.metrics import dice_coef, iou_score
#from model.pytorch.init_weights import init_weights
from model.layers import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


class ResUnetPlusPlus(nn.Module):
    def __init__(self):
        super(ResUnetPlusPlus, self).__init__()
        in_channels = 3
        n_classes = 2
        filters=[16, 32, 64, 128, 256]

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

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


#def test(net, testloader, criterion, device):
#    losses = AverageMeter()
#    ious = AverageMeter()
#    dices_livers = AverageMeter()
#    dices_tumors = AverageMeter()

#    # switch to evaluate mode
#    net.eval()

#    with torch.no_grad():
#        for i, (input, target) in tqdm(enumerate(testloader), total=len(testloader)):
#            input = input.to(device)
#            target = target.to(device)

            # compute output
            
#            output = net(input)
#            loss = criterion(output, target)
#            iou = iou_score(output, target)
#            dice_liver = dice_coef(output, target)[0]
#            dice_tumor = dice_coef(output, target)[1]

#            losses.update(loss.item(), input.size(0))
#            ious.update(iou, input.size(0))
#            dices_livers.update(torch.tensor(dice_liver), input.size(0))
#            dices_tumors.update(torch.tensor(dice_tumor), input.size(0))

#    log = OrderedDict([
#        ('loss', losses.avg),
#        ('iou', ious.avg),
#        ('dice_liver', dices_livers.avg),
#        ('dice_tumor', dices_tumors.avg)
#    ])

#    return losses.avg, log

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
