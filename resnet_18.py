# -*- coding: utf-8 -*-

import h5py
import math
import copy
import scipy.io as io
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.nn.functional as F
from scipy.io import loadmat
from torch.autograd import Function
from torch.nn.parameter import Parameter

class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.clone().detach()
    def close(self):
        self.hook.remove()

class Divclass:
    def __init__(self, depthList, posList):
        self.depthList = depthList
        self.posList = posList


def getMu(x):
    IsuseMax = 0
    bs = x.size()[0]
    depth = x.size()[1]
    h = x.size()[2]
    w = x.size()[3]
    x = x.transpose(2, 3).reshape([bs, depth, h * w])
    if (IsuseMax):
        _, p = torch.max(x, dim=2)
        p = torch.reshape(p, (bs, depth, 1)).float()  # index is long type
        mu_y = torch.ceil(p / h)
        mu_x = p - (mu_y - 1) * h
        sqrtvar = torch.Tensor([])
    else:
        tmp_x = torch.Tensor(range(1, h + 1)).reshape(-1, 1).repeat([bs, depth, w, 1]).squeeze(3).cuda()
        tmp_y = torch.Tensor(range(1, w + 1)).repeat([bs, depth, h, 1])
        tmp_y = tmp_y.transpose(2, 3).reshape([bs, depth, h * w]).cuda()
        sumXtmp = torch.sum(x, 2).unsqueeze(2)
        sumX = torch.max(sumXtmp, (torch.ones(sumXtmp.size()).cuda() * 0.000000001))
        mu_xtmp = torch.round((torch.sum(tmp_x.mul(x), 2).unsqueeze(2)).div(sumX))
        mu_x = torch.max(mu_xtmp, torch.ones(mu_xtmp.size()).cuda())
        mu_ytmp = torch.round((torch.sum(tmp_y.mul(x), 2).unsqueeze(2)).div(sumX))
        mu_y = torch.max(mu_ytmp, torch.ones(mu_ytmp.size()).cuda())
        sqrtvartmp1 = mu_x.repeat([1, 1, h * w])
        sqrtvartmp2 = mu_y.repeat([1, 1, h * w])
        sqrtvar = torch.sqrt((torch.sum((tmp_x - sqrtvartmp1).mul(tmp_x - sqrtvartmp1).mul(x), 2).unsqueeze(2) + torch.sum((tmp_y - sqrtvartmp2).mul(tmp_y - sqrtvartmp2).mul(x), 2).unsqueeze(2)).div(sumX))
        p = (mu_x + (mu_y - 1) * h).reshape([bs, depth, 1, 1])
    tmp = torch.linspace(-1, 1, h).repeat(mu_x.size()).cuda()
    for i in range(bs):
        mu_x[i, :, :] = torch.gather(tmp[i, :, :], 1, (mu_x[i, :, :] - 1).long())
        mu_y[i, :, :] = torch.gather(tmp[i, :, :], 1, (mu_y[i, :, :] - 1).long())
    mu_x = mu_x.reshape([bs, depth, 1, 1])
    mu_y = mu_y.reshape([bs, depth, 1, 1])
    sqrtvar = sqrtvar.reshape([bs, depth])
    return mu_x, mu_y, sqrtvar


def getMask(mask_parameter, mask_weight, posTempX, posTempY, bs, depth, h, w):
    mask = torch.abs(posTempX - mask_parameter['mu_x'].repeat([1, 1, h, w]))
    mask = mask + torch.abs(posTempY - mask_parameter['mu_y'].repeat([1, 1, h, w]))
    mask = 1 - mask.mul(mask_weight.reshape(depth, 1, 1).repeat([bs, 1, h, w]))
    #mask = torch.max(mask, torch.ones(mask.size()).cuda() * (-1))
    mask = torch.ones(mask.size()).cuda() * (-1)
    for i in range(depth):
        if not (mask_parameter['filter'][i].equal(torch.ones(1))):
            mask[:, i, :, :] = 1
    return mask


def get_sliceMag(sliceMag,label,x):

    for lab in range(label.shape[1]):
        index = (label[:, lab, :, :] == 1).reshape(label.shape[0])
        if torch.sum(index) != 0:
            (tmp, idx) = torch.max(x[index, :, :, :], dim=2)
            (tmp, idx) = torch.max(tmp, dim=2)
            tmp = tmp.reshape(tmp.size()[0], tmp.size()[1], 1, 1)
            meantmp = torch.mean(tmp, 0)
            if (torch.sum(sliceMag[:, lab]) == 0):
                sliceMag[:, lab] = torch.max(meantmp,(torch.ones(meantmp.size()) * 0.1).cuda()).reshape(meantmp.size()[0])
            else:
                tmptmp = 0.9
                index = (meantmp == 0).reshape(meantmp.size()[0])
                meantmp[index, 0, 0] = sliceMag[index, 0].cuda()
                sliceMag[:, lab] = (sliceMag[:,lab] * tmptmp).cuda()+meantmp.reshape(meantmp.size()[0])*(1-tmptmp)
    return sliceMag


class conv_mask_F(Function):
    @staticmethod
    def forward(self, x, weight, bias, mask_weight, padding, label, Iter, density, mask_parameter):
        bs = x.shape[0]
        depth = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        posTemp_x = torch.linspace(-1, 1, h).reshape(-1, 1).repeat([depth, 1, w])
        posTemp_y = torch.linspace(-1, 1, w).repeat([depth, h, 1])
        posTempX = posTemp_x.repeat([bs, 1, 1, 1]).cuda()
        posTempY = posTemp_y.repeat([bs, 1, 1, 1]).cuda()
        mask_parameter['mu_x'], mask_parameter['mu_y'], mask_parameter['sqrtvar'] = getMu(x)
        mask = getMask(mask_parameter, mask_weight, posTempX, posTempY, bs, depth, h, w)
        input = x.mul(mask)
        x_relu = torch.max(input, torch.zeros(input.size()).cuda())

        parameter_sliceMag = mask_parameter['sliceMag'].clone().data
        mask_parameter['sliceMag'] = get_sliceMag(mask_parameter['sliceMag'],label,x)

        self.save_for_backward(x, weight, bias, mask_weight, torch.Tensor([padding]), label, mask, Iter, density,
                               mask_parameter['filter'], mask_parameter['mag'], mask_parameter['sqrtvar'], mask_parameter['strength'],parameter_sliceMag)

        return F.conv2d(x_relu, weight, bias, padding=padding)

    @staticmethod
    def backward(self, grad_output):
        x, weight, bias, mask_weight, padding, label, mask, Iter, density, parameter_filter, parameter_mag, parameter_sqrtvar, parameter_strength, parameter_sliceMag = self.saved_tensors

        input = x.mul(torch.max(mask, torch.zeros(mask.size()).cuda()))
        if self.needs_input_grad[0]:
            x_grad = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, padding=int(padding.item()))
        if self.needs_input_grad[1]:
            weight_grad = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, padding=int(padding.item()))
        if bias is not None and self.needs_input_grad[2]:
            bias_grad = grad_output.sum(0).sum((1, 2))

        depth = x.size()[1]
        h = x.size()[2]
        w = x.size()[3]
        depthList = (parameter_filter > 0).nonzero()[:, 0].reshape(-1, 1)
        labelNum = label.size()[1]
        Div_list = []

        if (labelNum == 1):
            theClass = label
            posList = (theClass == 1).nonzero()[:, 0].reshape(-1, 1)
            Div = Divclass(depthList, posList)
            Div_list.append(Div)
        else:
            (theClass, indextmp) = torch.max(label, dim=1)
            theClass = theClass.unsqueeze(2)
            if (parameter_sliceMag.size()[0] == torch.Tensor([]).size()[0]):
                posList = (theClass == 1).nonzero()[:, 0].reshape(-1, 1)
                Div = Divclass(depthList, posList)
                Div_list.append(Div)
            else:
                sliceM = parameter_sliceMag
                for i in range(labelNum):
                    Div = Divclass(depthList=torch.Tensor([]), posList=torch.Tensor([]))
                    Div_list.append(Div)
                (val, index) = torch.max(sliceM[depthList, :].squeeze(1), dim=1)
                for lab in range(labelNum):
                    (Div_list[lab].depthList, indextmp) = torch.sort(depthList[index == lab], dim=0)
                    Div_list[lab].posList = (label[:, lab, :, :] == 1).nonzero()[:, 0].reshape(-1, 1)

        imgNum = label.size()[0]
        alpha = 0.5
        x_grad = x_grad.mul(torch.max(mask, torch.zeros(mask.size()).cuda()))

        if ((torch.sum(parameter_filter == 1)) > 0):
            parameter_strength = torch.mean(torch.mean(x.mul(mask), 2), 2).transpose(1, 0).cuda()
            mask_tmp = (torch.from_numpy(copy.deepcopy(mask.cpu().detach().numpy()[::-1, ::-1, :, :]))).cuda()
            alpha_logZ_pos = (torch.log(torch.mean(torch.exp(torch.mean(torch.mean(x.mul(mask_tmp), 2), 2).div(alpha)), 0)) * alpha).reshape(depth, 1)
            alpha_logZ_neg = (torch.log(torch.mean(torch.exp(torch.mean(torch.mean(-x, 2), 2).div(alpha)), 0)) * alpha).reshape(depth, 1)

            # restrict
            #alpha_logZ_pos[alpha_logZ_pos > 10000.] = torch.tensor(10000.).cuda()
            #alpha_logZ_pos[alpha_logZ_pos < -10000.] = torch.tensor(-10000.).cuda()

            #alpha_logZ_neg[alpha_logZ_neg > 10000.] = torch.tensor(10000.).cuda()
            #alpha_logZ_neg[alpha_logZ_neg < -10000.] = torch.tensor(-10000.).cuda()

            alpha_logZ_pos[torch.isinf(alpha_logZ_pos)] = torch.max(alpha_logZ_pos[torch.isinf(alpha_logZ_pos) == 0])
            alpha_logZ_neg[torch.isinf(alpha_logZ_neg)] = torch.max(alpha_logZ_neg[torch.isinf(alpha_logZ_neg) == 0])

        for lab in range(len(Div_list)):
            if (labelNum == 1):
                w_pos = 1
                w_neg = 1
            else:
                if (labelNum > 10):
                    w_pos = 0.5 / (1 / labelNum)
                    w_neg = 0.5 / (1 - 1 / labelNum)
                else:
                    w_pos = 0.5 / density[lab]
                    w_neg = 0.5 / (1 - density[lab])

            mag = torch.ones([depth, imgNum]).div(1 / Iter).div(parameter_mag).cuda()
            dList = Div_list[lab].depthList
            dList = dList[(parameter_filter[dList] == 1).squeeze(1)].reshape(-1, 1)
            if (dList.size()[0] != torch.Tensor([]).size()[0]):
                List = Div_list[lab].posList.cuda()
                if (List.size()[0] != torch.Tensor([]).size()[0]):
                    strength = torch.exp((parameter_strength[:, List].squeeze(2))[dList, :].squeeze(1).div(alpha)).mul((parameter_strength[:, List].squeeze(2))[dList, :].squeeze(1) - alpha_logZ_pos[dList].squeeze(1).repeat(1, List.size()[0]) + alpha)
                    # restrict
                    #strength[strength > 10000.] = torch.tensor(10000.).cuda()
                    #strength[strength < -10000.] = torch.tensor(-10000.).cuda()
                    strength[torch.isinf(strength)] = torch.max(strength[torch.isinf(strength) == 0])
                    strength[torch.isnan(strength)] = 0
                    strength = (strength.div((torch.mean(strength, 1).reshape(-1, 1).repeat(1, List.size()[0])).mul((mag[:, List].squeeze(2))[dList, :].squeeze(1)))).transpose(0, 1).reshape(List.size()[0],dList.size()[0], 1, 1)
                    strength[torch.isnan(strength)] = 0
                    # restrict
                    #strength[strength > 10000.] = torch.tensor(10000.).cuda()
                    #strength[strength < -10000.] = torch.tensor(-10000.).cuda()
                    strength[torch.isinf(strength)] = torch.max(strength[torch.isinf(strength) == 0])
                    index_dList = dList.repeat(List.size()[0], 1)
                    index_List = List.reshape(-1, 1).repeat(1, dList.size()[0]).reshape(List.size()[0] * dList.size()[0], 1)
                    x_grad[index_List, index_dList, :, :] = ((x_grad[List, :, :, :].squeeze(1))[:, dList, :, :].squeeze(2) - (mask[List, :, :,:].squeeze(1))[:,dList,:,:].squeeze(2).mul(strength.repeat(1, 1, h, w) * (0.00001 * w_pos))).reshape(List.size()[0] * dList.size()[0],1, h, w)

                list_neg = (label != 1).nonzero()[:, 0].reshape(-1, 1)
                if (list_neg.size()[0] != torch.Tensor([]).size()[0]):
                    strength = torch.mean((torch.mean((x[list_neg, :, :, :].squeeze(1))[:, dList, :, :].squeeze(2), 2).unsqueeze(2)),3).unsqueeze(2).transpose(0, 1).reshape(dList.size()[0], list_neg.size()[0])
                    strength = torch.exp(-strength.div(alpha)).mul(-strength - alpha_logZ_neg[dList].squeeze(2).repeat(1, list_neg.size()[0]) + alpha)
                    # restrict
                    #strength[strength > 10000.] = torch.tensor(10000.).cuda()
                    #strength[strength < -10000.] = torch.tensor(-10000.).cuda()
                    strength[torch.isinf(strength)] = torch.max(strength[torch.isinf(strength) == 0])
                    strength[torch.isnan(strength)] = 0
                    strength = (strength.div((torch.mean(strength, 1).reshape(-1, 1).repeat(1, list_neg.size()[0])).mul((mag[:, list_neg].squeeze(2))[dList, :].squeeze(1)))).transpose(0, 1).reshape(list_neg.size()[0], dList.size()[0], 1, 1)
                    strength[torch.isnan(strength)] = 0
                    # restrict
                    #strength[strength > 10000.] = torch.tensor(10000.).cuda()
                    #strength[strength < -10000.] = torch.tensor(-10000.).cuda()
                    strength[torch.isinf(strength)] = torch.max(strength[torch.isinf(strength) == 0])
                    index_dList = dList.repeat(list_neg.size()[0], 1)
                    index_list_neg = list_neg.reshape(-1, 1).repeat(1, dList.size()[0]).reshape(list_neg.size()[0] * dList.size()[0], 1)
                    x_grad[index_list_neg, index_dList, :, :] = ((x_grad[list_neg, :, :, :].squeeze(1))[:, dList, :, :].squeeze(2) + (strength.reshape(list_neg.size()[0], dList.size()[0], 1, 1).repeat(1, 1, h, w)) * (0.00001 * w_neg)).reshape(list_neg.size()[0] * dList.size()[0], 1, h, w)

        beta = 3.0
        mask_weight_grad = torch.zeros(depth, 1).cuda()
        parameter_sqrtvar = parameter_sqrtvar.transpose(0, 1)

        for lab in range(len(Div_list)):
            dList = Div_list[lab].depthList.cuda()
            List = Div_list[lab].posList
            if ((dList.size()[0] != torch.Tensor([]).size()[0]) and (List.size()[0] != torch.Tensor([]).size()[0])):
                tmp = ((torch.sum((parameter_strength[:, List].squeeze(2))[dList, :].squeeze(1).mul((parameter_sqrtvar[:, List].squeeze(2))[dList, :].squeeze(1)), 1)).
                       div(torch.sum((parameter_strength[:, List].squeeze(2))[dList, :].squeeze(1), 1))).reshape(-1, 1)
                tmptmp = beta / tmp
                tmp = torch.max(torch.min(tmptmp, torch.ones(tmptmp.size()).cuda() * 3),torch.ones(tmptmp.size()).cuda() * 1.5)
                tmp = (tmp - mask_weight[dList].squeeze(2)) * (-10000)
                mask_weight_grad[dList] = tmp.unsqueeze(2)

        return x_grad, weight_grad, bias_grad, mask_weight_grad, None, None, None, None, None, None, None, None, None, None, None


class conv_mask(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, labelnum, loss_type, ):
        super(conv_mask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding[0]
        self.alphainit = 2.0
        self.mask_parameter = None
        self.label_num = labelnum
        self.losstype = loss_type

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.mask_weight = Parameter(torch.ones(in_channels, 1) * self.alphainit)
        self.bias = Parameter(torch.zeros(out_channels))
        self.init_mask_parameter()


    def forward(self, x, label, Iter, density):
        out = conv_mask_F.apply(x, self.weight, self.bias, self.mask_weight, self.padding, label, Iter, density, self.mask_parameter)
        return out

    def init_mag(self):
        mag = torch.Tensor([0.1])
        # mag need to be modified for multiple classifications
        if self.losstype == 'softmax':
            if self.label_num > 10:
                mag = mag / 50
                if self.model == 'vgg_m':
                    mag = mag / 1000000
            else:
                mag = mag * 0.2
        return mag

    def init_mask_parameter(self):
        mag = self.init_mag()
        partRate = 1
        textureRate = 0
        partNum = round(partRate * self.in_channels)
        textureNum = round((textureRate + partRate) * self.in_channels) - partNum
        filtertype = torch.zeros(self.in_channels, 1)
        filtertype[0:partNum] = 1
        filtertype[partNum:partNum + textureNum] = 2
        sliceMag = torch.zeros(self.in_channels, self.label_num)
        self.mask_parameter = {'posTemp': {'posTemp_x': None, 'posTemp_y': None},
                        'mu_x': None,
                        'mu_y': None,
                        'sqrtvar': None,
                        'strength': None,
                        'sliceMag': sliceMag,
                        'filter': filtertype,
                        'mag': mag}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)#new padding

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = nn.ZeroPad2d(1)#new paddig

    def forward(self, x):
        identity = x
        out = self.pad2d(x) #new padding
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out) #new padding
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class resnet_18(resnet.ResNet):
    def __init__(self, pretrain_path,num_classes,dropout_rate,losstype,block=BasicBlock,  layers=[2,2,2,2], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet_18, self).__init__(resnet.BasicBlock, [2, 2, 2, 2])
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.label_num = num_classes
        self.pretrian_path = pretrain_path
        self._norm_layer = norm_layer
        self.losstype = losstype
        self.inplanes = 64
        self.dilation = 1
        self.linear = nn.Linear(2048, 16)
        self.bn = nn.BatchNorm1d(16, momentum=0.01)
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0,
                               bias=False)#new padding
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)#new padding
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.mask1 = nn.Sequential(
            conv_mask(256* block.expansion, 256* block.expansion, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), labelnum=self.label_num,
                      loss_type=self.losstype, ), )

        self.mask2 = nn.Sequential(
            conv_mask(256 * block.expansion, 256 * block.expansion, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      labelnum=self.label_num,
                      loss_type=self.losstype, ), )

        self.avgpool = nn.Sequential(
            #nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Linear(512* block.expansion, num_classes)

        self.pad2d_1 = nn.ZeroPad2d(1)#new paddig
        self.pad2d_3 = nn.ZeroPad2d(3)#new paddig

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                    nn.init.constant_(m.bn2.weight, 0)

        self.init_weight()
        self.activation = SaveFeatures(list(self.children())[-1])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def init_weight(self):
        state_dict = torch.load(self.pretrian_path)
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           'fc' not in k and 'layer4.2' not in k}  # 'fc' not in k and 'layer4.1' not in k and
        model_dict = self.state_dict()
        #for k in model_dict:
        #    print(k)
        # print('####################################')
        # print('####################################')
        #for k,v in state_dict.items():
        #    print(k)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict,strict=False)
        torch.nn.init.normal_(self.mask1[0].weight.data, mean=0, std=0.01)
        torch.nn.init.normal_(self.mask2[0].weight.data, mean=0, std=0.01)

        torch.nn.init.normal_(self.fc.weight.data, mean=0, std=0.01)
        torch.nn.init.zeros_(self.fc.bias.data)


    def forward(self, x, label, Iter, density):
        # See note [TorchScript super()]


        x = self.pad2d_3(x) #new padding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2d_1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.mask1[0](x, label, Iter, density)
        x = self.relu1(x)
        x = self.mask2[0](x, label, Iter, density)
        x = self.relu2(x)
        # f_map = x.detach()
        x = self.layer4(x)

        # f_map = x.detach()
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)


        return x
    # def forward(self, x):
    #     return self._forward_impl(x), None

    def __call__(self, inputs, label, Iter, density):
        super().__call__(inputs, label, Iter, density)
        return self.activation.features

        # Pass a list of ints representing the modules of the encoder for which you want to extract features
    def create_forward_hooks(self, layer_list):
        modules = list(self.modules())
        self.activations = {i: SaveFeatures(modules[i]) for i in layer_list}
    
    # Pass the int value of the layer for which you want a feature map.  Only call this after passing 
    # inputs to the encoder and you will get the output associated with those inputs
    def extract_layer_features(self, layer):
        return self.activations[layer].features
    
    def close_forward_hooks(self):
        for activation in self.activations.values():
            activation.close()

