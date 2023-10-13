import torch
import torch.nn as nn
import math
import sys
sys.path.append('..')


# adaptation module
class MatPerturb(nn.Module):
    def __init__(self, dims):
        super(MatPerturb, self).__init__()
        self.perturbation = torch.nn.Parameter(torch.zeros(dims), requires_grad=True)
        # torch.nn.init.uniform_(self.perturbation)
        torch.nn.init.zeros_(self.perturbation)
        self.register_parameter('perturb', self.perturbation)
        self.activation = nn.Sigmoid()
        # self.scale = scale

    def forward(self, x, noise, threshold, scale):
        # normalized_adapt = torch.vstack([(a - torch.mean(a)).unsqueeze(0) for a in self.perturbation])
        batch_mean = torch.mean(self.perturbation, dim=(2, 3))
        normalized_delta = self.perturbation - batch_mean[:, None, None] * torch.ones(self.perturbation.shape[-2:]).to(x.device)
        x_tilda = x + scale * normalized_delta
        return x_tilda, normalized_delta

    def initialize(self):
        # torch.nn.init.uniform_(self.perturbation)
        torch.nn.init.zeros_(self.perturbation)


class CNN2d(nn.Module):
    def __init__(self, n_channels, active=None):
        super(CNN2d, self).__init__()
        if active is not None:
            activation = getattr(torch.nn, active)
        layer_list = []
        for i in range(len(n_channels) - 1):
            layer_list.append(nn.Conv2d(n_channels[i], n_channels[i+1], kernel_size=1, stride=1))
            if active is not None and i != len(n_channels) - 2:
                layer_list.append(activation())
        # layer_list.append(nn.Tanh())
        self.nn_main = nn.Sequential(*layer_list)
        self.normalize = nn.InstanceNorm2d(n_channels[-1])
        # self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, noise, threshold, scale):
        if noise is not None: adapt = self.nn_main(noise)
        else: adapt = self.nn_main(x)
        # batch_mean = torch.mean(adapt, dim=(2,3))
        # normalized_adapt = adapt - batch_mean[:, :, None, None] * torch.ones(adapt.shape[-2:]).to(x.device)
        # normalized_adapt = self.normalize(adapt)
        normalized_adapt = adapt
        # normalized_adapt = normalize_and_scale(adapt, threshold)
        x_tilda = x + scale * normalized_adapt
        return x_tilda, normalized_adapt

    def initialize(self):
        for conv in self.nn_main:
            class_name = conv.__class__.__name__
            if class_name.find('Conv') != -1:
                conv.reset_parameters()

########################################
# Copied from https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations/
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation, use_dropout, use_bias, skip=True):
        super(ResnetBlock, self).__init__()
        self.skip = skip
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        if self.skip:
            out = x + self.conv_block(x)
        else:
            out = self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, n_blocks=6, n_downsampling=0,
                 act_type='selu', use_dropout=False, padding_type='reflect', skip=True):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU()
        elif act_type == 'hardswish':
            self.act = nn.Hardswish()
        else:
            self.act = nn.ReLU()

        model0 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        for i in range(n_downsampling):
            mult = 2**i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, activation=self.act,
                                   use_dropout=use_dropout, use_bias=use_bias, skip=skip)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model0 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]

        model0 += [nn.ReflectionPad2d(1)]
        model0 += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
        model0 += [nn.Tanh()]

        self.model0 = nn.Sequential(*model0)

    def get_params(self):
        return self.parameters()

    def forward(self, x):
        delta = self.model0(x)
        return delta

