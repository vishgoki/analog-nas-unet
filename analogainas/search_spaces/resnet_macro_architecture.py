"""Search Space Macro-architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
import functools


def round_(filter):
    return round(filter / 3)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='down', kernel_size=3, padding=1, upsample_kernel=2):
        super(UNetBlock, self).__init__()
        if mode == 'up':
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=upsample_kernel, stride=2, padding=upsample_kernel//2)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class UNetGroup(nn.Module):
    def __init__(self, in_channels, out_channels, depth, mode='down', merge_mode='concat'):
        super(UNetGroup, self).__init__()
        self.mode = mode
        self.merge_mode = merge_mode
        self.blocks = nn.ModuleList()
        channels = in_channels

        for i in range(depth):
            self.blocks.append(UNetBlock(channels, out_channels, mode='down'))
            channels = out_channels
        
        if mode == 'up':
            self.up = UNetBlock(channels, out_channels, mode='up')

    def forward(self, x, skip=None):
        if self.mode == 'down':
            skips = []
            for block in self.blocks:
                x = block(x)
                skips.append(x)
            return x, skips
        elif self.mode == 'up':
            x = self.up(x)
            if skip is not None and self.merge_mode == 'concat':
                x = torch.cat([x, skip], dim=1)
            for block in self.blocks:
                x = block(x)
            return x

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.down_groups = nn.ModuleList()
        self.up_groups = nn.ModuleList()

        num_groups = len(config['channels']) - 1
        channels = [config['in_channels']] + list(config['channels'])
        up_channels = list(config['channels'])

        # Creating the downsampling groups
        for i in range(num_groups):
            depth = config['num_res_units']
            group = UNetGroup(channels[i], channels[i+1], depth=depth, mode='down')
            self.down_groups.append(group)

        # Creating the upsampling groups
        for i in range(num_groups - 1, -1, -1):
            depth = config['num_res_units']
            group = UNetGroup(up_channels[i], channels[i], depth=depth, mode='up', merge_mode='concat')
            self.up_groups.append(group)

        # Final convolution
        self.final_conv = nn.Conv3d(up_channels[0], config['out_channels'], kernel_size=1)

    def forward(self, x):
        skips = []
        for group in self.down_groups:
            x, skip = group(x)
            skips.append(skip[-1])  # Collect the last skip connection from each group

        skips = reversed(skips)  # Reverse for the upsampling path
        for skip, group in zip(skips, self.up_groups):
            x = group(x, skip=skip)

        x = self.final_conv(x)
        return x


class ResidualBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_size,
                 stride,
                 branch_index):
        super(ResidualBranch, self).__init__()

        self.residual_branch = nn.Sequential()

        self.residual_branch.add_module('Branch_{}:ReLU_1'
                                        .format(branch_index),
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_1'
                                        .format(branch_index),
                                        nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=filter_size,
                                                  stride=stride,
                                                  padding=round_(filter_size),
                                                  bias=False))
        self.residual_branch.add_module('Branch_{}:BN_1'
                                        .format(branch_index),
                                        nn.BatchNorm2d(out_channels))
        self.residual_branch.add_module('Branch_{}:ReLU_2'
                                        .format(branch_index),
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_2'
                                        .format(branch_index),
                                        nn.Conv2d(out_channels,
                                                  out_channels,
                                                  filter_size,
                                                  stride=1,
                                                  padding=round_(filter_size),
                                                  bias=False))
        self.residual_branch.add_module('Branch_{}:BN_2'
                                        .format(branch_index),
                                        nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.residual_branch(x)


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SkipConnection, self).__init__()

        self.s1 = nn.Sequential()
        self.s1.add_module('Skip_1_AvgPool',
                           nn.AvgPool2d(1, stride=stride))
        self.s1.add_module('Skip_1_Conv',
                           nn.Conv2d(in_channels,
                                     int(out_channels / 2),
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False))

        self.s2 = nn.Sequential()
        self.s2.add_module('Skip_2_AvgPool',
                           nn.AvgPool2d(1, stride=stride))
        self.s2.add_module('Skip_2_Conv',
                           nn.Conv2d(in_channels,
                                     int(out_channels / 2)
                                     if out_channels % 2 == 0
                                     else int(out_channels / 2) + 1,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False))

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = F.relu(x, inplace=False)
        out1 = self.s1(out1)

        out2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        out2 = self.s2(out2)

        out = torch.cat([out1, out2], dim=1)
        out = self.batch_norm(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 filter_size,
                 res_branches,
                 stride):
        super(BasicBlock, self).__init__()

        self.branches = nn.ModuleList([
            ResidualBranch(n_input_plane,
                           n_output_plane,
                           filter_size,
                           stride,
                           branch + 1)
            for branch in range(res_branches)])

        self.skip = nn.Sequential()
        if n_input_plane != n_output_plane or stride != 1:
            self.skip.add_module('Skip_connection',
                                 SkipConnection(n_input_plane,
                                                n_output_plane,
                                                stride))

    def forward(self, x):
        out = sum([self.branches[i](x)
                   for i in range(len(self.branches))])
        return out + self.skip(x)


class ResidualGroup(nn.Module):
    def __init__(self,
                 block,
                 n_input_plane,
                 n_output_plane,
                 n_blocks,
                 filter_size,
                 res_branches,
                 stride):
        super(ResidualGroup, self).__init__()
        self.group = nn.Sequential()
        self.n_blocks = n_blocks

        self.group.add_module('Block_1',
                              block(n_input_plane,
                                    n_output_plane,
                                    filter_size,
                                    res_branches,
                                    stride=1))

        # The following residual block do not perform
        # any downsampling (stride=1)
        for block_index in range(2, n_blocks + 1):
            block_name = 'Block_{}'.format(block_index)
            self.group.add_module(block_name,
                                  block(n_output_plane,
                                        n_output_plane,
                                        filter_size,
                                        res_branches,
                                        stride=1))

    def forward(self, x):
        return self.group(x)


class Network(nn.Module):
    def __init__(self, config, input_dim=(1, 32, 32), classes=10):
        super(Network, self).__init__()

        self.M = config["M"]
        self.residual_blocks = {'Group_1': config["R1"],
                                'Group_2': config["R2"],
                                'Group_3': config["R3"],
                                'Group_4': config["R4"],
                                'Group_5': config["R5"]
                                }

        self.widen_factors = {'Group_1': config["widenfact1"],
                              'Group_2': config["widenfact2"],
                              'Group_3': config["widenfact3"],
                              'Group_4': config["widenfact4"],
                              'Group_5': config["widenfact5"]
                              }

        self.res_branches = {'Group_1': config["B1"],
                             'Group_2': config["B2"],
                             'Group_3': config["B3"],
                             'Group_4': config["B4"],
                             'Group_5': config["B5"]
                             }

        self.conv_blocks = {'Group_1': config["convblock1"],
                            'Group_2': config["convblock2"],
                            'Group_3': config["convblock3"],
                            'Group_4': config["convblock4"],
                            'Group_5': config["convblock5"]
                            }

        # Add filter_size to the config space to be considered.
        self.filters_size = {'Group_1': 3,
                             'Group_2': 3,
                             'Group_3': 3,
                             'Group_4': 3,
                             'Group_5': 3
                             }

        self.model = nn.Sequential()
        block = BasicBlock
        self.blocks = nn.Sequential()
        self.blocks.add_module('Conv_0',
                               nn.Conv2d(3,
                                         config["out_channel0"],
                                         kernel_size=7,
                                         stride=1,
                                         padding=1,
                                         bias=False))

        self.blocks.add_module('BN_0',
                               nn.BatchNorm2d(config["out_channel0"]))

        feature_maps_in = int(round(config["out_channel0"]
                              // self.widen_factors['Group_1']))

        self.blocks.add_module('Group_1',
                               ResidualGroup(block,
                                             config["out_channel0"],
                                             feature_maps_in,
                                             self.residual_blocks['Group_1'],
                                             self.filters_size['Group_1'],
                                             self.res_branches['Group_1'],
                                             1))
        feature_maps_out = feature_maps_in
        for m in range(2, self.M + 1):
            feature_maps_out = int(round(feature_maps_in
                                   // self.widen_factors['Group_{}'
                                                         .format(m)]))
            self.blocks.add_module('Group_{}'.format(m),
                                   ResidualGroup(block,
                                                 feature_maps_in,
                                                 feature_maps_out,
                                                 self.residual_blocks[
                                                     'Group_{}'.format(m)],
                                                 self.filters_size[
                                                     'Group_{}'.format(m)],
                                                 self.res_branches[
                                                     'Group_{}'.format(m)],
                                                 2 if m in (self.M, self.M - 1)
                                                 else 1))
            feature_maps_in = feature_maps_out

        self.feature_maps_out = feature_maps_out
        self.blocks.add_module('ReLU_0',
                               nn.ReLU(inplace=True))
        self.blocks.add_module('AveragePool',
                               nn.AvgPool2d(8, stride=1))

        self.model.add_module("Main_blocks", self.blocks)
        self.fc_len = functools.reduce(operator.mul,
                                       list(self.blocks(
                                            torch.rand(1, *input_dim)
                                            ).shape))

        self.fc = nn.Linear(self.fc_len, classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
