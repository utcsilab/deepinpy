#!/usr/bin/env python

import torch

from deepinpy.utils import utils


class UnrollNet(torch.nn.Module):
    def __init__(self, module_list, data_list, num_unrolls):
        super().__init__()

        self.module_list = module_list
        self.data_list = data_list
        self.num_unrolls = num_unrolls
        self.metadata_list = None

    def batch(self, data):
        for i in range(len(self.module_list)):
            self.module_list[i].batch(data)

    def forward(self, x):
        metadata_list = []
        for i in range(self.num_unrolls):
            _data_list = []
            for module, data in zip(self.module_list, self.data_list):
                x = module(x)
                m = module.get_metadata()
            metadata_list.append(m)
        self.metadata_list = metadata_list
        return x

    def get_metadata(self):
        return self.metadata_list

    #def forward_all(self, x):
        #data_list = []
        #_x = x
        #x_list = [_x]
        #for i in range(self.num_unrolls):
            #_x_list = []
            #_data_list = []
            #for module in self.module_list
                #_x, _d = self.model(_x)
                #_x_list.append(_x)
                #_data_list.append(_d)
            #x_list.append(_x_list)
            #data_list.append(_data_list)
        #return x_list, data_list
