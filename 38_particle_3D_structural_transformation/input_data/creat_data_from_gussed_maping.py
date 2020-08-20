#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:57:19 2020

@author: ben
"""

import numpy as np


def read_xyz_to_data(file1, file2, one_on_one_map):
    Temp1 = open(file1, "r").readlines()[2:]
    Temp2 = open(file2, "r").readlines()[2:]
    data1, data2 = [], []
    for i, pair in enumerate(one_on_one_map):
        tmp1 = Temp1[pair[0]-1].strip("\n").split()[1:4]
        data1.append( [float(v) for v in tmp1] )
        tmp2 = Temp2[pair[1]-1].strip("\n").split()[1:4]
        data2.append( [float(v) for v in tmp2] )
    return data1, data2

def write_data_to_xyz(data, out_file):
    with open(out_file, "w") as fout:
        fout.write(f"{len(data)}\nAr\n")
        for pt in data:
            x, y, z = round(pt[0], 8), round(pt[1], 8), round(pt[2], 8)
            fout.write(f"Ar\t{x}\t{y}\t{z}\n")

### Ico, Oct atom pairs:
one_on_one_map = [[6, 1], [38, 36], [33, 23], [20, 10], [28, 16],
                  [23, 26], [37, 27], [32, 6], [8, 21], [27, 30],
                  [34, 13], [14, 7], [12, 17], [2, 9], [11, 3],
                  [29, 22], [24, 20], [25, 8], [10, 29], [15, 24],
                  [30, 15], [22, 31],[18, 5], [4, 38], [1, 2],
                  [3, 14], [19, 28], [9, 25], [16, 18], [26, 33],
                  [17, 37], [31, 32], [21, 19], [7, 4], [5, 12],
                  [13, 11], [35, 34], [36, 35]]

file1 = "./38_Ico.xyz"
file2 = "./38_Oct.xyz"
data1, data2 = read_xyz_to_data(file1, file2, one_on_one_map)
write_data_to_xyz(data1, "./updated_38_Ico.xyz")
write_data_to_xyz(data2, "./updated_38_Oct.xyz")