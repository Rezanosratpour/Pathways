# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 00:48:04 2022

@author: Reza
"""


def BIAS(Obs,Product, th) :
    N=len(Obs)
    F=0.0
    H=0.0
    M=0.0
    Th=th
    for i in range(N):
        O=float(Obs[i])
        P=float(Product[i])
        if O>=Th and P>=Th :  H=H+1
        if O<Th and P>=Th :   F=F+1
        if O>=Th and P<Th :   M=M+1
    try:
        bias=(H+F)/(H+M)
    except:
        bias="Undefined"
    return bias

def CSI(Obs,Product,th):
    N=len(Obs)
    F=0.0
    H=0.0
    M=0.0
    for i in range(N):
        O=float(Obs[i])
        P=float(Product[i])
        if O>=th and P>=th :  H=H+1
        if O<th and P>=th :   F=F+1
        if O>=th and P<th :   M=M+1
    try:
        csi=(H)/(H+M+F)
    except:
        csi="Undefined"
    return csi

def FAR(Obs,Product, th):
    N=len(Obs)
    F=0.0
    H=0.0
    Th=th
    for i in range(N):
        O=float(Obs[i])
        P=float(Product[i])
        if O>=Th and P>=Th :  H=H+1
        if O<Th and P>=Th :   F=F+1
    try:
        far=(F)/(F+H)
    except:
        far="Undefined"
    return far


def POD(Obs,Product, th):
    N=len(Product)
    H=0
    M=0
    Th=th
    for i in range(N):
        O=float(Obs[i])
        P=float(Product[i])
        if O>=Th and P>=Th :  H=H+1
        if O>=Th and P<Th  :  M=M+1
    try:
        pod=float(H)/float(H+M)
    except:
        pod="Undefined"
    return pod

import numpy