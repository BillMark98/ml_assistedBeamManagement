# -*- coding: utf-8 -*-
lX=None
lz=int
ln=False
lP=list
lB=filter
lS=print
lh=len
lq=range
lW=Exception
lV=True
lv=dict
ly=enumerate
ls=zip
lD=map
li=str
lo=max
lu=super
lf=set
lH=open
lF=min
le=sum
lG=round
lU=any
lp=bool
lR=ValueError
lb=float
lJ=globals
lt=eval
mc=tuple
# mr=file
"""
Created on Jun 1st

@author: Panwei

top k version 2, also include rss, custom loss function

"""

import os
rn=os.listdir
rz=os.mkdir
rX=os.makedirs
rI=os.getcwd
rO=os.path
ra=os.chdir
import time 
rP=time.time
import copy
import torch
rF=torch.load
rH=torch.save
rf=torch.topk
ru=torch.Tensor
ro=torch.device
ri=torch.backends
rD=torch.manual_seed
rs=torch.sum
ry=torch.mean
rv=torch.arange
rV=torch.matmul
rW=torch.tensor
rq=torch.utils
rh=torch.nn
rS=torch.optim
rB=torch.cuda
import datetime
import itertools
re=itertools.product
import numpy as np
wt=np.int64
wJ=np.bitwise_xor
wb=np.argmin
wR=np.linspace
wp=np.sqrt
wU=np.nan
wG=np.nanmax
we=np.nanmin
wF=np.flipud
# lo=max
# lu=super
# lf=set
# lH=open
# lF=min
# le=sum
# lG=round
# lU=any
wH=np.max
wf=np.min
wu=np.argmax
wo=np.floor
wi=np.argsort
wD=np.flip
ws=np.unique
wy=np.round
wv=np.diff
wV=np.concatenate
wW=np.vstack
wq=np.savetxt
wh=np.apply_along_axis
wS=np.equal
wB=np.reshape
wP=np.save
wn=np.hstack
wz=np.pi
wX=np.linalg
wI=np.dot
wO=np.sin
wa=np.cos
wA=np.arctan2
wd=np.random
wN=np.stack
wM=np.isscalar
wQ=np.abs
wK=np.std
wj=np.mean
wg=np.float64
wT=np.copy
wE=np.expand_dims
wx=np.log2
wk=np.nan_to_num
wL=np.divide
wY=np.sum
wC=np.tile
wm=np.transpose
wl=np.log10
wr=np.power
wc=np.loadtxt
rt=np.zeros
rJ=np.ceil
rb=np.isinf
rR=np.isnan
rp=np.append
rU=np.arange
rG=np.array
import pandas as pd
lw=pd.DataFrame
lr=pd.concat
lc=pd.read_csv
import shutil
lm=shutil.copy
# import train_test2_customLoss as func
import matplotlib.pyplot as plt
# from helpFunc import *
import json
lY=json.dumps
lC=json.dump
import ast
lL=ast.literal_eval
# os.chdir(os.getcwd())

import sys

ra(rO.dirname(rO.abspath(__file__)))

# import math
import configparser
lk=configparser.ConfigParser
import glob
lx=glob.glob

frankfurtTxArrays=[
5194466,
3054566,
5836116,
4434416,
4581536,
3163526,
5853136,
3625286,
5955386,
3552206,
3725856,
4063746,
1193486,
5751526,
1564986,
5355566,
5203856,
1625746,
4465066,
4513396,
2805226,
3251696,
182996,
3821656,
4272166,
3594326,
5285056,
5942206,
3782306,
2376016,
5152246,
5101656,
5392746,
5853826,
3453006,
4362816,
2484006,
2245166,
1893956,
6064536,
1774496,
2174486,
4475786,
3085836,
2261176,
3623066
]


def cr(rxBeamSet, dataMatrix, txNum = 64, rxNum = 16, cutoff = 64, offset=0, keepFront=lX, keepRear = lX):
    """
    given the beamSet, the dataMatrix, returns the rss coresponding to the rx given in the rxBeamSet
    rxBeamSet starts from 1 ( the matlab and used beam indexing)
    column starts from 0  (python indexing)
    with cutoff meaning for each beam select up till cutoff 
    
    keepFront: keep the front several columnss 
        None or ndarray
        
    keepRear: keep the rear columns
        None or ndarray
    """
    colIndices = rG([])
    for b in rxBeamSet:
        b = b - 1 # convert to python index
        b_colIndices = txNum * b + rU(txNum) + offset
        colIndices = rp(colIndices, b_colIndices)
    if keepFront is not lX:
        colIndices = rp(keepFront, colIndices)
    if keepRear is not lX:
        colIndices = rp(colIndices, keepRear)
    colIndices = colIndices.astype(lz)
    return dataMatrix[:, colIndices]

def cw(dataMatrix, defaultVal = -174, drop=ln):
    """
    replace all NaN to defaultVal
    if drop=True, drop those rows which has na
    
    """
    if drop:
        dataMatrix = dataMatrix[~rR(dataMatrix).any(axis=1),:]
        # also drop inf
        dataMatrix = dataMatrix[~rb(dataMatrix).any(axis=1),:]
        return dataMatrix
    dataMatrix = dataMatrix.fillna(defaultVal)
    return dataMatrix

def cl(beamIndex, TxNum, RxNum):
    """
    given the number of tx, number of rx, the matrix index (scalar) of beamIndex,

    num rx * num tx
    return the tx index, rx index

    the index is arranged as follows 
        1,2,3, ... TxNum, 
        TxNum+1, ...
    
    beamIndex starts from 0 (output from the NN)
    to calculate, add 1
    """
    beamIndexP1 = beamIndex + 1
    rxIndex = rJ(beamIndexP1 / TxNum).astype(lz)
    txIndex = beamIndexP1 % TxNum + 1
    return txIndex, rxIndex

def cm(rootFolder, fileName="test_accs.txt", rowCount = 1, saveName = "testAcc"):
    """
    given rootFolder, iterate through file to extrat the testAcc and plot
    """
    fileDepth = lx(rO.join(rootFolder,"*/"))
    dirsDepth = lP(lB(lambda f: rO.isdir(f), fileDepth))
    lS(dirsDepth)
    # dirsDepth = list(dirsDepth)
    totalNum = lh(dirsDepth)
    testAcc = rt((rowCount, totalNum))
    for dirName in dirsDepth:
        # extract the number
        dirName_suffix = dirName.split("/")[-2]
        dirName_suffix = dirName_suffix.split("_")[-1]
        colIndex = lz(dirName_suffix)
        # form the test acc.txt
        fileAcc = rO.join(dirName, fileName)
        if rO.exists(fileAcc):
            testTemp = wc(fileAcc)
            testAcc[:, colIndex] = testTemp[:rowCount]
    lS(testAcc)
    # plot
    plt.figure()
    legends = []
    colors = ["g*-", "b*-","r*-","m*-"]
    for i in lq(rowCount):
        plt.plot(rU(1,lh(testAcc[i,:])+1), testAcc[i,:], colors[i])
        legends.append(f"top{i+1}-accuracy")
    plt.xlabel("M")
    plt.ylabel("acc[%]")
    plt.title("accuracy wrt M value")
    plt.legend(legends)
    plt.savefig(rO.join(rootFolder, saveName + ".png"))
    plt.savefig(rO.join(rootFolder, saveName + ".pdf"))
    plt.savefig(rO.join(rootFolder, saveName + ".svg"))
    plt.show()

# def getTxRxBeamIndex(beamIndex, TxNum, RxNum):
#     """
#     given the number of tx, number of rx, the matrix index (scalar) of beamIndex,

#     num rx * num tx
#     return the tx index, rx index

#     According to the norm, first sweep rx, and for each fixed rx, sweep tx this is how the index works
#     """
#     txIndex = np.ceil(beamIndex / TxNum)
#     rxIndex = beamIndex % TxNum + 1
#     return txIndex, rxIndex

def cC(powerDB):
    """
    convert power in dbm to power (mW)
    """
    powermW = wr(10, powerDB/10)
    return powermW

def cY(powermW):
    """
    convert power in mW to dmB
    """
    powerDBM = 10 * wl(powermW)
    return powerDBM

def cL(refBeamIndices, predBeamIndices,totalBeamRSS, beamNums = 5):
    """
    given the reference and predicted beam indices, calcualte the power loss metric see (5) in paper

    refBeamIndices:
        (numData,beamNums)   while each row correspond to the predicted best beamNums  beam indices
    
    totalBeamRSS:
        (numData, totalBeamNums)
    """
    if refBeamIndices.shape[0] != predBeamIndices.shape[0]:
        raise lW("refBeam shape has different number of rows as predBeamIndices")
    if refBeamIndices.shape[0] != totalBeamRSS.shape[0]:
        raise lW("refBeam shape has different number of rows as totalBeamRss")
    beamNums = refBeamIndices.shape[1]
    # get the noise power
    noisePower = totalBeamRSS.min()
    noisePower = cC(noisePower)
    rowIndices = wm(wC(rU(0, totalBeamRSS.shape[0]), (beamNums,1)))
    refBeamRSS = totalBeamRSS[rowIndices, refBeamIndices]
    predBeamRSS = totalBeamRSS[rowIndices, predBeamIndices]

    # convert to power in mW and subtract the minimum
    refBeamRSS = cC(refBeamRSS) - noisePower
    predBeamRSS = cC(predBeamRSS) - noisePower
    plLoss = rt(beamNums)
    for i in lq(beamNums):
        # sum up till that column
        refSums = wY(refBeamRSS[:,rU(i+1)], axis=1)
        predSums = wY(predBeamRSS[:,rU(i+1)], axis=1)
        # possible to have inf cz devide by 0
        # powerQuotient = np.divide(refSums, predSums)
        powerQuotient = wL(predSums, refSums)
        # replace nan to 0
        if rR(powerQuotient).any():
            lS("Quotient contains nan, replace with 0")
            powerQuotient = wk(powerQuotient)
        powerQuotient = powerQuotient.mean()
        plLoss[i] = cY(powerQuotient)
    lS("pLLoss")
    lS(plLoss)
    return plLoss

def ck(RSS, B = 800e6):
    #GET_SNR Summary of this function goes here
    #   Detailed explanation goes here

    EIRP_A = 32
    T = 330  


    k = 1.3806504e-23 #Boltzmann constant
    EIRP_bs = 25 # EIRP Basestation (5G-NR model)
    NF = 10 # Noise Fig (802.11 requirement)
    L1 = 5 # 5dB implemenation loss (802.11 requirement)
    N = 10*wl(k*B*T*1000) #factor 1000 to get the noise into dBm  

    SNR = RSS + (EIRP_bs - EIRP_A) - (N+NF+L1)

    #SuperC
    #max sensitivity:Smax = -62dBm
    #S' = Smax +(EIRP_bs-EIRP-A) = -69dBm
    #SNRmin = -4.5dBm
    #SNRmax = 20dBm
    #SNR(RSS=S') = -6.62dBm < SNRmin
    return SNR

def cx(SNR, bandwidth = 800e6):
    # Capacity given by Shannon limit

    #   RSS and noise in dBm

    # ETSI TR 136 942 V14.0.0, A.2 Modelling of Link Adaption
    # https://www.etsi.org/deliver/etsi_tr/136900_136999/136942/14.00.00_60/tr_136942v140000p.pdf
    # page 99
    # NF = 9 dB may be appropriate for LTE
    # 802.11 standard requires 10 dB NF + 5 dB implementa

    SINR_min = -4.5 
    alpha = 0.75
    SINR_max = 20 
    TH_max = 4 # max rate is 4 Gbps

    #RSSnondB = 10.^((RSS-30)/10)
    #noisenondB = 10.^((noise-30)/10)

    # attenuated shannon
    SNRnondB = wr(10, (SNR/10))# input SNR is in dB, convert to linear scale.
    SpectralEff = alpha * wx(1+SNRnondB)

    # lower limit
    idx = SNR < SINR_min
    SpectralEff[idx] = 0

    # upper limit
    idxH_max = SNR > SINR_max
    SpectralEff[idxH_max] = TH_max

    # convert to rate using bandwidth (in Hz)
    # bandwidth = 800e6 # 800 MHz
    C = bandwidth * SpectralEff
    return C


def cE(rss, bandwidth = 800e6):
    """
    convert the RSS to channel cpacity
    """
    # first conver
    
    SNR = ck(rss, bandwidth)
    # get C
    channelCapa = cx(SNR, bandwidth)
    return channelCapa

def cT(channelCapPred, channelCapRef, defaultVal=-1,drop=lV, quotientMethod = 2):
    """
    compute the channel capacity
    default value = -1 (nan i.e. deadzone)
    
    quotientMethod:
    
    1. cap_i_pred/cap_i_ref
    2. cap_i_pred/cap_1_ref
    """
    channelReturnDict = lv()
    if quotientMethod == 1:
        effi = wL(channelCapPred, channelCapRef)
    elif quotientMethod == 2:
        top1_refCap = wC(wE(channelCapRef[:,0],-1), (1, channelCapRef.shape[-1]))
        effi = wL(channelCapPred, top1_refCap)
        effi_ref = wL(channelCapRef, top1_refCap)
    effi = cw(effi, defaultVal, drop=drop)
    effi_ref = cw(effi_ref, defaultVal, drop=drop)
    
    if quotientMethod == 1:
        channelReturnDict["effi_q1"] = effi
    elif quotientMethod == 2:
        channelReturnDict["effi_q2"] = effi
    
    channelReturnDict["effi_ref_q2"] = effi_ref
    return channelReturnDict

def cg(dataMatrix):
    """
    accumulate the Sum of columns, i.e.
    
    c1, c2, c3, c4 ->
    c1, c1+c2, c1+c2+c3, c1+c2+c3+c4
    
    """
    colNums = dataMatrix.shape[1]
    for col in lq(1,colNums):
        dataMatrix[col] = dataMatrix[col-1] + dataMatrix[col]
    return dataMatrix

def cj(dataMatrix):
    """
        calculate cumulative mean (columnwise),
        
        i.e. if one row is given by
        
        
    e1,e2,...,ek
    
    then the top-k channel efficiency  is calculated
    
    e1, (e1+e2)/2, (e1+e2+e3)/3 ,.... (e1 + ... + ek)/k
        
    Args:
        dataMatrix (_type_): _description_
    """
    dataResult = wT(dataMatrix).astype(wg)
    # iterate throught column
    for i in lq(dataMatrix.shape[-1]):
        if dataMatrix.ndim == 1:
            dataResult[i] = wj(dataMatrix[:(i+1)])
        elif dataMatrix.ndim ==2:
            dataResult[:,i] = wj(dataMatrix[:,:(i+1)], axis=-1)
        else:
            raise lW("cumMeanCol do not support dim > 2 array")
    return dataResult

def cK(rss, bandwidth = 800e6):
    """
    given the RSS of topk beams, (so each column is the RSS corresponding to one topk beam) calculate the channel capacity
    """
    # make no sense to accumulate the rss, because each (row,column) corresponds to one particular bpl, and each time one can only use that so could not use several pairs
    # rss_accum = cumColSum(rss)
    # channelCapa = convertRSStoChannel(rss_accum, bandwidth)
    
    channelCapa = cE(rss, bandwidth)
    return channelCapa

def cQ(rssMatrix, beamIndices, beamNum = 5, bandwidth=800e6):
    """
    From the beamIndices, compute the channel capacity matrix

    Args:
        rssMatrix (_type_):
            stores the rss for each beam index where row corresponds to a (tx, rx) spot
            so the wanted rss should be like
            rssMatrix[:, beamIndices]
        beamIndices (_type_): the beamIndices (column indices to index the rssMatrix)
        bandwidth (_type_, optional): _description_. Defaults to 800e6.
        
        
    """
    # convert beamIndices to int
    beamIndices = beamIndices.astype(lz)
    # get the RSS
    rss_val = rssMatrix[rU(rssMatrix.shape[0])[:,lX], beamIndices]
    channelCap = cK(rss_val,bandwidth = bandwidth)
    return channelCap

def cM(rssMatrix, predIndices,refIndices, beamNum=5, bandwidth=800e6, dropOutage=lV, calculateMean=ln,quotientMethod=2):
    """
    
    from the predicted indices, rssMatrix, calculate the capacity efficiency
    
    top-k beams the efficiency is calculated cumulatively mean-wise, i.e. if quotient of capacity of predicted / capcity of ref row is given
    
    e1,e2,...,ek
    
    then the top-k channel efficiency  is calculated
    
    e1, (e1+e2)/2, (e1+e2+e3)/3 ,.... (e1 + ... + ek)/k
    
    Args:
        rssMatrix (_type_): _description_
        predIndices (_type_): _description_
        refIndices (_type_): _description_
        beamNum (int, optional): _description_. Defaults to 5.
        bandwidth (_type_, optional): _description_. Defaults to 800e6.
        calculateMean:
            if True, will calculate the mean of the efficiency
            
    Return:
    
        channelEffiReturnDict["channelEffi"] = channelEffi
    channelEffiReturnDict["channelEffi_ref"] = channelEffi_ref
    channelEffiReturnDict["channelEffi_std"] = channelEffi_std
    channelEffiReturnDict["channelEffi_ref_std"] = channelEffi_ref_std
    """
    channelCap_ref = cQ(rssMatrix, refIndices, beamNum=beamNum, bandwidth=bandwidth)
    # channelCap_ref = topK_rssToChannel(RSS_ref,bandwidth = bandwidth)
    # RSS_pred = rssMatrix[np.arange(rssMatrix.shape[0])[:,None], all_pred_labels[:,:beamNums]]
    channelCap_pred = cQ(rssMatrix, predIndices, beamNum=beamNum, bandwidth =bandwidth)
    
    # channelCap_pred = topK_rssToChannel(RSS_pred, bandwidth=bandwidth)
    # If the best beam is in the topX, then +1 hit for that batch
    # channelCapQuo = np.mean(getChannelEffiQuotient(channelCap_pred, channelCap_ref,drop=dropOutage),axis=0)
    channelEffiQuotientReturnDict = cT(channelCap_pred, channelCap_ref,drop=dropOutage,quotientMethod=quotientMethod)
    if quotientMethod == 1:
        channelEffi = channelEffiQuotientReturnDict["effi_q1"]
    elif quotientMethod == 2:
        channelEffi = channelEffiQuotientReturnDict["effi_q2"]
    channelEffi_ref = channelEffiQuotientReturnDict["effi_ref_q2"]
    
    if quotientMethod == 1:
        channelEffi = cj(channelEffi)
        
    channelEffi_std = wK(channelEffi,axis=0)    
    channelEffi_ref_std = wK(channelEffi_ref, axis=0)
    if calculateMean:
        channelEffi = wj(channelEffi,axis=0)
        channelEffi_ref = wj(channelEffi_ref, axis=0)
    channelEffiReturnDict = lv()
    channelEffiReturnDict["channelEffi"] = channelEffi
    channelEffiReturnDict["channelEffi_ref"] = channelEffi_ref
    channelEffiReturnDict["channelEffi_std"] = channelEffi_std
    channelEffiReturnDict["channelEffi_ref_std"] = channelEffi_ref_std
    
    return channelEffiReturnDict
                    
    
def cN(folderPath, fileNames, outFilePath, mode = "topBeam"):
    """
    given a list of fileNames, each one stores the dataMatrix at one tx position, merge it with an extra tx id
    
    Parameters:
    
    mode: 
        matrix form
        "topBeam": pos_x, pos_y, Index1,RSS1,Index2,RSS2,Index3,RSS3,Index4,RSS4,Index5,RSS5,
        "total" :  pos_x, pos_y, topBeamIndex1, ..., topBeamIndexk, totalBeamRss....
    """
    for index, fileName in ly(fileNames):
        dataPath = rO.join(folderPath,fileName)
        dataMatrix = lc(dataPath, index_col=ln)
        rowNum = dataMatrix.shape[0]
        dataMatrix["tx_id"] = rowNum * [index+1]
        if index == 0:
            dataMatrixTotal = dataMatrix
        else:
            dataMatrixTotal = lr([dataMatrixTotal, dataMatrix])
    
    outFileTotalPath = rO.join(folderPath, outFilePath)
    # save the data
    dataMatrixTotal.to_csv(outFileTotalPath, index=ln)
    
def cd(folderPath, fileNames, txPositions, outFilePath, mode = "topBeam"):
    """
    given a list of fileNames, each one stores the dataMatrix at one tx position, merge it with an extra tx id
    
    Parameters:
    
    mode: 
        matrix form
        "topBeam": pos_x, pos_y, Index1,RSS1,Index2,RSS2,Index3,RSS3,Index4,RSS4,Index5,RSS5,
        "total" :  pos_x, pos_y, topBeamIndex1, ..., topBeamIndexk, totalBeamRss....
        
    txPositions:
        with tx positions, (tuple wise)
    
    saved matrix:
    
        the original data, tx_pos_x, tx_pos_y, tx_id
    """
    
    for index, (fileName, txPos) in ly(ls(fileNames, txPositions)):
        dataPath = rO.join(folderPath,fileName)
        dataMatrix = lc(dataPath, index_col=ln)
        rowNum = dataMatrix.shape[0]
        dataMatrix["tx_pos_x"] = rowNum * [txPos[0]]
        dataMatrix["tx_pos_y"] = rowNum * [txPos[1]]
        dataMatrix["tx_id"] = rowNum * [index+1]
        if index == 0:
            dataMatrixTotal = dataMatrix
        else:
            dataMatrixTotal = lr([dataMatrixTotal, dataMatrix])
    
    outFileTotalPath = rO.join(folderPath, outFilePath)
    # save the data
    dataMatrixTotal.to_csv(outFileTotalPath, index=ln)
    
def cA(testval, refvalue, upBound, tolerance=0):
    if (wQ(testval - refvalue) <= tolerance):
        return lV
    if testval > refvalue:
        testval = (testval + tolerance) % upBound
    else:
        refvalue = (refvalue + tolerance) % upBound
    
    return (wQ(testval - refvalue) <= tolerance)
def ca(coarseBeamIndex, fineBeamIndex, coarseTxNum, coarseRxNum, fineTxNum, fineRxNum, tolerance = 0):
    """
    calculate for coarse, fine beam index, whether it hits
    note beamIndex starts from 0 so add 1

    Args:
        coarseBeamIndex (_type_): _description_
        fineBeamIndex (_type_): _description_
        fileFine (_type_): _description_
        coarseTxNum (_type_): _description_
        coarseRxNum (_type_): _description_
        fineTxNum (_type_): _description_
        fineRxNum (_type_): _description_
        
    return:
        txhit, rxhit
    """
    coarseBeamIndex += 1
    fineBeamIndex += 1
    # get the coarsetx
    if not wM(coarseBeamIndex):
        raise lW("currently only support scalar")

    coarseTx, coarseRx = cl(coarseBeamIndex, coarseTxNum, coarseRxNum)
    fineTx, fineRx = cl(fineBeamIndex, fineTxNum, fineRxNum)
    if fineTxNum % coarseTxNum != 0:
        raise lW("currently only support integer multiple of coarseTxNum")
    if fineRxNum % coarseRxNum != 0:
        raise lW("currently only support integer multiple of coarseRxNum")
    
    ratioTx = fineTxNum / coarseTxNum
    ratioRx = fineRxNum / coarseRxNum
    
    totalTxHit = wY(cA(rJ(fineTx/ratioTx), coarseTx, upBound=coarseTxNum,tolerance=tolerance))
    totalRxHit = wY(cA(rJ(fineRx/ratioRx), coarseRx, upBound=coarseRxNum,tolerance=tolerance))
    
    return totalTxHit, totalRxHit
    
def cO(folderPath, fileCoarse, fileFine, coarseTxNum, coarseRxNum, fineTxNum, fineRxNum,tolerance=0):
    """
    given the coarse beam rss, fine beam rss, validate the ratio that the fine is within the region of coarse

    Args:
        fileCoarse (_type_): _description_
        fileFine (_type_): _description_
    """
    dataCoarsePath = rO.join(folderPath,fileCoarse)
    dataCoarseMatrix = lc(dataCoarsePath, index_col=ln)
    rowCoarseNum = dataCoarseMatrix.shape[0]
    dataFinePath = rO.join(folderPath,fileFine)
    dataFineMatrix = lc(dataFinePath, index_col=ln)
    rowFineNum = dataFineMatrix.shape[0]
    if rowFineNum != rowCoarseNum:
        raise lW("row fine num inequal row coarse")
    
    totalTxHit = 0
    totalRxHit = 0
    for rowNum in lq(rowFineNum):
        coarseX = dataCoarseMatrix.iloc[rowNum,0]
        coarseY = dataCoarseMatrix.iloc[rowNum,1]
        coarseBeamIndex = dataCoarseMatrix.iloc[rowNum,2]
    
    
        fineX = dataFineMatrix.iloc[rowNum,0]
        fineY = dataFineMatrix.iloc[rowNum,1]
        fineBeamIndex = dataFineMatrix.iloc[rowNum,2]
        if (coarseX != fineX) or (coarseY != fineY):
            raise lW("position data not equal, rowNum:{0}".format(rowNum))
        
        txhit, rxhit = ca(coarseBeamIndex, fineBeamIndex, coarseTxNum, coarseRxNum, fineTxNum, fineRxNum,tolerance=tolerance) 
        totalTxHit += txhit
        totalRxHit += rxhit
    
    # calculate ratio
    
    txhitratio = totalTxHit/rowFineNum
    rxhitratio = totalRxHit / rowFineNum
    
    lS("hit ratio with tolerance:{0}".format(tolerance))
    lS("tx hit: {0}".format(txhitratio) )
    lS("rx hit: {0}".format(rxhitratio) )
    return txhitratio, rxhitratio
    
def cI(tx_station):
    # given the tx_station, get the x and y position
    # modulo 6
    tx_station = tx_station//10
    if tx_station < 99999:
        tx_x = tx_station//100
        tx_y = tx_station % 100
    else:
        tx_x = tx_station//1000
        tx_y = tx_station % 1000
    return tx_x, tx_y
   
    
def cX(txArrays):
    printStr = ""
    for num, txPos in ly(txArrays):
        tx_x, tx_y = cI(txPos)
        printStr += "({0},{1})".format(tx_x, tx_y)
        if (num+1) % 7 == 0:
            printStr += "\\\\"
        else:
            printStr += "&"
    lS(printStr)
def cz(path):
    """
    given the path, replace \ with \\
    """
    path.replace("\\","\\\\")
    return path
def cn(indices, totalLen):
    # each row of indices represent one vector with 1 at those specified by the row, the length of vector is given by totalLen
    if indices.ndim == 1:
        indices = indices.reshape((-1,1))
    vecs = rt((indices.shape[0], totalLen))
    vecs[rU(vecs.shape[0])[:,lX], indices] = 1
    return vecs

   
def cP(key):
    """Convert a tuple key to a string representation."""
    return '_'.join(lD(li, key))  # Example: (1, 'a') becomes "1_a"
    
# def is_boolean(element:any) -> bool:
    
#     if element is None:
#         return False
#     if (element == "True") or (element == "False"):
#         return True
    
# def convertToBool(element:any):
#     if element == "True":
#         return True
#     elif element == "False":
#         return False
#     else:
#         raise Exception("not true nor false")
# def is_int(element:any) -> bool:
#     """
#     check if it is convertible to int
#     """
#     if element is None:
#         return False
#     try:
#         int(element)
#         return True
#     except ValueError:
#         return False

# def is_float(element:any) -> bool:
#     """
#     check if it is convertible to float
#     """
#     if element is None:
#         return False
#     try:
#         float(element)
#         return True
#     except ValueError:
#         return False


# def parseConfig(configFile):
#     """
#     parse the config file
#     """
#     config = configparser.ConfigParser()
#     # perserve case
#     config.optionxform=str
#     config.read(configFile)
#     myGlobalVars = globals()
#     # assign all key_name (as a variable name) to corresponding value
#     for section in config.sections():
#         for key,value in config[section].items():
#             # if list then use json.load
#             value = value.strip()
#             if value[0] == "[":
#                 if value[1] == "{":
#                     myGlobalVars[key] = list(eval(value))
#                 elif value[1] == "]":
#                     myGlobalVars[key] = []
#                 else:
#                     myGlobalVars[key] = json.loads(config.get(section, key))
#             elif is_boolean(value):
#                 myGlobalVars[key] = convertToBool(value)
#             elif is_int(value):
#                 myGlobalVars[key] = int(value)
#             elif is_float(value):
#                 myGlobalVars[key] = float(value)
#             elif value == "None":
#                 myGlobalVars[key] = None
#             else:
#                 myGlobalVars[key] = value
    
    # assign some other variables
    
    # global rxSet
    # bestBeamNum = len(beamWeight)
    
    # rxSet = set(list(range(1,rxIndexMax))) # index start from 1
    # combinations = list(itertools.product(scen_idxs, n_beams_list, norm_types, 
    #                                   noises, [1 for i in range(n_reps)], trainModes, lr_scheduleParamArr))

    # sfs_beamSet = set()
    
# if __name__ == "__main__":

#     # testName = r"C:\Users\Panwei"
#     # data_folder = os.path.join(os.getcwd(), "data")
#     # testNameNew = parseWinPath(testName)
#     # print(testNameNew)
#     # mergeData(data_folder, ["dataMatrixAll_tx1_best5.csv", "dataMatrixAll_tx2_best5.csv"], "dataMatrixAll_Total_best5.csv")
#     # mergeData(data_folder, ["dataMatrixPruneRSS5_ignan.csv", "dataMatrix2PruneRSS5_ignan.csv"], "dataMatrixTotalPruneRSS5_ignan.csv")
#     # mergeData(data_folder, ["dataMatrixPruneRSS5.csv", "dataMatrix2PruneRSS5.csv"], "dataMatrixTotalPruneRSS5.csv")
    
    
#     # mergeDataAddTxPos(data_folder, ["dataMatrixAll_tx1_best5.csv", "dataMatrixAll_tx2_best5.csv"],[(169,128),(174,126)], "dataMatrixAll_Total_txPos_best5.csv")
#     # mergeDataAddTxPos(data_folder, ["dataMatrixPruneRSS5_ignan.csv", "dataMatrix2PruneRSS5_ignan.csv"], [(169,128),(174,126)],"dataMatrixTotalPruneRSS5_ignan_txPos.csv")
#     # mergeDataAddTxPos(data_folder, ["dataMatrixPruneRSS5.csv", "dataMatrix2PruneRSS5.csv"],[(169,128),(174,126)], "dataMatrixTotalPruneRSS5_txPos.csv")
    
    
#     # parseConfig("config.ini")
#     # arr = np.reshape(np.arange(9),(3,3))
#     # arrMean = cumMeanCol(arr)
#     # print(arrMean)
#     # data_folder = os.path.join(os.getcwd(), "data")
#     data_folder = "/storage/archive/Panwei/DATA/data"
    
#     # checkFineCoarseBestHitRatio(data_folder,"dataMatrixAll_tx1_rx4tx4_best5.csv","dataMatrixAll_tx1_best5.csv",16,16,64,16, tolerance=0)
    
    
#     # checkFineCoarseBestHitRatio(data_folder,"data_superc_TX1_rx2tx4_totalRSS_best5.csv","dataMatrixAll_tx1_best5.csv",16,8,64,16, tolerance=2)
    
#     mergeData(data_folder, ["data_frankfurt_TX3054566_rx4tx16_totalRSS_best5.csv","data_frankfurt_TX4434416_rx4tx16_totalRSS_best5.csv","data_frankfurt_TX5194466_rx4tx16_totalRSS_best5.csv","data_frankfurt_TX5853136_rx4tx16_totalRSS_best5.csv","data_frankfurt_TX5836116_rx4tx16_totalRSS_best5.csv","data_frankfurt_TX4581536_rx4tx16_totalRSS_best5.csv","data_frankfurt_TX3163526_rx4tx16_totalRSS_best5.csv"], "data_frankfurt_combine_rx4tx16_totalRSS_best5.csv")
    
#     mergeData(data_folder, ["data_frankfurt_TX4434416_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX5194466_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX4581536_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX3163526_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX3054566_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX5853136_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX5836116_rx4tx16_totalRSS_best_removeDead5.csv"], "data_frankfurt_combine_rx4tx16_totalRSS_removeDead_best5.csv")
    
#     mergeData(data_folder, ["data_frankfurt_TX4434416_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX5194466_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX4581536_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX3163526_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX3054566_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX5853136_rx4tx16_totalRSS_best_removeDead5.csv","data_frankfurt_TX5836116_rx4tx16_totalRSS_best_removeDead5.csv"], "data_frankfurt_combine_rx4tx16_totalRSS_removeDead_best5.csv")
#     # print(isWithinTolerance(1.2,1.2,0))
    
    
#     # printTxStations(frankfurtTxArrays)



# -*- coding: utf-8 -*-
"""
@author: Panwei
"""

import os
rn=os.listdir
rz=os.mkdir
rX=os.makedirs
rI=os.getcwd
rO=os.path
ra=os.chdir
import utm
lE=utm.from_latlon
import csv
lT=csv.writer
import time
rP=time.time
import shutil
lm=shutil.copy
import numpy as np
wt=np.int64
wJ=np.bitwise_xor
wb=np.argmin
wR=np.linspace
wp=np.sqrt
wU=np.nan
wG=np.nanmax
we=np.nanmin
wF=np.flipud
# wH=np.max
# wf=np.min
wu=np.argmax
wo=np.floor
wi=np.argsort
wD=np.flip
ws=np.unique
# wy=np.lG
wv=np.diff
wV=np.concatenate
wW=np.vstack
wq=np.savetxt
wh=np.apply_along_axis
wS=np.equal
wB=np.reshape
wP=np.save
wn=np.hstack
wz=np.pi
wX=np.linalg
wI=np.dot
wO=np.sin
wa=np.cos
wA=np.arctan2
wd=np.random
wN=np.stack
wM=np.isscalar
wQ=np.abs
wK=np.std
wj=np.mean
wg=np.float64
wT=np.copy
wE=np.expand_dims
wx=np.log2
wk=np.nan_to_num
wL=np.divide
# wY=np.sum
wC=np.tile
wm=np.transpose
wl=np.log10
wr=np.power
wc=np.loadtxt
rt=np.zeros
rJ=np.ceil
rb=np.isinf
rR=np.isnan
rp=np.append
rU=np.arange
rG=np.array
import scipy.io as scipyio
import seaborn as sns
lg=sns.heatmap
import matplotlib.pyplot as plt
import matplotlib.image as mplimage

import torch
rF=torch.load
rH=torch.save
rf=torch.topk
ru=torch.Tensor
ro=torch.device
ri=torch.backends
rD=torch.manual_seed
# lo=max
# lu=super
# lf=set
# lH=open
# lF=min
# le=sum
# lG=round
# lU=any

rs=torch.sum
ry=torch.mean
rv=torch.arange
rV=torch.matmul
rW=torch.tensor
rq=torch.utils
rh=torch.nn
rS=torch.optim
rB=torch.cuda
import torch.cuda as cuda
lK=cuda.device_count
lj=cuda.is_available
import torch.optim as optimizer
lN=optimizer.SGD
lM=optimizer.lr_scheduler
lQ=optimizer.Adam
import torch.nn as nn
lI=nn.ModuleList
lO=nn.ReLU
la=nn.Linear
lA=nn.LogSoftmax
ld=nn.Module
import torchvision.transforms as transf
from torch.utils.data import Dataset, DataLoader
import sys

ra(rO.dirname(rO.abspath(__file__)))

# orig_stdout = sys.stdout
# f = open('out.txt', 'w')
# sys.stdout = f

# from helpFunc import *

ra(rI())

TxNum = 64
RxNum = 16

colors = ["g", "b","r", "m","c","y","k", "khaki","blueviolet","olivedrab","chocolate","mediumvioletred","deeppink","mediumblue","aquamarine"]
linestyle = ["solid","dashed","dotted", "dashdot"]
marker=["o","*","+","d"]
def cB(paramDict):
    paramString = ""
    for key,value in paramDict.items():
        paramString += f"{key}_{str(value)}"
    return paramString

def cS(scen_idx, n_beams, norm_type, noise,trainMode, lr_schedule_params):
    paramString = cB(lr_schedule_params)
    return f'scenario{scen_idx}beams{n_beams}norm{norm_type}noise{noise}trainM{trainMode}_{paramString}'


def ch(arr, ax=lX):
    """ Computes min-max normalization of array <arr>. """
    return (arr - arr.min(axis=ax)) / (arr.max(axis=ax) - arr.min(axis=ax))
    
def cq(arr):
    """ Computes min-max normalization of array <arr>. """
    return (arr - arr.min()) / (arr.max() - arr.min())
    
def cW(arr, arr_min, arr_max, ax = 1):
    """ Computes  min max normalization given the bound useful for slicing of the region"""
    return (arr - arr_min) / (arr_max - arr_min)

def cV(lat_long):
    """ Assumes lat and long along row. Returns same row vec/matrix on 
    cartesian coords."""
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = lE(lat_long[:,0], lat_long[:,1])
    return wN((x,y), axis=1)


def cv(pos, noise_variance_in_m=1):
    
    n_samples = pos.shape[0]
    
    # Get noise in xy coordinates
    dist = wd.normal(0, noise_variance_in_m, 2 * n_samples)
    # ang = np.random.uniform(0, 2*np.pi, n_samples)
    xy_noise = wN((dist[:n_samples], dist[n_samples:]), axis=1)
    
    # # Get position in xy coordinates
    # x, y, zn, zl = utm.from_latlon(pos[:,0], pos[:,1])
    # xy_pos = np.stack((x,y), axis=1)

    # Apply noise to position and return conversion to lat_long coordinates
    xy_pos_noise = pos + xy_noise
    
    # lat,long = utm.to_latlon(xy_pos_noise[:,0], xy_pos_noise[:,1], zn, zl)
    # pos_with_noise = np.stack((lat,long), axis=1)
    # return pos_with_noise
    return xy_pos_noise


def cy(pos1, pos2, norm_type, params=lX):
    """
    Normalizations:
    1- lat&long -> min_max
    2- lat&long -> min_max (north-aware)
    3- lat&long -> cartesian -> min_max    
    4- lat&long -> cartesian -> rotation -> min_max
    5- lat&long -> cartesian -> distance & angle -> center angle at 90º -> 
                -> normalize to 0-1: divide distance and a angle by max values
    6- restrict region, this will then normalize to the original region (so orig_min, orig_max) not
    based on the current region's bound
    
    Advantages of each normalization:
    1- the simplest...
    2- better for transfer learning (TL)
    3- reference is the earth axis
    4- common reference: the BS. --> Should improve in Transfer Learning!
    5- same as 4, but using polar coordinates with more 
       transferable normalizations (not min_max)
    """
    
    if norm_type == 1:
        pos_norm = ch(pos2)
    
    if norm_type == 2:
        # Check where the BS is and flip axis
        pos_norm = ch(pos2)
        
        avg_pos2 = wj(pos2, axis=0)
        
        if pos1[0,0] > avg_pos2[0]:
            pos_norm[:,0] = 1 - pos_norm[:,0]
        if pos1[0,1] > avg_pos2[1]:
            pos_norm[:,1] = 1 - pos_norm[:,1]
        
    if norm_type == 3:
        pos_norm = ch(cV(pos2))
        
        
    if norm_type  == 4:
        # For relative positions, rotate axis, and min_max it.
        pos2_cart = cV(pos2)
        pos_bs_cart = cV(pos1)
        avg_pos2 = wj(pos2_cart, axis=0)
        
        vect_bs_to_ue = avg_pos2 - pos_bs_cart
        
        theta = wA(vect_bs_to_ue[1], vect_bs_to_ue[0])
        rot_matrix = rG([[ wa(theta), wO(theta)],
                               [-wO(theta), wa(theta)]])
        pos_transformed =  wI(rot_matrix, pos2.T).T
        pos_norm = ch(pos_transformed)
        
    
    if norm_type == 5:
        pos2_cart = cV(pos2)
        pos_bs_cart = cV(pos1)
        pos_diff = pos2_cart - pos_bs_cart
        
        # get distances and angle from the transformed position
        dist = wX.norm(pos_diff, axis=1)
        ang = wA(pos_diff[:,1], pos_diff[:,0])
        
        # Normalize distance + normalize and offset angle
        dist_norm = dist / lo(dist)
        
        # 1- Get the angle to the average position
        avg_pos = wj(pos_diff, axis=0)
        avg_pos_ang = wA(avg_pos[1], avg_pos[0])
        
        # A small transformation to the angle to avoid having breaks 
        # between -pi and pi
        ang2 = rt(ang.shape)
        for i in lq(lh(ang)):
            ang2[i] = ang[i] if ang[i] > 0 else ang[i] + 2 * wz
        
        avg_pos_ang2 =            avg_pos_ang + 2 * wz if avg_pos_ang < 0 else avg_pos_ang
        
        # 2- Offset angle avg position at 90º
        offset2 = wz/2 - avg_pos_ang2
        ang_final = ang2 + offset2
        
        # MAP VALUES OF 0-PI TO 0-1
        ang_norm = ang_final / wz
        
        pos_norm = wN((dist_norm,ang_norm), axis=1)
    if norm_type == 6:
        pos_norm = cW(pos2, params["arr_min"], params["arr_max"])
    
    return pos_norm

def cs(rss, norm_type, params=lX, excludeCols = lX):
    """
    Normalizations:
    1- lat&long -> min_max
    2- lat&long -> min_max (north-aware)
    3- lat&long -> cartesian -> min_max    
    4- lat&long -> cartesian -> rotation -> min_max
    5- lat&long -> cartesian -> distance & angle -> center angle at 90º -> 
                -> normalize to 0-1: divide distance and a angle by max values
    
    Advantages of each normalization:
    1- the simplest...
    2- better for transfer learning (TL)
    3- reference is the earth axis
    4- common reference: the BS. --> Should improve in Transfer Learning!
    5- same as 4, but using polar coordinates with more 
       transferable normalizations (not min_max)
       
    excludeCols,
    
    if = -1 means exclude the last
    if = -k means exclue the last k columns
    """
    
    if excludeCols is not lX:
        rss_temp = rss[:, excludeCols:]
        rss = rss[:, :excludeCols]
    if norm_type == 1:
        rss_norm = ch(rss)
    elif norm_type == 2:
        rss_norm = cq(rss)
    elif norm_type == 6:
        rss_norm = cW(rss, params["arr_min"], params["arr_max"])
    
    if excludeCols is not lX:
        if rss_temp.ndim == 1:
            rss_temp = wE(rss_temp,-1)
        rss_norm = wn((rss_norm,rss_temp))
    return rss_norm
    

# def save_data(split, filename,
#               x_train, x_val, x_test, y_train, y_val, y_test, y_test_pwr):
    
# no y_test_pwr
def cD(split, filename,
              x_train, x_val, x_test, y_train, y_val, y_test):
    wP(filename + '_x_train', x_train)
    wP(filename + '_y_train', y_train)
    wP(filename + '_x_val', x_val)
    wP(filename + '_y_val', y_val)
    wP(filename + '_x_test', x_test)
    wP(filename + '_y_test', y_test)
    # np.save(filename + '_y_test_pwr', y_test_pwr)
    


class DataFeed(Dataset):
    def __init__(self, x_train, y_train, transform=lX):
        if y_train.ndim == 1:
            all_data = wn((x_train, wB(y_train, (lh(y_train),1) )))
        else:
            all_data = wn((x_train, y_train))

        self.samples = all_data.tolist()
        self.transform = transform
        self.seq_len = all_data.shape[-1]

    def __len__(self):
        return lh( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # pos_data = torch.zeros((self.seq_len,))
        # for i,s in enumerate(sample):
        #     x = s
        #     pos_data[i] = torch.tensor(x, requires_grad=False)
        pos_data = rW(sample, requires_grad=ln)
        return pos_data

# customized loss function

class CustomLoss(ld):
    def __init__(self, beamNums = 5, beamWeight = [1,0,0,0,0],reduction="mean", cuda_device_id="cpu"):
        """
        beamNums: the number of beams used to calculate the loss,
        beamWeight: for each considered beam, calculate the beamweight
        """
        lu(CustomLoss, self).__init__()
        self.beamNums = beamNums
        self.beamWeight = rG(beamWeight)
        self.reduction = reduction
        self.cuda_device_id = cuda_device_id

    def forward(self, output, target):
        # target = torch.LongTensor(target)
        # convert output to logSoftMax
        m = lA(dim=1)
        output_logsoft = m(output).to(self.cuda_device_id)
        loss = rV(output_logsoft[rv(output_logsoft.size(0)).unsqueeze(1),target], rW(self.beamWeight).unsqueeze(1).float().to(self.cuda_device_id))
        if self.reduction == "mean":
            loss = ry(loss.squeeze())
        elif self.reduction == "sum":
            loss = rs(loss.squeeze())
        loss = loss * -1
        return loss
# customized loss function

class CustomLossWideBeam(ld):
    # def __init__(self, beamNums = 16, beamWeight = [1,0,0,0,0],reduction="mean", cuda_device_id="cpu"):
    def __init__(self, beamNums = 16, targetNums = 8, beamWeight = [1,1,1,1,1,1,1,1],reduction="mean", cuda_device_id="cpu"):
        """
        beamNums: the number of beams used to calculate the loss,
        beamWeight: for each considered beam, calculate the beamweight
        """
        lu(CustomLossWideBeam, self).__init__()
        self.beamNums = beamNums
        self.beamWeight = rG(beamWeight)
        self.reduction = reduction
        self.cuda_device_id = cuda_device_id

    def forward(self, output, target):
        # target = torch.LongTensor(target)
        # convert output to logSoftMax
        m = lA(dim=1)
        # rowNum = output.size(0)
        # output_logsoft = m(output).to(self.cuda_device_id)
        # outputTotalBeam = output.size(1)
        # # Convert index vector to binary vector
        # binary_vector = torch.zeros(rowNum,outputTotalBeam)
        # # binary_vector = torch.zeros(outputTotalBeam, device=device)
        # for i in range(rowNum):
        #     binary_vector[i,target[i,:]]= 1
        
        # predicted_binary_vectors = torch.zeros_like(nn_outputs)
        loss = rV(output_logsoft[rv(output_logsoft.size(0)).unsqueeze(1),target], rW(self.beamWeight).unsqueeze(1).float().to(self.cuda_device_id))
        if self.reduction == "mean":
            loss = ry(loss.squeeze())
        elif self.reduction == "sum":
            loss = rs(loss.squeeze())
        loss = loss * -1
        return loss

# Model Definition (Fully connected, ReLu)
class NN_FCN(ld):
    def __init__(self, num_features, num_output, nodes_per_layer, n_layers):
        lu(NN_FCN, self).__init__()
        self.n_layers = n_layers
        
        if n_layers < 2:
            raise lW('A NN must include at least input and output layers.')
        
        self.layer_in = la(num_features, nodes_per_layer)
        if n_layers > 2:
            self.std_layer = la(nodes_per_layer, nodes_per_layer)
        
        self.layer_out = la(nodes_per_layer, num_output)
        self.relu = lO()
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_in(inputs))
        if self.n_layers > 2:
            for n in lq(self.n_layers-2):
                x = self.relu(self.std_layer(x))
        
        x = self.layer_out(x)
        return (x)

# Model Definition (Fully connected, ReLu)
class NN_FCN_deepIA(ld):
    def __init__(self, num_features, num_output=24, nodes_per_layer = [32,64,128,64,32], n_layers=5, cuda_device_id = "cpu"):
        lu(NN_FCN_deepIA, self).__init__()
        self.n_layers = n_layers
        
        if n_layers < 2:
            raise lW('A NN must include at least input and output layers.')
        
        self.layer_in = la(num_features, nodes_per_layer[0])
        self.innerLayers = []
        for layer in lq(1,n_layers):
            
            self.innerLayers.append(la(nodes_per_layer[layer-1], nodes_per_layer[layer]))
            # self.std_layer = nn.Linear(nodes_per_layer, nodes_per_layer)
        # convert to ModuelList
        self.innerLayers = lI(self.innerLayers)
        self.layer_out = la(nodes_per_layer[-1], num_output)
        self.relu = lO()
        self.cuda_device_id = cuda_device_id
        
    def forward(self, inputs):
        # x = self.relu(self.layer_in(inputs)).to(self.cuda_device_id)
        x = self.relu(self.layer_in(inputs))
        if self.n_layers > 2:
            for n in lq(self.n_layers-1):
                # x = self.relu(self.innerLayers[n](x)).to(self.cuda_device_id)
                x = self.relu(self.innerLayers[n](x))
        
        x = self.layer_out(x).to(self.cuda_device_id)
        return (x)

def co(x_train, y_train, z_train, x_val, y_val, z_val, backup_folder, 
              num_epochs, model, train_batch_size, lr, decay_L2, 
              top_stats=[1,2,3,5], rnd_seed=0, 
              fixed_GPU=lV, backup_best_model=lV, 
              save_all_pred_labels=lV, make_plots=lV,
              print_top_stats_per_epoch=ln, beamNums = 5,beamWeight=[0.5,0.2,0.1,0.1,0.1], saveNameSuffix="", chooseLastEpochTrain = ln,
              lr_start = 0.002, lr_end = 0.1, AddAverage=ln, lr_scheduler="cosineAnneal", lr_scheduleParams=lX, bandwidth=800e6, dropOutage=lV, checkChannel = lV):
    """
    chooseLastEpochTrain: choose the model of the last trained epoch regardless of the best acc
    
    z_train,
    z_val
    store the RSS value, used to get the channel capacity
    
    the training z_train| y_train 
    """
    
    inputFeatureNum = x_train.shape[1]
    returnDict = lv()
    addAverage = ln

    if (inputFeatureNum > 2) and AddAverage:
        addAverage = lV
    # Make dir if doesn'torch exist
    if not rO.exists(backup_folder):
        rX(backup_folder)    
        
    x_train_colNum = x_train.shape[1]
    y_train_colNum = y_train.shape[1]
    z_train_colNum = z_train.shape[1]
    x_val_colNum = x_val.shape[1]
    y_val_colNum = y_val.shape[1]
    z_val_colNum = z_val.shape[1]
    # # Copy the training files
    # try:
    #     shutil.copy(os.path.basename(__file__), backup_folder)
    # except:
    #     try:
    #         # shutil.copy('4-train_test.py', backup_folder)  not found 
    #         shutil.copy('train_test_py', backup_folder)
    #     except:
    #         try:
    #             shutil.copy(os.path.join(os.getcwd(),'train_test_py'), os.path.join(backup_folder,'train_test_func_copy.py'))
    #         except:
    #             print('One can only copy when executed in a terminal.')
    #             input('Press any key to continue without a backup of the code...')
    
    # Save CSV with the predicted labels of each epoch?
    if save_all_pred_labels:
        save_directory = rO.join(backup_folder,'saved_analysis_files')
        if not rO.exists(save_directory):
            rX(save_directory)
    
    # Save the model of best epoch
    if backup_best_model:
        checkpoint_directory = rO.join(backup_folder, 'model_checkpoint')    
        if not rO.exists(checkpoint_directory):
            rX(checkpoint_directory)
            
        net_name = rO.join(checkpoint_directory, 'nn_beam_pred' + saveNameSuffix)
    
    # Before the loaders since they will shuffle data!
    rD(rnd_seed)
    ri.cudnn.deterministic = lV
    
    # Data Input to Torch
    proc_pipe = transf.Compose([transf.ToTensor()])
    
    # Create torch DataLoaders
    if y_train.ndim == 1:
        wE(y_train,-1)
    # stack that with z
    y_train = wn((y_train, z_train))
    if y_val.ndim == 1:
        wE(y_val,-1)
    y_val = wn((y_val, z_val))
    train_loader = DataLoader(DataFeed(x_train, y_train, transform=proc_pipe),
                              batch_size=train_batch_size, #num_workers=8,
                              shuffle=lV)
    
    val_loader =  DataLoader(DataFeed(x_val, y_val, transform=proc_pipe),
                             batch_size=train_batch_size, #num_workers=8,
                             shuffle=ln)
    # shuffle False to match the y_val (in case of save pred labels=True)
    
    # We're collecting top1, top2, top3, and top5 statistics: in top_stats
    n_top_stats = lh(top_stats)
    # n_val_samples = len(y_val)
    n_val_samples = x_val.shape[0]
    n_labels = model.layer_out.out_features
    
    # Select GPU
    if fixed_GPU:
        cuda_device_id = ro("cuda:0" if rB.is_available() else "cpu")
    else: # pick random GPU
        gpu_id = wd.choice(rU(0,lK()))
        cuda_device_id = ro(f"cuda:{gpu_id}") 
       
    
    # Per batch running losses
    running_training_loss = rt(num_epochs)
    running_val_loss = rt(num_epochs)
    
    # Accuracies
    running_accs = rt((num_epochs, n_top_stats))
    running_acc_stds = rt((num_epochs, n_top_stats))
    channelCap_accs = rt((num_epochs, n_top_stats))
    channelCap_acc_refs = rt((num_epochs, n_top_stats))
    best_accs = rt(n_top_stats)
    best_accs[0] = -1
    best_epoch = -1
    
    # All labels predicted in the test set
    # all_pred_labels = np.zeros((num_epochs, n_val_samples, train_batch_size, beamNums))
    all_pred_labels = rt((train_batch_size, beamNums))
    # all_test_labels = np.zeros((num_epochs, n_val_samples))
        
    # Model Training
    t_0 = rP()
    
    # For reproducibility
    # cuda not available
    # with cuda.device(cuda_device_id):
    if lV:
        
        # Build the network:
        # net = model.cuda()
        net = model.to(cuda_device_id)


        #  Optimization parameters:
        # criterion = nn.CrossEntropyLoss()
        criterion = CustomLoss(beamNums=beamNums, beamWeight=beamWeight, cuda_device_id=cuda_device_id)
        # opt = optimizer.Adam(net.parameters(), lr=lr, weight_decay=decay_L2)
        if (lr_scheduler == "cosineAnneal"):
            opt = lQ(net.parameters(), lr=lr_scheduleParams["opt_lr_init"], weight_decay=decay_L2)
            # follow paper
            # # opt = optimizer.AdaBound(net.parameters(), lr=lr, weight_decay=decay_L2)
            # LR_sch = optimizer.lr_scheduler.LinearLR(opt,start_factor=0.01,total_iters=10)
            # LR_sch = optimizer.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0 = 5, T_mult = 1, eta_min = 2e-3)
            LR_sch = lM.CosineAnnealingWarmRestarts(opt,T_0 = lr_scheduleParams["T_0"], T_mult = lr_scheduleParams["T_mult"], eta_min = lr_scheduleParams["eta_min"])
        if (lr_scheduler == "CyclicLR"):
            if lr_scheduleParams["optim"] == "SGD":
                opt = lN(net.parameters(), lr=lr_scheduleParams["max_lr"], momentum=lr_scheduleParams["optim_momentum"])
                # follow paper
                # # opt = optimizer.AdaBound(net.parameters(), lr=lr, weight_decay=decay_L2)
                # LR_sch = optimizer.lr_scheduler.LinearLR(opt,start_factor=0.01,total_iters=10)
                # LR_sch = optimizer.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0 = 5, T_mult = 1, eta_min = 2e-3)
                LR_sch = lM.CyclicLR(opt,base_lr = lr_scheduleParams["base_lr"], max_lr=lr_scheduleParams["max_lr"], step_size_up = lr_scheduleParams["step_size_up"], mode=lr_scheduleParams["mode"]) 
        if (lr_scheduler == "MultiStepLR"):
            if lr_scheduleParams["optim"] == "Adam":
                opt = lQ(net.parameters(), lr=lr_scheduleParams["lr"], weight_decay=lr_scheduleParams["weight_decay"])
                LR_sch = lM.MultiStepLR(opt, lr_scheduleParams["T"], gamma=lr_scheduleParams["gamma"])
                
        # final_value = 0.2
        # init_value = 0.02
        # number_in_epoch = num_epochs
        # update_step = (final_value / init_value) ** (1 / number_in_epoch)
        # lr = init_value
        # LR_sch = optimizer.lr_scheduler.StepLR(opt, step_size=2, gamma = 0.8)
        # LR_sch = optimizer.lr_scheduler.LinearLR(opt,start_factor=0.01,total_iters=10)
        # LR_sch = optimizer.lr_scheduler.MultiStepLR(opt, [20,40], gamma=0.2)
        # "Decays the learning rate of each parameter group by 
        #  gamma once the number of epoch reaches one of the milestones."
        
        # Converges slower, less accurate, but is more robust 
        # (i.e. less variability across runs)
        # opt = optimizer.AdamW(net.parameters(), lr=lr, weight_decay=decay_L2)
        # LR_sch = optimizer.lr_scheduler.ReduceLROnPlateau(opt, 'min')
        for epoch in lq(num_epochs):
            lS(f'========== Epoch No. {epoch+1: >2} ==========')
            t_1 = rP()
            
            # Cummulative training/test validation losses
            training_cum_loss = 0
            val_cum_loss = 0
            # print current LR
            lS("Epoch %d: lr = %.4f"%(epoch, opt.param_groups[0]["lr"]))
            # Data aspect: X is in first (2) positions, Label is on last (3rd)
            for tr_count, y in ly(train_loader):
                net.train()
                # x = y[:, :-beamNums].type(torch.Tensor).to(cuda_device_id)
                x = y[:, :x_train_colNum].type(ru).to(cuda_device_id)
                # if (epoch == 0) and (x.shape[1] > 2):
                #     addAverage = True
                # save the average
                x_averNew = ry(x, dim=0)
                if tr_count == 0:
                    x_averOld = x_averNew
                else:
                    x_averOld = x_averOld * (tr_count)/(tr_count+1) + x_averNew * 1/(tr_count+1)


                # label = y[:, -beamNums:].long().to(cuda_device_id)
                label = y[:, x_train_colNum:(x_train_colNum + y_train_colNum)].long().to(cuda_device_id)
                opt.zero_grad()
                out = net.forward(x)
                batch_loss = criterion(out, label)
                batch_loss.backward()
                opt.step()                    
                training_cum_loss += batch_loss.item()
            
            # Each batch loss is the average loss of each sample in it. 
            # Avg. over batches to obtain the per sample training loss avg.
            running_training_loss[epoch] = training_cum_loss / (tr_count + 1)
            
            lS('Start validation')
            
            # List of best 5 Predicted Beams for each test sample
            total_hits = rt((lh(val_loader), n_top_stats))
            total_hits_batchMean = rt((lh(val_loader), n_top_stats))
            channelEffi = rt((lh(val_loader), n_top_stats))
            channelEffi_ref = rt((lh(val_loader), n_top_stats))
            
            x_averOld = x_averOld.unsqueeze(dim=0)
            n_val_loader = lh(val_loader)
            val_cum_loss_seq = []
            for idx, data in ly(val_loader):
                net.eval()
                # x = data[:, 1:4].type(torch.Tensor).to(cuda_device_id)
                # x = data[:, :(inputFeatureNum)].type(torch.Tensor).to(cuda_device_id)
                x = data[:, :x_val_colNum].type(ru).to(cuda_device_id)
                if addAverage:
                    # currently use static average
                    x[:, 2:] = x_averOld[:,2:]
                # x = data[:, :-1].type(torch.Tensor).to(cuda_device_id)
                # the last 5 labels
                # label = data[:, -(beamNums):].long().to(cuda_device_id)
                label = data[:, x_val_colNum:(x_val_colNum + y_val_colNum)].long().to(cuda_device_id)
                opt.zero_grad()
                out = net.forward(x)
                # here only one, choose -5 because this corresponds to the correct best beam
                val_cum_loss_temp = criterion(out, label.to(cuda_device_id)).item()
                # val_cum_loss_temp = criterion(out, label[:,-beamNums:].to(cuda_device_id)).item()
                val_cum_loss += val_cum_loss_temp
                val_cum_loss_seq.append(val_cum_loss_temp)
                
                label = label.cpu().numpy()
                
                _, predLabel = rf(out, beamNums, dim=1)
                # all_pred_labels[epoch,idx] = predLabel.to(cuda_device_id).numpy()
                all_pred_labels = predLabel.cpu().numpy()
                    # torch.argsort(out, dim=1, descending=True).cpu().numpy()[0]
                # get the capacity
                if checkChannel:
                    RSS_val = data[:, -z_val_colNum:].cpu().numpy()
                    # extract the reference rss
                    # # RSS_ref = RSS_val[np.arange(RSS_val.shape[0])[:,None], label]
                    # channelCap_ref = getChannelCapacityFromBeamIndices(RSS_val, label, beamNum=beamNums, bandwidth=bandwidth)
                    # # channelCap_ref = topK_rssToChannel(RSS_ref,bandwidth = bandwidth)
                    # # RSS_pred = RSS_val[np.arange(RSS_val.shape[0])[:,None], all_pred_labels[:,:beamNums]]
                    # channelCap_pred = getChannelCapacityFromBeamIndices(RSS_val, all_pred_labels[:,:beamNums], beamNum=beamNums, bandwidth =bandwidth)
                    # # channelCap_pred = topK_rssToChannel(RSS_pred, bandwidth=bandwidth)
                    # # If the best beam is in the topX, then +1 hit for that batch
                    # channelEffi[idx,:] = np.mean(getChannelEffiQuotient(channelCap_pred, channelCap_ref,drop=dropOutage),axis=0)
                    # channelEffi[idx,:] = np.mean(getChannelEffiFromBeamIndices(RSS_val, all_pred_labels[:,:beamNums], label, beamNums, bandwidth, dropOutage), axis=0)
                    # channelEffi[idx,:] = getChannelEffiFromBeamIndices(RSS_val, all_pred_labels[:,:beamNums], label, beamNums, bandwidth, dropOutage, calculateMean=True)
                    channelEffiReturnDict = cM(RSS_val, all_pred_labels[:,:beamNums], label, beamNums, bandwidth, dropOutage, calculateMean=lV,quotientMethod=2)
                    channelEffi[idx,:] = channelEffiReturnDict["channelEffi"]
                    channelEffi_ref[idx,:] = channelEffiReturnDict["channelEffi_ref"]
                for i in lq(n_top_stats):
                    # hit = (set(all_pred_labels[epoch, idx, :top_stats[i]]) == set(label[:top_stats[i]]))
                    # convert each row to set
                    # equalSetCount = np.equal(np.apply_along_axis(set, 1, all_pred_labels[epoch,idx,:,:top_stats[i]]), np.apply_along_axis(set,1,label[:,:top_stats[i]])).sum()
                    # note the wide beam label might not be useful 
                    equalSetCount = wS(wh(lf, 1, all_pred_labels[:,:top_stats[i]]), wh(lf,1,label[:,:top_stats[i]])).sum()
                    # total_hits[idx,i] += equalSetCount
                    total_hits[idx,i] = equalSetCount
                    total_hits_batchMean[idx, i] = equalSetCount / data.shape[0]
                    # calculate the proportion to best channel capacity
                    # channelEffi[idx,i] = np.mean(np.divide(channelCap_pred, channelCap_ref),axis=0)
                    # hit = np.any(all_pred_labels[epoch, idx, :top_stats[i]] == label)
                    # total_hits[i] += 1 if hit else 0
            # compute the channel capacity differences
            
            # Average the number of correct guesses (over the total samples)
            running_accs[epoch,:] = total_hits.sum(axis = 0) / n_val_samples
            running_acc_stds[epoch,:] = wK(total_hits_batchMean,axis = 0)
            # it is possible that channelEffi still contains nan, because some epoch gives zero array (all deleted by purify data), so purify again

            running_val_loss[epoch] = val_cum_loss / (idx + 1)
            # print val loss variance
            lS(f'Mean batch running valLoss: {np.mean(val_cum_loss_seq)} ')
            lS(f'Std valLoss: {np.std(val_cum_loss_seq)}')
            lS(f'Total Loss: {val_cum_loss}')
            if print_top_stats_per_epoch:
                for i in lq(n_top_stats):
                    lS(f'Average Top-{top_stats[i]} accuracy '
                          f'{running_accs[epoch,i]*100:.2f}')
                    lS(f"std:{np.std(total_hits[:,i])}")
            
            if checkChannel:
                channelEffi = cw(channelEffi, drop=lV)
                channelEffi_ref = cw(channelEffi_ref, drop = lV)
                channelCap_accs[epoch,:] = wj(channelEffi, axis=0)
                channelCap_acc_refs[epoch,:] = wj(channelEffi_ref, axis=0)
                # # Gather avg. loss of each test sample
                # running_val_loss[epoch] = val_cum_loss / n_val_samples
                # should mimic the way training loss is calculated
                lS(f'Mean batch running channelEffi: {channelCap_accs[epoch,:]} ')
                lS(f'Std valLoss: {np.std(channelEffi, axis=0)}')
            # print(f'Total Loss: {val_cum_loss}')
            # if print_top_stats_per_epoch:
            #     for i in range(n_top_stats):
            #         print(f'Average Top-{top_stats[i]} accuracy '
            #               f'{running_accs[epoch,i]*100:.2f}')
            #         print(f"std:{np.std(total_hits[:,i])}")
            
            
            # Check if current accuracy of top-1 beam surpasses the best so far
            if (not chooseLastEpochTrain) and (running_accs[epoch, 0] > best_accs[0]):
                lS("NEW BEST!")
                if backup_best_model:
                    rH(net.state_dict(), net_name)
                best_accs[:] = running_accs[epoch, :]
                best_epoch = epoch + 1
                
            lS(f'Curr (top-1) accuracy: {running_accs[epoch, 0]*100:2.2f}%')
            lS(f'Best (top-1) accuracy: {best_accs[0]*100:2.2f}%')
            

            # Take a learning step
            LR_sch.step()
                
            # lr *= update_step
            # opt.param_groups[0]["lr"] = lr
            # With ReduceLROnPlateau: LR_sch.step(running_val_loss[epoch])
            
            lS(f'Time taken for epoch {epoch+1}: {time.time() - t_1:.2f} s.')

        if chooseLastEpochTrain:
            # save the model        
            if backup_best_model:
                rH(net.state_dict(), net_name)
            best_accs[:] = running_accs[-1, :]
                # best_epoch = e

                
    # Write all predicted beams, for each sample to a CSV
    if save_all_pred_labels:
        lS("Saving the predicted value in a csv file")  
        for epoch in lq(num_epochs):
            predicted_csv_name = f"pred_beams_epoch_{epoch+1}.csv"
            csv_path = rO.join(save_directory, predicted_csv_name)
            with lH(csv_path, "w", newline="") as f:
                writer = lT(f)
                writer.writerows(ls(y_val, all_pred_labels[epoch]))
    
    lS('--------------------------------------------')
    lS(f'Total time taken for training: {(time.time() - t_0):.2f} s.')
    
    lS(f'Best Epoch: {best_epoch}')
    lS('Best Validation Results:')
    for i in lq(n_top_stats):
        lS(f'\tAverage Top-{top_stats[i]} validation accuracy '
              f'{best_accs[i]*100:.2f}')
          
    # Save what was best epoch
    with lH(rO.join(backup_folder, f'best_epoch={best_epoch}.txt'), 'w'):
        pass
    
    # Save best accuracies
    wq(rO.join(backup_folder, 'best_val_accs.txt'), 
               best_accs * 100, fmt='%.2f')
    if chooseLastEpochTrain:
        # also save the last epoch val_accus

        wq(rO.join(backup_folder, 'lastEpoch_val_accs.txt'), 
               best_accs * 100, fmt='%.2f')
    
    if make_plots:
        # Plot Top-1, Top-2, Top-3 validation accuracies across epochs
        epochs = rU(1, num_epochs+1)
        
        # plt.figure()
        # plt.plot(epochs, running_accs[:,0]*100, 'g*-', lw=2.0, label='Top-1 Accuracy')
        # plt.plot(epochs, running_accs[:,1]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
        # plt.plot(epochs, running_accs[:,2]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
        # plt.plot(epochs, running_accs[:,3]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
        # plt.xlabel('Number of Epochs')
        # plt.ylabel('Validation Accuracy [%]')
        # plt.legend()
        # plt.grid()
        cF(wm(running_accs * 100), backup_folder, "position_beam_val_acc.png", ylabel='Validation Accuracy [%]', top_beams=[1,2,3,4,5], title="Validation Accuracy",xlabel='Number of Epochs', xVals=epochs)
        
        cF(wm(running_acc_stds * 100), backup_folder, "position_beam_val_acc_std.png", ylabel='validation std[%]', top_beams=[1,2,3,4,5], title="Validation Accuracy Std",xlabel='Number of Epochs', xVals=epochs)
        
        # plt.savefig(os.path.join(backup_folder, 'position_beam_val_acc.pdf'))
        # plt.savefig(os.path.join(backup_folder, 'position_beam_val_acc.png'))
        plt.close("all")
    
        # Plot Training vs Validation loss across epochs
        plt.figure()
        plt.plot(epochs, running_training_loss, 'g*-', lw=2.0, label='Training Loss')
        plt.plot(epochs, running_val_loss, 'b*-', lw=2.0, label='Validation Loss')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Average Loss per sample')
        plt.legend()
        plt.grid()
        # plt.savefig(os.path.join(backup_folder, 'training_vs_validation_loss.pdf'))
        plt.savefig(rO.join(backup_folder, 'training_vs_validation_loss.png'))
        plt.close("all")

        # save the running_accs
        wq(rO.join(backup_folder, "running_accs.csv"),running_accs * 100,delimiter=",",fmt='%.2f')
        lS("saved runing acs at: " + rO.join(backup_folder, "running_accs.csv"))
        
        wq(rO.join(backup_folder, "running_acc_stds.csv"),running_acc_stds,delimiter=",")

        # plot the Achievable Rate
        if checkChannel:
            
            cF(wm(channelCap_accs * 100), backup_folder, 'position_beam_val_channelEffi.png', ylabel='Achievable Rate [%]', top_beams=top_stats, title="Achievable Rate", xlabel="Number of Epochs",xVals=epochs)
            
            cF(wm(channelCap_accs * 100), backup_folder, 'position_beam_val_channelEffi_withRef.png', ylabel='Achievable Rate [%]', top_beams=[1,3,5], title="Achievable Rate", xlabel="Number of Epochs",xVals=epochs, plotRef = lV, refacc=wm(channelCap_acc_refs * 100), selectRows=[0,2,4])
            
            
            cF(wm(channelCap_accs * 100), backup_folder, 'position_beam_val_channelEffi_withRef_total.png', ylabel='Achievable Rate [%]', top_beams=top_stats, title="Achievable Rate", xlabel="Number of Epochs",xVals=epochs, plotRef = lV, refacc=wm(channelCap_acc_refs * 100))
            
            # plt.figure()
            # plt.plot(epochs, channelCap_accs[:,0]*100, 'g*-', lw=2.0, label='Top-1 Accuracy')
            # plt.plot(epochs, channelCap_accs[:,1]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
            # plt.plot(epochs, channelCap_accs[:,2]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
            # plt.plot(epochs, channelCap_accs[:,3]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
            # plt.xlabel('Number of Epochs')
            # plt.ylabel('Achievable Rate [%]')
            # plt.legend()
            # plt.grid()
            # # plt.savefig(os.path.join(backup_folder, 'position_beam_val_acc.pdf'))
            # plt.savefig(os.path.join(backup_folder, 'position_beam_val_channelEffi.png'))
            # plt.close("all")
        
            # # Plot Training vs Validation loss across epochs
            # plt.figure()
            # plt.plot(epochs, running_training_loss, 'g*-', lw=2.0, label='Training Loss')
            # plt.plot(epochs, running_val_loss, 'b*-', lw=2.0, label='Validation Loss')
            # plt.xlabel('Number of Epochs')
            # plt.ylabel('Average Loss per sample')
            # plt.legend()
            # plt.grid()
            # # plt.savefig(os.path.join(backup_folder, 'training_vs_validation_loss.pdf'))
            # plt.savefig(os.path.join(backup_folder, 'training_vs_validation_loss.png'))
            # plt.close("all")

            # save the channelCap_accs
            wq(rO.join(backup_folder, "channelCap_accs.csv"),channelCap_accs,delimiter=",")
            lS("saved runing acs at: " + rO.join(backup_folder, "channelCap_accs.csv"))
        
    
    # save the path to dictionary
    returnDict["modelPath"] = net_name
    returnDict["bestAcc"] = best_accs[0] * 100
    returnDict["x_aver"] = x_averOld
    returnDict["position_beam_val_acc_path_png"] = rO.join(backup_folder, 'position_beam_val_acc.png')
    returnDict["position_beam_val_acc_std_path_png"] = rO.join(backup_folder, 'position_beam_val_acc_std.png')
    # returnDict["position_beam_val_acc_path_pdf"] = os.path.join(backup_folder, 'position_beam_val_acc.pdf')
    returnDict["training_vs_validation_loss_png"] = rO.join(backup_folder, 'training_vs_validation_loss.png')
    
    if checkChannel:
        returnDict["position_beam_val_channelEffi_png"] = rO.join(backup_folder,"position_beam_val_channelEffi.png")
        
        returnDict["position_beam_val_channelEffi_withRef_png"] = rO.join(backup_folder,"position_beam_val_channelEffi_withRef.png")
        returnDict["position_beam_val_channelEffi_withRef_total_png"] = rO.join(backup_folder,"position_beam_val_channelEffi_withRef_total.png")
    
    # returnDict["training_vs_validation_loss_pdf"] = os.path.join(backup_folder, 'training_vs_validation_loss.pdf')

    return returnDict



def cu(x_test, model, batch_size = 1024, beamNums = 5):
    
    # Data Input to Torch
    proc_pipe = transf.Compose([transf.ToTensor()])
    n_test_samples = x_test.shape[0]
    test_loader = DataLoader(DataFeed(x_test, rt(n_test_samples), 
                                      transform=proc_pipe),
                             batch_size=batch_size, shuffle=ln)
    # shuffle = False is important! This way, test labels have the same order.
    
    n_labels = model.layer_out.out_features
    # all_pred_labels = np.zeros((n_test_samples, n_labels))
    all_pred_labels = rt((n_test_samples, beamNums))
    
    cuda_device_id = ro("cuda:0" if rB.is_available() else "cpu")
    # with cuda.device(cuda_device_id):
    if lV:
        net = model.to(cuda_device_id)
        net.eval() # a evaluation switch: turns off Dropouts, BatchNorms, ...
           
        for idx, data in ly(test_loader):
            # x = data[:, :2].type(torch.Tensor).to(cuda_device_id)
            x = data[:, :-1].type(ru).to(cuda_device_id)
            totalDataNum = data.shape[0]
            out = net.forward(x)
            
            _, predLabel = rf(out, beamNums, dim=1)
            # Sort labels according to activation strength
            all_pred_labels[rU(idx * batch_size, idx*batch_size + totalDataNum)] = predLabel.cpu().numpy()

            # all_pred_labels[idx] = \
            #     torch.argsort(out, dim=1, descending=True).to(cuda_device_id).numpy()[0]
        

    return all_pred_labels.astype(lz)


# find lr
def cf(model, x_train, y_train, savePath, train_batch_size = 1024, init_value=1e-8, final_value=10.0, fixed_GPU = lV,
beamNums = 5, beamWeight = [1,0,0,0,0]):
    lr = init_value
    # opt = optimizer.Adam(net.parameters(), lr=2, weight_decay=decay_L2)
    
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []

    # construct train_loader
    proc_pipe = transf.Compose([transf.ToTensor()])
    train_loader = DataLoader(DataFeed(x_train, y_train, transform=proc_pipe),
                            batch_size=train_batch_size, #num_workers=8,
                            shuffle=lV)
    number_in_epoch = lh(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    

        # Select GPU
    if fixed_GPU:
        cuda_device_id = ro("cuda:0" if rB.is_available() else "cpu")
    else: # pick random GPU
        gpu_id = wd.choice(rU(0,lK()))
        cuda_device_id = ro(f"cuda:{gpu_id}") 
    
    net = model.to(cuda_device_id)   
    opt = lQ(net.parameters(), lr=2)
    opt.param_groups[0]["lr"] = lr

    criterion = CustomLoss(beamNums=beamNums, beamWeight=beamWeight, cuda_device_id=cuda_device_id)
    for tr_count, y in ly(train_loader):
        batch_num += 1
        x = y[:,:-beamNums].type(ru).to(cuda_device_id)

        label = y[:, -beamNums:].long().to(cuda_device_id)
        opt.zero_grad()
        out = net.forward(x)
        batch_loss = criterion(out, label)
        # Crash out if loss explodes

        if batch_num > 1 and batch_loss > 10 * best_loss:
            if(lh(log_lrs) > 20):
                return log_lrs[10:-5], losses[10:-5]
            else:
                return log_lrs, losses

        # Record the best loss

        if batch_loss < best_loss or batch_num == 1:
            best_loss = batch_loss

        # Store the values
        losses.append(batch_loss.item())
        log_lrs.append((lr))

        # Do the backward pass and optimize
        batch_loss.backward()
        opt.step()                    

        # Update the lr for the next step and store

        lr *= update_step
        opt.param_groups[0]["lr"] = lr
    plt.figure()
    plt.plot(log_lrs, losses)
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.savefig(rO.join(savePath, "lr_find.png"))
    lS("saved at " + rO.join(savePath, "lr_find.png"))
    plt.close("all")
    if(lh(log_lrs) > 20):
        return log_lrs[10:-5], losses[10:-5]
    else:
        return log_lrs, losses
    
def cH(x_train, x_val, x_test, y_train, y_val, y_test):
    
    n_train_samples = lh(y_train)
    n_val_samples = lh(y_val)
    n_test_samples = lh(y_test)
    n_samples = n_train_samples + n_val_samples + n_test_samples
    
    lS(f'Samples in Training: {n_train_samples}\n'
          f'Samples in Validation: {n_val_samples}\n'
          f'Samples in Test: {n_test_samples}\n'
          f'Total samples: {n_samples}')
    
    lS(f'x_train is {x_train.shape}\n'
          f'y_train is {y_train.shape}\n'
          f'x_val   is {x_val.shape}\n'
          f'y_val   is {y_val.shape}\n'
          f'x_test  is {x_test.shape}\n'
          f'x_test  is {y_test.shape}')
    
def cF(testacc,saved_folder, saveFigName, ylabel, top_beams=[1,2,3,4,5],title="", xlabel="epoch", selectRows=[], labels=[], xVals = [], plotRef=ln, refacc=lX):
    """
    plot aggregate result
    
    title:
        if ==  ""
            will use ylabel as title
            
    note top_beams start counting from 1
            
    testacc :
        shape:   topNumBeams * epochNums
    selectRows = []
        will plot  only the selected row number
    """                    
    plt.figure()
    if lh(selectRows) == 0:
        selectRows = lP(lq(testacc.shape[0]))
    if lh(labels) == 0:
        labels = [f'Top-{i} Accuracy' for i in top_beams]
    if lh(xVals) == 0:
        xVals = rU(1, testacc.shape[-1]+1)
    plotCount = 0
    if plotRef:
        if refacc.ndim ==1:
            # expand dimensions
            refacc = wC(wE(refacc, -1), (1,testacc.shape[-1]))
    for i in top_beams:
        i = i-1
        if i in selectRows:
            plt.plot(xVals, testacc[i,:], color=colors[i], linestyle="solid", marker='*', lw=2.0, label=labels[plotCount])
            if plotRef:
                plt.plot(xVals, refacc[i,:], color=colors[i], linestyle="dashed", label="ref-" + labels[plotCount])
            # plt.plot(np.arange(1, testacc.shape[-1]+1), testacc[1,:]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
            # plt.plot(np.arange(1, testacc.shape[-1]+1), testacc[2,:]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
            # plt.plot(np.arange(1, testacc.shape[-1]+1), testacc[3,:]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
            plotCount += 1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title == "":
        title = ylabel
    plt.title(title)
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.tight_layout()
    plt.grid()
    plt.savefig(rO.join(saved_folder, saveFigName))
    plt.show()
    
    
def ce(algos, testAcc_sfs, testAcc_msb,channelEffi_sfs_acc,channelEffi_msb_acc, top_runs_folder, testNetWrapperReturnDict, top_beams, extraNumCols, trainMode):
    # plt.figure()
    if lh(algos) == 2:
        cF(wW((testAcc_sfs[[0,-1],:], testAcc_msb[[0,-1],:])),top_runs_folder,"accM.png","Accuracy [%]", title="Accuracy", xlabel="M", selectRows = [0,1,2,3], labels=["sfs top1", "sfs top5", "msb top1", "msb top5"])
        
        # plot channel capacity
        cF(wW((channelEffi_sfs_acc[[0,-1],:], channelEffi_msb_acc[[0,-1],:])),top_runs_folder,"channelM.png","Achievable Rate [%]", title="Achievable Rate", xlabel="M", selectRows = [0,1,2,3], labels=["sfs top1", "sfs top5","msb top1", "msb top5"])
        
        # plot channel capacity
        refAcc = testNetWrapperReturnDict["channelEffi_ref"] * 100
        refAcc = wC(wE(refAcc,-1), (1, channelEffi_sfs_acc.shape[-1]))


        cF(wW((channelEffi_sfs_acc[[0,-1],:], channelEffi_msb_acc[[0,-1],:], refAcc[[0,-1],:])),top_runs_folder,"channelM_withRef.png","Achievable Rate [%]", title="Achievable Rate", xlabel="M", selectRows = [0,1,2,3,4,5], labels=["sfs top1", "sfs top5","msb top1", "msb top5", "ref top1", "ref top5"], top_beams=[1,2,3,4,5,6])                
        # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[0,:],"b*-")
        # plt.plot(np.arange(1, testAcc_msb.shape[-1] + 1), testAcc_msb[0,:],"m*-")
        
        # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[4,:],"g*-")
        # plt.plot(np.arange(1, testAcc_msb.shape[-1] + 1), testAcc_msb[4,:],"c*-")
        # plt.xlabel("M")
        # plt.ylabel("Accuracy [%]")
        # plt.title("Accuracy")
        # plt.legend(["sfs top1","msb top1", "sfs top5", "msb top5"])
        # plt.savefig(os.path.join(top_runs_folder, "accM.png"))
        # plt.show()
    lS("sfs accuracy:")
    lS(testAcc_sfs)
    lS("msb accuracy:")
    lS(testAcc_msb)
    wq(rO.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
    
    wq(rO.join(top_runs_folder, "msb_accuracy.txt"), testAcc_msb, fmt='%.2f')
    lS("acc saved at: " + rO.join(top_runs_folder, "sfs_accuracy.txt"))
    
    wq(rO.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
    # separte top-k
    if "SFS" in algos:
        cF(testAcc_sfs, top_runs_folder, "accM_sfs.png", "Accuracy [%]",top_beams,"Accuracy", "M")
        
        cF(channelEffi_sfs_acc, top_runs_folder, "accM_channelEffi_sfs_select.png", "Achievable Rate", top_beams, "", "M", plotRef = lV, refacc=wm(testNetWrapperReturnDict["channelEffi_ref"] * 100), selectRows=[0,2,4])
        
        cF(channelEffi_sfs_acc, top_runs_folder, "accM_channelEffi_sfs.png", "Achievable Rate", top_beams, "", "M", plotRef = lV, refacc=wm(testNetWrapperReturnDict["channelEffi_ref"] * 100))
        
        # plt.figure()
            
        # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[0,:]*100, 'g*-', lw=2.0, label='Top-1 Accuracy')
        # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[1,:]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
        # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[2,:]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
        # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[3,:]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
        # plt.xlabel("M")
        # plt.ylabel("Accuracy [%]")
        # plt.title("Accuracy")
        # plt.legend()
        # plt.savefig(os.path.join(top_runs_folder, "accM_sfs.png"))
        # plt.show()
        
    # msb
    if "MSB" in algos:
        cF(testAcc_msb, top_runs_folder, "accM_msb.png", "Accuracy [%]",top_beams,"Accuracy", "M")
        cF(channelEffi_msb_acc, top_runs_folder, "accM_channelEffi_msb_select.png", "Achievable Rate", top_beams, "", "M",plotRef = lV, refacc=wm(testNetWrapperReturnDict["channelEffi_ref"] * 100), selectRows=[0,2,4])
        cF(channelEffi_msb_acc, top_runs_folder, "accM_channelEffi_msb.png", "Achievable Rate", top_beams, "", "M",plotRef = lV, refacc=wm(testNetWrapperReturnDict["channelEffi_ref"] * 100))
        
        # plt.figure()
            
        # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[0,:]*100, 'g*-', lw=2.0, label='Top-1 Accuracy')
        # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[1,:]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
        # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[2,:]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
        # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[3,:]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
        # plt.xlabel("M")
        # plt.ylabel("Accuracy [%]")
        # plt.title("Accuracy")
        # plt.legend()
        # plt.savefig(os.path.join(top_runs_folder, "accM_msb.png"))
        # plt.show()
        
    
    lS("sfs accuracy:")
    lS(testAcc_sfs)
    lS("msb accuracy:")
    lS(testAcc_msb)
    wq(rO.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
    
    wq(rO.join(top_runs_folder, "msb_accuracy.txt"), testAcc_msb, fmt='%.2f')
    lS("acc saved at: " + rO.join(top_runs_folder, "sfs_accuracy.txt"))
    



    lS("sfs channelEffi:")
    lS(testAcc_sfs)
    lS("msb channelEffi:")
    lS(testAcc_msb)
    wq(rO.join(top_runs_folder, "sfs_channelEffi.txt"), channelEffi_sfs_acc, fmt='%.2f')
    
    wq(rO.join(top_runs_folder, "msb_channelEffi.txt"), channelEffi_msb_acc, fmt='%.2f')
    wq(rO.join(top_runs_folder, "channelEffi_ref.txt"), testNetWrapperReturnDict["channelEffi_ref"] * 100, fmt='%.2f')
    lS("channel effi saved at: " + rO.join(top_runs_folder, "sfs_channelEffi.txt"))
    
    # np.savetxt(os.path.join(top_runs_folder, "sfs_channelEffi_txs.txt"), channelEffi_txs_sfs_acc, fmt='%.2f')
    
    # np.savetxt(os.path.join(top_runs_folder, "msb_channelEffi_txs.txt"), channelEffi_txs_msb_acc, fmt='%.2f')
    
    # savetxt cannot save 3d, so save binary
    if (extraNumCols > 0) and (trainMode > 4):
        wP(rO.join(top_runs_folder, "sfs_channelEffi_txs.npy"), channelEffi_txs_sfs_acc)
        
        
        wP(rO.join(top_runs_folder, "msb_channelEffi_txs.npy"), channelEffi_txs_msb_acc)
        
        wP(rO.join(top_runs_folder, "sfs_accuracy_txs.npy"), testAcc_txs_sfs)
        
        wP(rO.join(top_runs_folder, "msb_accuracy_txs.npy"), testAcc_txs_msb)          
            
def cG(totalDataMatrix, bm_method, trainMode, n_beams, bestBeamNum, norm_type, pos_params, rss_params,beamRSSstartIndex, beamRSSendIndex):
    """generate data needed, given trai

    Args:
        totalDataMatrix (_type_): the dataframe that stores the total data
        bm_method (_type_): string
        
        beamRSSstartIndex: index where beamRSS starts
        
        beamRSS_endIndex: the index where beam RSS ends so the beam RSS is expected to
        have
        
        so to get beamRSS
            totalDataMatrix[beamRSSstartIndex: (beamRSS_endIndex+1)]
    """
    
    returnDict = lv()
    totalDataMatrix = cw(totalDataMatrix, defaultVal=-174)
    mapPos = totalDataMatrix.iloc[:,[0,1]].to_numpy()
    allBeamRSS_orig = totalDataMatrix.iloc[:, beamRSSstartIndex:(beamRSSendIndex+1)].to_numpy()
    extraNumCols = allBeamRSS_orig.shape[1] - n_beams
    if extraNumCols > 0:
        excludeCols = -extraNumCols
        
    # normallize RSS
    allBeamRSS = cs(allBeamRSS_orig, norm_type, params=rss_params, excludeCols=excludeCols)

    bestBeamIndices = totalDataMatrix.iloc[:,2: (2 + bestBeamNum)].to_numpy()
    posBestBeamIndices = wV([mapPos, totalDataMatrix.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
    
    if trainMode == 7:
        txPosCols = totalDataMatrix[["tx_pos_x", "tx_pos_y"]].to_numpy()
        mapPos = wV((mapPos, txPosCols),axis=-1)
    if trainMode > 4:
        tx_id = totalDataMatrix["tx_id"].to_numpy()
            
    if bm_method == "twostage-test":
        # expect to see the totalDataMatrix_test
        pass
        
def cU(run_folder,n_samples,trainingInput, outputReference, posBestBeamIndices,totalRSS,n_beams, num_epochs, train_batch_size, 
                        lr, decay_L2, run_idx = 1, train_val_test_split = [60,20,20], force_seed = -1, plotData=ln, top_beams = [1,2,3,5], findLR = ln, findLR_train_batch_size=1024, findLR_init_value = 2e-3, findLR_final_value = 2e-1, fixed_GPU = lV, totalBeams = 5, beamWeight = [1,0,0,0,0], saveNameSuffix="", chooseLastEpochTrain = lV, AddAverage = ln, lr_scheduler="cosineAnneal", lr_scheduleParams=lX, modelType = "deepIA", modelParams = lX, bandwidth=800e6, dropOutage=lV, loadModel = ln, loadModelPath = "", testModelDict=lX, plotPosBestBeam = lV,checkChannel = lV):
    """
    wrapper for the train net

    Args:
        run_folder (_type_): folder to save test result
        
        trainingInput (matrix):  input data
            positionCode:
                normalized beam RSS
        
        outputReference(matrix):  output reference (will  compare with NN output to calcualte losses)
            positionCode
                best beam indices (each row like 5 top1,top2...)
        
        posBestBeamIndices(matrix): used for plotting, each row
            xPos, yPos, bestBeam (top1)
            
        totalRSS: stores the RSS corresponding to each trainingInputRow
            used for channelcapacity calculation
    """

    # Create folder for the run
    if not rO.isdir(run_folder):
        rz(run_folder)
    
    # Shuffle Data (fixed, for reproducibility)
    if force_seed >= 0:
        wd.seed(force_seed)
    else:
        wd.seed(run_idx)
    sample_shuffle_list = wd.permutation(n_samples)
    
    # Select sample indices for each set
    first_test_sample = lz((1-train_val_test_split[2]) / 100 * n_samples)
    train_val_samples = sample_shuffle_list[:first_test_sample]
    test_samples = sample_shuffle_list[first_test_sample:]

    train_ratio = wY(train_val_test_split[:2]) / 100
    first_val_sample = lz(train_val_test_split[1] / 100 * 
                        lh(train_val_samples) / train_ratio)
    val_samples = train_val_samples[:first_val_sample]
    train_samples = train_val_samples[first_val_sample:]
    
    x_train = trainingInput[train_samples]
    # y_val = outputReference[val_samples,:]
    y_train = outputReference[train_samples,:]
    z_train = totalRSS[train_samples,:]
    # y_train = beam_data[train_samples]
    x_val = trainingInput[val_samples]
    # y_val = beamPos[val_samples]
    y_val = outputReference[val_samples,:]
    z_val = totalRSS[val_samples,:]
    
    # y_val = beam_data[val_samples]
    x_test = trainingInput[test_samples]
    # y_test = beamPos[test_samples]
    y_test = outputReference[test_samples,:]
    z_test = totalRSS[test_samples,:]
    
    # y_test = beam_data[test_samples]
    # y_test_pwr = beam_pwrs[test_samples]
    if plotData:                
        # plot data 
        lS("Plot data train, val, test")
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=lV,layout= "constrained")
        ax1.scatter(posBestBeamIndices[train_samples,0], posBestBeamIndices[train_samples,1], c=posBestBeamIndices[train_samples,2], vmin=1, vmax = 24,cmap='jet')
        asp1 = wv(ax1.get_xlim())[0] / wv(ax1.get_ylim())[0]
        ax1.set_aspect(asp1)

        ax1.set_ylabel('y normalized', labelpad=15)
        ax1.set_xlabel('x normalized', fontsize=13)

        ax2.scatter(posBestBeamIndices[val_samples,0], posBestBeamIndices[val_samples,1], c=posBestBeamIndices[val_samples,2], vmin=1, vmax = 24,cmap='jet')
        asp2 = wv(ax2.get_xlim())[0] / wv(ax2.get_ylim())[0]
        ax2.set_aspect(asp2)

        ax2.set_xlabel('x normalized', fontsize=13)

        ax3.scatter(posBestBeamIndices[test_samples,0], posBestBeamIndices[test_samples,1], c=posBestBeamIndices[test_samples,2], vmin=1, vmax = 24,cmap='jet')
        asp3 = wv(ax3.get_xlim())[0] / wv(ax3.get_ylim())[0]
        ax3.set_aspect(asp3)
        ax3.set_xlabel('x normalized', fontsize=13)

        totalName = "train_val_test_dataPlot"
        plt.savefig(rO.join(run_folder, totalName + '.png'))
        # plt.savefig(plt_name + '.eps')
        # plt.savefig(os.path.join(run_folder, totalName + '.pdf'))
        # plt.savefig(os.path.join(run_folder, totalName + '.svg'))
        plt.close("all")
    cH(x_train, x_val, x_test, 
                                y_train, y_val, y_test)
    
    # filename = os.path.join(run_folder, 
    #                         scen_str + f'_{n_beams}_{norm_type}')
    
    # filename = os.path.join(run_folder,f'_{n_beams}_{norm_type}')
    
    # SAVE_DATA_FILENAME = filename
    
    # save_data(train_val_test_split, filename, 
    #             x_train, x_val, x_test, y_train, y_val, y_test)
    # save_data(train_val_test_split, filename, 
    #                x_train, x_val, x_test, y_train, y_val, y_test, 
    #                y_test_pwr)

# ---------------------- Phase 5: Train & Test ------------------------    

    # Useful little variables
    n_test_samples = lh(x_test)
    n_top_stats = lh(top_beams)
    
    # Variables for compatibility (when not all predictors are used)
    n_bins, bin_size, prediction_per_bin = lX, lX, lX
    trained_model = lX

# 1- KNN
    # if quantize_input:
    #     n_bins = 200
    #     x_train = np.round(x_train / (1/n_bins)) * (1/n_bins)

    # Create Model
    # model = NN_FCN(x_train.shape[-1], n_beams, nodes_per_layer, layers)  
    cuda_device_id = ro("cuda:0" if rB.is_available() else "cpu")
    # print("in top k cuda device: ")
    # print(cuda_device_id)

    if modelType == "deepIA":
        model = NN_FCN_deepIA(x_train.shape[-1], n_beams, cuda_device_id=cuda_device_id)  
    elif modelType == "pos":
        model = NN_FCN(x_train.shape[-1], n_beams, modelParams["nodes_per_layer"], modelParams["layers"])  
    elif modelType == "wideBeam":
        model = NN_FCN(x_train.shape[-1], n_beams, modelParams["nodes_per_layer"], modelParams["layers"])  

    if loadModel:
        model.load_state_dict(rF(loadModelPath))
        testacc = rt((n_top_stats,0))
        channelEffi_acc = rt((n_top_stats,0))
        
        # initial test
        testacc, channelEffi_acc,testNetWrapperReturnDict= cp(rO.join(run_folder,"initialTest"), x_train, y_train, x_test,y_test, z_test, totalRSS, n_beams, loadModelPath, test_samples, testModelDict["scen_idx"], posBestBeamIndices,testModelDict["knn_num_neighbors"], testModelDict["lt_predition_per_bin"], bin_size,n_bins,testacc,channelEffi_acc, extractTxNum=testModelDict["extractTxNum"], extractRxNum=testModelDict["extractRxNum"], totalBeams = totalBeams, N = 10000,predTxNum = testModelDict["predTxNum"], predRxNum = testModelDict["predRxNum"], n_top_stats = n_top_stats, top_beams = top_beams, modelType = modelType, modelParams = modelParams, bandwidth=bandwidth,dropOutage=lV,checkChannel=checkChannel)
                        
    # Train model on provided data + Write results to run_folder
    # trained_model_path = \
    #     train_net(x_train, y_train, x_val, y_val, run_folder, 
    #                    num_epochs, model, train_batch_size, 
    #                    lr, decay_L2, top_stats=top_beams, rnd_seed=0,
    #                    fixed_GPU=False, backup_best_model=True, 
    #                    save_all_pred_labels=False, make_plots=True)

    # trained_model_path = \
    #     train_net(x_train, y_train, x_val, y_val, run_folder, 
    #                    num_epochs, model, train_batch_size, 
    #                    lr, decay_L2, top_stats=top_beams, rnd_seed=0,
    #                    fixed_GPU=True, backup_best_model=True, 
    #                    save_all_pred_labels=False, make_plots=True,beamWeight=[1,0,0,0,0])        
    # trained_model_path, x_aver = \
    if findLR:
        cf(model, x_train, y_train, run_folder, train_batch_size = findLR_train_batch_size, init_value=findLR_init_value, final_value=findLR_final_value, fixed_GPU =  fixed_GPU,
beamNums = totalBeams, beamWeight = beamWeight)
        returnDict = lv() #only for consistency of the function interface
    else:
        returnDict = co(x_train, y_train, z_train, x_val, y_val,z_val, run_folder, 
                        num_epochs, model, train_batch_size, 
                        lr, decay_L2, top_stats=top_beams, rnd_seed=0,
                        fixed_GPU=fixed_GPU, backup_best_model=lV, 
                        save_all_pred_labels=ln, make_plots=lV, beamWeight = beamWeight, saveNameSuffix=saveNameSuffix, chooseLastEpochTrain = chooseLastEpochTrain, AddAverage=AddAverage, lr_scheduler=lr_scheduler, lr_scheduleParams=lr_scheduleParams,bandwidth=bandwidth, dropOutage=dropOutage,checkChannel = checkChannel)
                            
    returnDict["x_test"] = x_test
    returnDict["x_testRowIndex"] = test_samples
    returnDict["x_train"] = x_train
    returnDict["x_val"] = x_val
    
    returnDict["y_test"] = y_test
    returnDict["y_train"] = y_train
    returnDict["y_val"] = y_val

    returnDict["z_train"] = z_train
    returnDict["z_val"] = z_val
    returnDict["z_test"] = z_test
    
    returnDict["model"] = model
    returnDict["cuda_device_id"] = cuda_device_id
    return returnDict

def cp(runs_folder, x_train, y_train, x_test,y_test, z_test, totalRSS, n_beams, bestModelPath, x_testRowIndex, scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testacc,channelEffi_acc, extractTxNum=1, extractRxNum=1, totalBeams = 5, N = 10000,predTxNum = 64, predRxNum = 16, n_top_stats = 5, top_beams = [1,2,3,5], modelType = "deepIA", modelParams = lX, bandwidth=800e6,dropOutage=lV, plotPredFig=lV,testBatchSize=1024, checkChannel = lV, simpleOutput=ln):
    """
    
    test_net wrapper
    Args:
        x_train (_type_): _description_
        n_beams (_type_): _description_
        cuda_device_id (_type_): _description_
        
        totalRSS: 
            stores the RSS data
            
        bestModelPath (_type_): _description_
        
        posBestBeamIndices: used in prediction_map, 
            xpos,ypos, bestBeamIndex
        
        allBeamRSS (_type_): _description_
        extractTxNum (int, optional): used to extract data from provided datamatrix Defaults to 1.
        extractRxNum (int, optional): _description_. Defaults to 1.
        N : num of pts generated to predict
        predTxNum = 24  : used to indicate the total available tx index for prediction map plot
        
        predRxNum = 1
        
        testacc:
            (totalBeams, N)  where N is the size of the rxSet max

        dropOutage:
            default True, if set, when calculating channelEffi, will skip those lines where channel_ref has 0            
        simpleOutput:
            default False, if set, then will only get the pred_beams and return
    """
    
    # make run folder in case not exist
    if not rO.exists(runs_folder):
        rX(runs_folder)
        
    cuda_device_id = ro("cuda:0" if rB.is_available() else "cpu")
    
    if modelType == "deepIA":
        trained_model = NN_FCN_deepIA(x_train.shape[-1], n_beams, cuda_device_id=cuda_device_id)  
    elif modelType == "pos":
        trained_model = NN_FCN(x_train.shape[-1], n_beams, modelParams["nodes_per_layer"], modelParams["layers"])  
    # trained_model = NN_FCN_deepIA(x_train.shape[-1], n_beams, cuda_device_id=cuda_device_id)

    trained_model.load_state_dict(rF(bestModelPath))
    
    # # set the whole data set
    # beamRSS_whole = getBeamSetRSS(beamSet, allBeamRSS, txNum = extractTxNum, rxNum = extractRxNum)
    # Test model on test data
    # pred_beams = test_net(x_test, trained_model, totalBeams)
    n_test_samples = x_test.shape[0]
# prediction_map(N, ai_strategy, n_beams, scen_idx, run_folder,
#                    x_train, y_train, x_test, n, prediction_per_bin, 
#                    bin_size, n_bins, trained_nn):

    # prediction_map(N, "NN", x_aver, n_beams, scen_idx, run_folder, x_train, y_train, x_test, knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, trained_model)
    if posBestBeamIndices is lX:
        plotPredFig = ln
    pred_beams,pred_returnDict = rx(N, "NN", n_beams, scen_idx, runs_folder, x_train, x_testRowIndex, x_test, posBestBeamIndices,totalRSS, knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, trained_model, txNum = predTxNum, rxNum = predRxNum, predBeamNum=totalBeams, bandwidth=bandwidth,plotFig=plotPredFig,testBatchSize=testBatchSize)
    if simpleOutput:
        return pred_beams, lX, lX
    # prediction_map(N, "NN", n_beams, scen_idx, runs_folder, x_train, y_train, beamRSS, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, trained_model, txNum = predTxNum, rxNum = predRxNum)
    # # testacc = []
    # Get top-1, top-2, top-3 and top-5 accuracies
    total_hits = rt(n_top_stats)
    channelEffi = rt(n_top_stats)
    
    # get the channel capacity
        # get the capacity
    # extract the reference rss
    # RSS_ref = z_test[np.arange(z_test.shape[0])[:,None], y_test]
    # channelCap_ref = topK_rssToChannel(RSS_ref,bandwidth = bandwidth)
    # channelCap_ref = getChannelCapacityFromBeamIndices(z_test,y_test,beamNum=totalBeams,bandwidth=bandwidth)
    # channelCap_pred = getChannelCapacityFromBeamIndices(z_test, pred_beams[:,:top_beams[-1]], beamNum=totalBeams, bandwidth=bandwidth)
    
    # # RSS_pred = z_test[np.arange(z_test.shape[0])[:,None], pred_beams[:,:top_beams[-1]]]
    # # channelCap_pred = topK_rssToChannel(RSS_pred, bandwidth=bandwidth)
    # # For each test sample, count times where true beam is in top 1,2,3,5
    # # If the best beam is in the topX, then +1 hit for that batch
    # channelEffi = getChannelEffiQuotient(channelCap_pred, channelCap_ref, defaultVal=-1, drop=True)
    # channelEffi = np.divide(channelCap_pred, channelCap_ref)
    # channelEffi = getChannelEffiFromBeamIndices(z_test,pred_beams[:,:top_beams[-1]], y_test,beamNum=totalBeams,bandwidth=bandwidth,dropOutage=dropOutage, calculateMean=True)
    if checkChannel:
        channelEffiReturnDict = cM(z_test,pred_beams[:,:top_beams[-1]], y_test,beamNum=totalBeams,bandwidth=bandwidth,dropOutage=dropOutage, calculateMean=lV, quotientMethod=2)
        channelEffi = channelEffiReturnDict["channelEffi"]
        channelEffi_ref = channelEffiReturnDict["channelEffi_ref"]
        # save also the pred_beams
        channelEffiReturnDict["pred_beams"] = pred_beams
        channelEffiReturnDict["pred_rate"] = pred_returnDict["pred_rate"]
        channelEffiReturnDict["ref_rate"] = pred_returnDict["ref_rate"] 
    else:
        channelEffiReturnDict = lX
    

    
    # channelEffi = np.mean(channelEffi, axis=0)
    for i in lq(n_top_stats):
        # hit = (set(all_pred_labels[epoch, idx, :top_stats[i]]) == set(label[:top_stats[i]]))
        # convert each row to set
        # equalSetCount = np.equal(np.apply_along_axis(set, 1, all_pred_labels[epoch,idx,:,:top_stats[i]]), np.apply_along_axis(set,1,label[:,:top_stats[i]])).sum()
        equalSetCount = wS(wh(lf, 1, pred_beams[:,:top_beams[i]]), wh(lf,1,y_test[:,:top_beams[i]])).sum()
        total_hits[i] += equalSetCount
    # for i in range(n_test_samples):
    #     for j in range(n_top_stats):
    #         # hit = np.any(pred_beams[i][:top_beams[j]] == y_test[i])
    #         # different test model
    #         hit = (set(pred_beams[i][:top_beams[j]]) == set(y_test[i,:top_beams[j]]))
    #         total_hits[j] += 1 if hit else 0
    
    # Average the number of correct guesses (over the total samples)
    acc = wy(total_hits / n_test_samples, 4)
    
    # print(f'{ai_strategy} Results:')
    lS("*************************")
    lS('NN Results:')
    for i in lq(n_top_stats):
        lS(f'\tAverage Top-{top_beams[i]} accuracy {acc[i]*100:.2f}')
        if checkChannel:
            lS(f'\tAverage Top-{top_beams[i]} channel Effi {channelEffi[i]*100:.2f}')
            
    # Save Test acc to file
    wq(rO.join(runs_folder, 'test_accs.txt'), 
            acc * 100, fmt='%.2f')
    if checkChannel:
        wq(rO.join(runs_folder, 'channel_effi.txt'), 
                channelEffi * 100, fmt='%.2f')
        wq(rO.join(runs_folder, 'channel_effi_ref.txt'), 
                channelEffi_ref * 100, fmt='%.2f')
        
        channelEffi_acc = wn( (channelEffi_acc,wE(channelEffi * 100,-1)))
    # channelEffi_acc = np.hstack( (channelEffi_acc,np.expand_dims(channelEffi_ref * 100,-1)))
    # Save accumluated Test acc to file
    # testacc.append(acc[0]*100)
    testacc = wn((testacc, wE(acc*100,-1)))
    wq(rO.join(runs_folder, 'test_accs_accum.txt'), 
            testacc, fmt='%.2f')
    lS("testacc: ")
    lS(testacc)
    if checkChannel:
        wq(rO.join(runs_folder, 'channel_effi_accum.txt'), 
                channelEffi_acc, fmt='%.2f')
        lS("channelEffi_acc: ")
        lS(channelEffi_acc)
    plt.close("all")
    return testacc, channelEffi_acc, channelEffiReturnDict
    
def cR(runs_folder, x_train, y_train, x_test,y_test, n_beams, bestModelPath, x_testRowIndex, xDataTotalDataFrame, yDataTotalDataFrame,origDataDataFrame,totalRSSDataFrame, scen_idx, knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testacc, testacc_txs,channelEffi_acc, channelEffiacc_txs, extractTxNum=1, extractRxNum=1, totalBeams = 5, N = 10000,predTxNum = 64, predRxNum = 16, n_top_stats = 5, top_beams = [1,2,3,5], modelType = "deepIA", modelParams = lX, bandwidth=800e6, dropOutage=lV,plotPredFig=lV, testBatchSize=1024, checkChannel=lV,simpleOutput=ln):
    """
    
    test_net wrapper
    Args:
        x_train (_type_): _description_
        n_beams (_type_): _description_
        cuda_device_id (_type_): _description_
        bestModelPath (_type_): _description_
        beamSet (_type_): _description_
        allBeamRSS (_type_): _description_
        extractTxNum (int, optional): used to extract data from provided datamatrix Defaults to 1.
        extractRxNum (int, optional): _description_. Defaults to 1.
        N : num of pts generated to predict
        predTxNum = 24  : used to indicate the total available tx index for prediction map plot
        
        origDataDataFrame:
            used to extract best beam indices used for prediction map
            xpos, ypos, bestbeam
        totalRSSDataFrame:
            used to obtain RSSdata for each (tx,rx) pair
        
        predRxNum = 1
    """
    
    cuda_device_id = ro("cuda:0" if rB.is_available() else "cpu")
    
    if modelType == "deepIA":
        trained_model = NN_FCN_deepIA(x_train.shape[-1], n_beams, cuda_device_id=cuda_device_id)  
    elif modelType == "pos":
        trained_model = NN_FCN(x_train.shape[-1], n_beams, modelParams["nodes_per_layer"], modelParams["layers"])  
    # trained_model = NN_FCN_deepIA(x_train.shape[-1], n_beams, cuda_device_id=cuda_device_id)

    trained_model.load_state_dict(rF(bestModelPath))
    
    # # set the whole data set
    # beamRSS_whole = getBeamSetRSS(beamSet, allBeamRSS, txNum = extractTxNum, rxNum = extractRxNum)
    # Test model on test data
    # pred_beams = test_net(x_test, trained_model, totalBeams)
    n_test_samples = x_test.shape[0]
    
    origDataDataFrame["test"] = origDataDataFrame.shape[0] * [0]
    xDataTotalDataFrame["test"] = xDataTotalDataFrame.shape[0] * [0]
    xDataTotalDataFrame["initialIndex"] = xDataTotalDataFrame.index.values
    totalRSSData = totalRSSDataFrame.to_numpy()
    totalRSSDataFrame["test"] = totalRSSDataFrame.shape[0]*[0]
    # # get the index
    # origDataTestIndex = origDataDataFrame.columns.get_loc("test")
    # xDataTotalTestIndex = xDataTotalDataFrame.columns.get_loc("test")
    # mark all row which corresponds to xtest as 1
    origDataDataFrame.loc[x_testRowIndex,"test"] = 1
    xDataTotalDataFrame.loc[x_testRowIndex,"test"] = 1
    totalRSSDataFrame.loc[x_testRowIndex,"test"] = 1
    
    xDataGp = xDataTotalDataFrame.groupby("tx_id")
    xDataGpElems = [xDataGp.get_group(x) for x in xDataGp.groups]
    txPosNum = lh(xDataGpElems)
    origDataGp = origDataDataFrame.groupby("tx_id")
    origDataGpElems = [origDataGp.get_group(x) for x in origDataGp.groups]
    
    totalRSSGp = totalRSSDataFrame.groupby("tx_id")
    totalRSSGpElems = [totalRSSGp.get_group(x) for x in totalRSSGp.groups]
    # add one column to testacc_txs
    # testacc_txs = np.hstack((testacc_txs, np.expand_dims(np.zeros(testacc_txs.shape[0]),-1)))
    testacc_txs = wV((testacc_txs, wE(rt((testacc_txs.shape[0], testacc_txs.shape[1])),-1)), axis=-1)
    channelEffiacc_txs = wV((channelEffiacc_txs, wE(rt((channelEffiacc_txs.shape[0], channelEffiacc_txs.shape[1])),-1)), axis=-1)
    if simpleOutput:
        pred_beams = cu(x_test, trained_nn, beamNums = predBeamNum, batch_size = testBatchSize)
        return pred_beams, lX,lX,lX
        
    for tx_index, xData, origData, totalRSS in ls(rU(txPosNum), xDataGpElems, origDataGpElems, totalRSSGpElems):
        # testacc_txs_currTx = testacc_txs[tx_index,:,:]
        # channelEffiacc_txs_currTx = channelEffiacc_txs[tx_index,:,:]
        # reset the index since the xData keeps the original index
        xData.reset_index(drop=lV, inplace=lV)
        origData.reset_index(drop=lV, inplace=lV)
        # get the xtest
        x_testRowIndexPerGp = xData.index[xData["test"] == 1].tolist()
        origTestRowIndex = origData.index[origData["test"] == 1].tolist()
        # # test if these two indices are the same
        assert(x_testRowIndexPerGp == origTestRowIndex)
        x_test_sub = xData.iloc[x_testRowIndexPerGp]
        x_testRowInOrigIndex = x_test_sub["initialIndex"].tolist()
        # because have added "test" and "initialIndex" columns, now throw that two columns away
        
        # also throw away the tx_id columns!
        x_test_sub = x_test_sub.iloc[:, :-3].to_numpy()
        y_test_sub = yDataTotalDataFrame.iloc[x_testRowInOrigIndex].to_numpy()
        # totalRSS added one column "test" delete that
        z_test_sub = totalRSS.iloc[x_testRowIndexPerGp, :-1].to_numpy()
        # note that the origData is only to provide reference, in particular for the plotting, the left figure will use predicted beam from its own prediction and do not use any of the origData, in particular x_test_sub and origData could have different rows, but this is only good for pos code, for deepIA, the xtest does not support position data, so rule of thumb, use xtest to feed NN, but use origData to extract position infor
        posBestBeamIndices = origData.to_numpy()
        # posBestBeamIndices = origData.iloc[origTestRowIndex].to_numpy()
# prediction_map(N, ai_strategy, n_beams, scen_idx, run_folder,
#                    x_train, y_train, x_test_sub, n, prediction_per_bin, 
#                    bin_size, n_bins, trained_nn):

    # prediction_map(N, "NN", x_aver, n_beams, scen_idx, run_folder, x_train, y_train, x_test_sub, knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, trained_model)
        sub_runs_folder = rO.join(runs_folder, "tx" + li(tx_index))
        if not rO.exists(sub_runs_folder):
            rX(sub_runs_folder)
        if posBestBeamIndices is lX:
            plotPredFig = ln
        pred_beams,pred_returnDict = rx(N, "NN", n_beams, scen_idx, sub_runs_folder, x_train, x_testRowIndexPerGp, x_test_sub, posBestBeamIndices, totalRSS.to_numpy(),knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, trained_model, txNum = predTxNum, rxNum = predRxNum, predBeamNum=totalBeams, bandwidth=bandwidth, plotFig=plotPredFig)
        
        # prediction_map(N, "NN", n_beams, scen_idx, runs_folder, x_train, y_train, beamRSS, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, trained_model, txNum = predTxNum, rxNum = predRxNum)
        # # testacc = []
        # Get top-1, top-2, top-3 and top-5 accuracies
        total_hits = rt(n_top_stats)
        # For each test sample, count times where true beam is in top 1,2,3,5
        # If the best beam is in the topX, then +1 hit for that batch
        # channelCap_ref = getChannelCapacityFromBeamIndices(z_test_sub,y_test_sub[:,:totalBeams],beamNum=totalBeams,bandwidth=bandwidth)
        # channelCap_pred = getChannelCapacityFromBeamIndices(z_test_sub, pred_beams[:,:totalBeams], beamNum=totalBeams, bandwidth=bandwidth)
        # channelEffi = getChannelEffiQuotient(channelCap_pred, channelCap_ref)
        # channelEffi = np.divide(channelCap_pred, channelCap_ref)
        # channelEffi = getChannelEffiFromBeamIndices(z_test_sub, pred_beams[:,:totalBeams], y_test_sub[:,:totalBeams], beamNum=totalBeams, bandwidth=bandwidth,dropOutage=dropOutage)
        if checkChannel:
            channelEffiReturnDict = cM(z_test_sub, pred_beams[:,:totalBeams], y_test_sub[:,:totalBeams], beamNum=totalBeams, bandwidth=bandwidth,dropOutage=dropOutage, calculateMean=lV, quotientMethod=2)
            channelEffi = channelEffiReturnDict["channelEffi"]
        else :
            channelEffiReturnDict = lX
    # channelEffi_ref = channelEffiReturnDict["channelEffi_ref"]
        # channelEffi = np.mean(channelEffi, axis=0)
        
        for i in lq(n_top_stats):
            # hit = (set(all_pred_labels[epoch, idx, :top_stats[i]]) == set(label[:top_stats[i]]))
            # convert each row to set
            # equalSetCount = np.equal(np.apply_along_axis(set, 1, all_pred_labels[epoch,idx,:,:top_stats[i]]), np.apply_along_axis(set,1,label[:,:top_stats[i]])).sum()
            equalSetCount = wS(wh(lf, 1, pred_beams[:,:top_beams[i]]), wh(lf,1,y_test_sub[:,:top_beams[i]])).sum()
            total_hits[i] += equalSetCount
                
        # for i in range(n_test_samples):
        #     for j in range(n_top_stats):
        #         # hit = np.any(pred_beams[i][:top_beams[j]] == y_test_sub[i])
        #         # different test model
        #         hit = (set(pred_beams[i][:top_beams[j]]) == set(y_test_sub[i,:top_beams[j]]))
        #         total_hits[j] += 1 if hit else 0
        
        # Average the number of correct guesses (over the total samples)
        acc = wy(total_hits / n_test_samples, 4)
        
        # print(f'{ai_strategy} Results:')

        lS("*************************")
        lS('NN Results:')
        for i in lq(n_top_stats):
            lS(f'\tAverage Top-{top_beams[i]} accuracy {acc[i]*100:.2f}')
            lS(f'\tAverage Top-{top_beams[i]} channel Effi {channelEffi[i]*100:.2f}')
                
        # Save Test acc to file
        wq(rO.join(runs_folder, 'test_accs_' + li(tx_index) + ".txt"), 
                acc * 100, fmt='%.2f')
        if checkChannel:
            wq(rO.join(runs_folder, 'channel_effi' + li(tx_index) + ".txt"), 
                    channelEffi * 100, fmt='%.2f')        
            channelEffiacc_txs[tx_index,:,-1] = channelEffi * 100
        # testacc_txs[tx_index,-1] = acc[0]*100
        # txNum, n_top_stats, epochNum
        testacc_txs[tx_index, :, -1] = acc * 100
        # testacc_txs_currTx= np.hstack((testacc_txs_currTx, np.expand_dims(acc*100,-1)))
        # channelEffiacc_txs_currTx = np.hstack( (channelEffiacc_txs_currTx,np.expand_dims(channelEffi * 100,-1)))
        # channelEffiacc_txs[tx_index,-1] = channelEffi[0] * 100
        # Save accumluated Test acc to file
        wq(rO.join(runs_folder, 'test_accs_' + li(tx_index) + '.txt'), 
                testacc_txs[tx_index,:,:], fmt='%.2f')
        if checkChannel:
            wq(rO.join(runs_folder, 'channel_effi_accum_' + li(tx_index) + '.txt'), 
                    channelEffiacc_txs[tx_index,:,:], fmt='%.2f')
            
            lS("channeleffi acc for tx" + li(tx_index))
            lS(channelEffiacc_txs[tx_index,:,:])
        
        lS("testacc for tx" + li(tx_index))
        lS(testacc_txs[tx_index,:,:])
        plt.close("all")
    # test for whole
    # 
    pred_beams = cu(x_test, trained_model, beamNums =totalBeams, batch_size= testBatchSize)
    total_hits = rt(n_top_stats)
    # For each test sample, count times where true beam is in top 1,2,3,5
    # If the best beam is in the topX, then +1 hit for that batch
    # channelCap_ref = getChannelCapacityFromBeamIndices(totalRSSData[x_testRowIndex,:],y_test[:,:totalBeams],beamNum=totalBeams,bandwidth=bandwidth)
    # channelCap_pred = getChannelCapacityFromBeamIndices(totalRSSData[x_testRowIndex,:], pred_beams[:,:totalBeams], beamNum=totalBeams, bandwidth=bandwidth)
    # channelEffi = getChannelEffiQuotient(channelCap_pred, channelCap_ref)
    # channelEffi = getChannelEffiFromBeamIndices(totalRSSData[x_testRowIndex,:], pred_beams[:,:totalBeams], y_test[:,:totalBeams], beamNum=totalBeams, bandwidth=bandwidth,dropOutage=dropOutage,calculateMean=True)
    
    if checkChannel:
        channelEffiReturnDict = cM(totalRSSData[x_testRowIndex,:], pred_beams[:,:totalBeams], y_test[:,:totalBeams], beamNum=totalBeams, bandwidth=bandwidth,dropOutage=dropOutage, calculateMean=lV, quotientMethod=2)
        channelEffi = channelEffiReturnDict["channelEffi"]
    else:
        channelEffiReturnDict = lX
    # channelEffi_ref = channelEffiReturnDict["channelEffi_ref"]
    # 
    # ch
    # channelEffi = np.divide(channelCap_pred, channelCap_ref)
    # channelEffi = np.mean(channelEffi, axis=0)
    
    for i in lq(n_top_stats):
        # hit = (set(all_pred_labels[epoch, idx, :top_stats[i]]) == set(label[:top_stats[i]]))
        # convert each row to set
        # equalSetCount = np.equal(np.apply_along_axis(set, 1, all_pred_labels[epoch,idx,:,:top_stats[i]]), np.apply_along_axis(set,1,label[:,:top_stats[i]])).sum()
        equalSetCount = wS(wh(lf, 1, pred_beams[:,:top_beams[i]]), wh(lf,1,y_test[:,:top_beams[i]])).sum()
        total_hits[i] += equalSetCount
            
    # for i in range(n_test_samples):
    #     for j in range(n_top_stats):
    #         # hit = np.any(pred_beams[i][:top_beams[j]] == y_test[i])
    #         # different test model
    #         hit = (set(pred_beams[i][:top_beams[j]]) == set(y_test[i,:top_beams[j]]))
    #         total_hits[j] += 1 if hit else 0
    
    # Average the number of correct guesses (over the total samples)
    acc = wy(total_hits / n_test_samples, 4)
    
    # print(f'{ai_strategy} Results:')
    lS("*************************")
    lS('NN Results:')
    for i in lq(n_top_stats):
        lS(f'\tAverage Top-{top_beams[i]} accuracy {acc[i]*100:.2f}')
        lS(f'\tAverage Top-{top_beams[i]} channel Effi {channelEffi[i]*100:.2f}')
            
    # Save Test acc to file
    wq(rO.join(runs_folder, 'test_accs' + ".txt"), 
            acc * 100, fmt='%.2f')
    if checkChannel:
        wq(rO.join(runs_folder, 'channel_effi.txt'), channelEffi * 100, fmt='%.2f')
        # save the top1-beam channel effi
        channelEffi_acc = wn( (channelEffi_acc,wE(channelEffi * 100,-1)))
    
    # channelEffi_acc.append(channelEffi_acc[0]*100)
    # # testacc.append(acc[0]*100)
    testacc = wn((testacc, wE(acc * 100,-1)))
    
    # Save accumluated Test acc to file
    wq(rO.join(runs_folder, 'test_accs_accum.txt'), 
            testacc, fmt='%.2f')

    if checkChannel:
        wq(rO.join(runs_folder, 'channel_effi_accum.txt'), 
                channelEffi_acc, fmt='%.2f')    
        lS("channelEffi_acc: ")
        lS(channelEffi_acc)
    
    lS("testacc: ")
    lS(testacc)
        
    plt.close("all")
    
    return testacc, testacc_txs, channelEffi_acc, channelEffiacc_txs,channelEffiReturnDict
    

def cb(path_list):
    """ Joins paths with os.path.join(). """
    n_path_parts = lh(path_list)
    
    if n_path_parts < 2:
        raise lW('Path list must have 2 or more elements to join.')
    
    s = rO.join(path_list[0], path_list[1])
    
    if n_path_parts > 2:
        for path_idx in lq(2, n_path_parts):
            s = rO.join(s, path_list[path_idx])
            
    return s

def cJ(arr):
    """ Returns ordered list based on # of occurences in 1D array. """
    vals, counts = ws(arr, return_counts=lV)
    return vals[wD(wi(counts))]


def ct(pos, bin_size, n_bins):
    # The bin indices will be flattened out
    # 
    # x2
    # ^
    # | d e f
    # | a b c
    # --------> x1
    # 
    # Will be mapped to: a b c d e f
    if pos[0] == 1:
        pos[0] -= 1e-9

    if pos[1] == 1:
        pos[1] -= 1e-9
        
    bin_idx = lz(wo(pos[0] / bin_size[0]) + 
                  1 / bin_size[0] * wo(pos[1] / bin_size[1]))    
    
    return lo(lF(bin_idx, n_bins-1), 0)


    
    
def rc(d, m, s, direction='N'):
    if direction in ['N', 'E']:
        mult = 1
    elif direction in ['S', 'W']:
        mult = -1
    else:
        raise lW('Invalid direction.')
        
    return mult * (d + m/60 + s/3600)


def rw(scen_idx):
    # Load corners and image from Google Earth
    if scen_idx in [1,2]:
        gps_bottom_left = [rc(33,25,14.49, 'N'),
                           rc(111,55,45.06, 'W')]
        gps_top_right   = [rc(33,25,12.35, 'N'),
                           rc(111,55,43.67, 'W')]
    if scen_idx in [3,4]:
        gps_bottom_left = [rc(33,25,4.31, 'N'),
                           rc(111,55,33.85, 'W')]
        gps_top_right   = [rc(33,25, 6.72, 'N'),
                           rc(111,55,35.96, 'W')]
    if scen_idx == 5:
        gps_bottom_left = [rc(33,25,15.62, 'N'), 
                           rc(111,55,45.17, 'W')]
        gps_top_right   = [rc(33,25,13.63, 'N'), 
                           rc(111,55,43.93, 'W')]
    if scen_idx == 6:
        gps_bottom_left = [rc(33,25,36.25, 'N'), 
                           rc(111,55,46.89, 'W')]
        gps_top_right   = [rc(33,25,33.17, 'N'),
                           rc(111,55,44.87, 'W')]
    if scen_idx == 7:
        gps_bottom_left = [rc(33,15,29.88, 'N'), 
                           rc(111,51,32.76, 'W')]
        gps_top_right   = [rc(33,15,31.96, 'N'),  
                           rc(111,51,34.10, 'W')]
    if scen_idx in [8,9]:
        gps_bottom_left = [rc(33,25,10.54, 'N'),
                           rc(111,55,44.62, 'W')]
        gps_top_right   = [rc(33,25,8.62, 'N'), 
                           rc(111,55,43.45, 'W')]

    return (gps_bottom_left, gps_top_right)


def rl(stats, pos1, pos2, pwr1, scen_idx):
    
    n_samples = lh(pwr1)
    n_labels = pwr1.shape[-1]
    beam_labels = wu(pwr1, axis=1)
    
    # 1- Avg SNR, noise and maximum power
    if 1 in stats:
        max_min_pwr_ratio_per_sample = [pwr1[i,beam_labels[i]] / wf(pwr1[i,:]) 
                                        for i in lq(n_samples)]
        avg_clearance_db = 10 * wl(wj(max_min_pwr_ratio_per_sample))
        lS(f"avg SNR = {avg_clearance_db:.2f} dB.")
        lS(f"avg_noise_floor = {np.mean(np.min(pwr1, axis=1)):.4f}")
        lS(f"avg max power = {np.mean(np.max(pwr1, axis=1)):.4f}")
    
    # 2- Avg Distance between BS and UE
    if 2 in stats:
        pos1_cart = cV(pos1)
        pos2_cart = cV(pos2)
        dist = wX.norm(pos2_cart - pos1_cart, axis=1)
        # dist_avg_pos = np.linalg.norm(np.mean(pos2_cart - pos1_cart, axis=0))
        lS(f"avg distance = {np.mean(dist):.2f} m.")
        # print(f"distance of avg position = {dist_avg_pos:.2f} m.")
        # these two are 98% correlated.... not very useful.
    
    # 3- Count how many beams( on avg.) have powers within 70% of the max power
    if 3 in stats: 
        thres = 0.7
        max_power_per_sample = wH(pwr1, axis=1)
        n_beams_within_thres = [wY(pwr1[i] > thres * max_power_per_sample[i])
                                for i in lq(n_samples)]
        avg_n_beams = wj(n_beams_within_thres)
        lS(f"avg # beams within {thres*100:.0f}% of max = {avg_n_beams:.2f} beams.")
    
    # 4- Power noise: 
    # Check previous and next adjacent samples. 
    # If they have the same "best beam", quantify the maximum 
    # variability (largest-smallest)/pwr of curr sample
    if 4 in stats:
        normed_variability = rt(n_samples-2)
        for sample_idx in lq(n_samples):
            if sample_idx not in [0, n_samples-1]:
                # best beam index
                bb = beam_labels[sample_idx]
                vals = wN((pwr1[sample_idx-1, bb],
                                 pwr1[sample_idx,bb],
                                 pwr1[sample_idx+1, bb]))
                normed_variability[sample_idx-1] =                    (lo(vals) - lF(vals)) / pwr1[sample_idx,bb]
                    
        lS(f'Beam power variability {np.mean(normed_variability):.4f}')

    if 5 in stats:
        # plot and save the array of the average power profile
        
        # There are two ways of normalizing the power for this plot. 
        # 1- min max across all data
        # norm_pwr = min_max(pwr1)
        # 2- divide by max of each sample
        norm_pwr = pwr1 / wH(pwr1, axis=1)[:,lX]
        
        max_idxs = wu(pwr1, axis=1)
        
        # aggregate samples in array to average at the end
        acc_pwrs = rt(pwr1.shape) # n_beams
        
        idx_in_middle = lz(wo(n_labels/2))
        
        # center the powers before accumulating
        for sample_idx in lq(n_samples):
            for idx_in_acc in lq(n_labels):
                if idx_in_acc < idx_in_middle: # left part
                    diff = idx_in_middle - idx_in_acc # always positive
                    original_idx = max_idxs[sample_idx] - diff
                    if original_idx < 0:
                        original_idx += 64
                else: # right part
                    diff = idx_in_acc - idx_in_middle
                    original_idx = (max_idxs[sample_idx] + diff) % 63
                acc_pwrs[sample_idx, idx_in_acc] = norm_pwr[sample_idx, original_idx]
        
        pwr_footprint = wj(acc_pwrs, axis=0)
        
        plt.plot(pwr_footprint, 'x-', label=f'scen-{scen_idx}')
        # et_xticks([idx_in_middle])
        # ax.set_xticklabels([idx_in_middle])
        # plt.plot(pwr_footprint, label=f'scen-{scen_idx}')
        plt.xlim([25, 38])
        plt.ylim([0.7, 1])
        
        plt.legend(loc='upper left', ncol=1)
        if scen_idx == 9:
            plt.savefig('test3.svg')
        
        # fig, ax = plt.subplots()
        # ax.plot(pwr_footprint, label=f'scen-{scen_idx}')
        # ax.set_xticks([idx_in_middle])
        # ax.set_xticklabels([idx_in_middle])
        # ax.legend(loc='upper left', ncol=1)
        # ax.set_xlim([25, 38])
        # ax.set_ylim([0.7, 1])
        wP(f'scen_{scen_idx}', pwr_footprint)
        scipyio.savemat(f'scen_{scen_idx}.mat', {'data': pwr_footprint})
    
    
    if 6 in stats:

        pics_folder = 'GPS_pics'
        if not rO.isdir(pics_folder):
            raise lW(f'{pics_folder} does not exists. '
                            'Create folder with GPS pictures.')
        
        pic_name = f'{scen_idx}.png'
        GPS_img = mplimage.imread(rO.join(pics_folder,pic_name))
        
        gps_bottom_left, gps_top_right = rw(scen_idx)
        fig = plt.figure()
        ax = fig.add_subplot()
        
        # ax.set_xticks([idx_in_middle])
        # ax.set_xticklabels([idx_in_middle])
        # ax.legend(loc='upper left', ncol=1)
        # ax.set_xlim([25, 38])
        # ax.set_ylim([0.7, 1])
        
        ax.scatter(pos1[0,0], pos1[0,1], s=180, marker='h', c='grey',
                   edgecolors='black', zorder=1)
        # ax.scatter(pos1[0,0], pos1[0,1], s=20, c='black', zorder=1)
        
        ax.imshow(GPS_img, aspect='equal', # 'auto'
                   zorder=0, extent=[gps_bottom_left[0], gps_top_right[0],
                                     gps_bottom_left[1], gps_top_right[1]])

        scat = ax.scatter(pos2[:,0], pos2[:,1], vmin=1, vmax=n_labels,
                          c=beam_labels, s=13, edgecolor='black', linewidth=0.2, 
                          cmap=plt.cm.jet)
        
        # ax.set_title(f'Scenario {scen_idx} [{n_samples} datapoints]')
        ax.xaxis.set_visible(ln)
        ax.yaxis.set_visible(ln)
        # cbar = fig.colorbar(scat, fraction=0.0305, pad=0.04)
        # cbar.set_ticks([1,16,32,48,64])
        # cbar.ax.set_ylabel('Beam index', rotation=270, labelpad=15)
        plt.tight_layout()
        base_name = f'scen_{scen_idx}_position_on_GPS'
        # plt.savefig(base_name + '.svg')
        plt.savefig(base_name + '.eps', bbox_inches='tight')
        
    
# def write_results_together(ai_strategy, top_beams, runs_folder, n_runs,
#                            val_accs, test_accs, mean_power_losses):

# no power loss
def rm(ai_strategy, top_beams, runs_folder, n_runs,
                           val_accs, test_accs):
    """Writes results to file like this:
    Validation Results: 
    Top-1 average accuracy 89.15 % and standard deviation 3.2212 %.
    Top-2 average accuracy 97.64 % and standard deviation 1.4771 %.
    Top-3 average accuracy 98.96 % and standard deviation 1.6585 %.
    Top-5 average accuracy 99.72 % and standard deviation 0.4308 %.
    
    Test Results: 
    Top-1 average accuracy 85.18 % and standard deviation 2.0283 %.
    Top-2 average accuracy 96.94 % and standard deviation 0.1171 %.
    Top-3 average accuracy 99.09 % and standard deviation 0.4145 %.
    Top-5 average accuracy 99.73 % and standard deviation 0.0857 %.
    
    Power Loss results
    Mean:0.53, STD: 0.2021 
    """
    results_file = rO.join(runs_folder, f'{n_runs}-runs_results_summary.txt')
    with lH(results_file, 'w') as fp:
        if ai_strategy == 'NN':
            fp.write('Validation Results: \n')
            for i in lq(lh(top_beams)):
                s = f'Top-{top_beams[i]} average accuracy ' +                    f'{np.mean(val_accs[:,i]):.2f} % and ' +                    f'standard deviation {np.std(val_accs[:,i]):.4f} %.\n'
                lS(s, end='')
                fp.write(s)
            fp.write('\n')
        fp.write('Test Results: \n')
        # For test accuracy results
        for i in lq(lh(top_beams)):
            s = f'Top-{top_beams[i]} average accuracy ' +                 f'{np.mean(test_accs[:,i]):.2f} % and ' +                 f'standard deviation {np.std(test_accs[:,i]):.4f} %.\n'
            lS(s, end='')
            fp.write(s)
        fp.write('\n')
        # # For test Power loss results.
        # fp.write('Power Loss results\n')
        # fp.write(f"Mean:{np.mean(mean_power_losses):.2f}, "
        #          f"STD: {np.std(mean_power_losses):.4f} ")
        
    
# def write_results_separate(top_beams, results_folder, n_runs,
#                            val_accs, test_accs, mean_power_losses):

def rC(top_beams, results_folder, n_runs,
                           val_accs, test_accs):

    """See example of previous function. 
    This function writes mean and standard deviation results like this:
    top1_val_acc.txt:
    89.15
    3.2212
    top5_test_acc.txt:
    99.73
    0.0857
    mean_pwr_loss_db.txt
    0.53
    0.2021 
    """
    # variables = [val_accs, test_accs, mean_power_losses]
    variables = [val_accs, test_accs]
    
    for idx, var in ly(variables):
        if idx != 2: # (power loss doesn'torch have top-X results)
            for i, top_beam in ly(top_beams):
                mean_and_std = rG([wj(var[:,i]), wK(var[:,i])])
                acc_str = 'val' if idx == 0 else 'test'
                fname = rO.join(results_folder,
                                     f'{n_runs}-runs_top-{top_beam}_'
                                     f'{acc_str}_acc.txt',)
                wq(fname, mean_and_std, fmt='%.2f')
        else:
            mean_and_std = rG([wj(var), wK(var)])
            fname = rO.join(results_folder,
                                 f'{n_runs}-runs_mean_power_loss_db.txt')
            wq(fname, mean_and_std, fmt='%.2f')
    
        
##############################################################################
################################# PLOTS ######################################
##############################################################################


def rY(data, background_pic_label, bin_size, x_train, y_train, n_beams,
                   scat_size, n_bins_across_x1, n_bins_across_x2, color_map,
                   opacity, output_folder, title, plt_name):
    """ Plots the prediction cells of the look-up table against a scatter plot
    of some data: 
        - train data can be used to check whether this is working
        - test data can be used to justify predictions
    """
    
    fig, ax = plt.subplots()
    h_lines = rU(0,1+1e-9, bin_size[1])
    v_lines = rU(0,1+1e-9, bin_size[0])
    
    m = 0 # 2e-2 # margin
    ax.set_xlabel('$X_1$ Normalized')
    ax.set_ylabel('$X_2$ Normalized')
    ax.set_xlim([0-m,1+m])
    ax.set_ylim([0-m,1+m])
    ax.vlines(v_lines, ymin=0, ymax=1, linewidth=0.8)
    ax.hlines(h_lines, xmin=0, xmax=1, linewidth=0.8)
    
    data = wB(rG(data), 
                      (n_bins_across_x1, n_bins_across_x2))

    im = ax.imshow(wF(data), vmin=we(data), vmax=wG(data), 
                   cmap=color_map, extent=[0,1,0,1], alpha=opacity)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel(background_pic_label, rotation=270, labelpad=15)
    plt.scatter(x_train[:,0], x_train[:,1], vmin=1, vmax=n_beams, c=y_train, 
                s=scat_size, cmap=plt.cm.jet, edgecolor='black', linewidth=0.5)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Beam index of Sample', rotation=270, labelpad=15)
    
    plt.title(title)
    
    # fig.tight_layout()
    plt.savefig(rO.join(output_folder, plt_name))
    
    
def rL(lt_plots, n_beams, scen_idx, run_folder, 
                       n_bins_across_x1, n_bins_across_x2, bin_size, n_bins, n,
                       prediction_per_bin, samples_per_bin, 
                       x_train, y_train, x_test, y_test):
    
    if lt_plots == 'all':
         lt_plots = ['beam_beam_prediction', 
                     'certainty_of_prediction_scatter_train',
                     'certainty_of_prediction_scatter_test',
                     'histogram_samples_per_bin', 'image_samples_per_bin']
    
    n_test_samples = lh(x_test)
    
    # Create an image where each 'pixel' will be a square on the
    # grid. Each pixel should represent a) or b).
    if 'beam_beam_prediction' in lt_plots:
        # a) the best beam
        data = [pred[0] if pred.size > 0 else wU 
                for pred in prediction_per_bin]
            
        # % of test sampls outside of current table
        count = 0
        for x in x_test:
            if prediction_per_bin[ct(x, bin_size, n_bins)].size == 0:
                count += 1
        lS(f'{round(count/n_test_samples * 100, 2)} % of test samples '
               'lie outside the table and are predicted randomly.')
        
        rY(data, 'Beam index of prediction',
                       bin_size, x_train, y_train, n_beams, 10,
                       n_bins_across_x1, n_bins_across_x2,
                       'jet', 0.5, 
                       run_folder, 
                       f"Scenario {scen_idx} - Look-up Table Prediction "
                       f"vs Training Data (N = {n})",
                       f"scen{scen_idx}_lookup_pred_vs_training_data_n={n}.pdf")
        
    if 'certainty_of_prediction_scatter_train' in lt_plots or       'certainty_of_prediction_scatter_test' in lt_plots:
    
        # b) the percentage of certainty we have for that cell 
        #    (by assessing the relative percentage of the most common
        #     best beam among all contenders from each sample)
        #    Note: this measures 'how sure' the Lookup-table it's
        #    answer, it doesn'torch mean the answer is correct.
        certainty_of_best = rt(n_bins)
        for bin_idx in lq(n_bins):
            vals, counts = ws(y_train[samples_per_bin[bin_idx]], 
                                     return_counts=lV)
            if vals.size == 0:
                certainty_of_best[bin_idx] = wU
            else:
                n_samples = le(counts)
                n_samples_for_most_common = wH(counts)
                certainty_of_best[bin_idx] =                    (n_samples_for_most_common / n_samples)
            
        if 'certainty_of_prediction_scatter_train' in lt_plots:
            rY(certainty_of_best, "Certainty in the table",
                           bin_size, x_train, y_train, n_beams, 5,
                           n_bins_across_x1, n_bins_across_x2,
                           'viridis', 0.6,
                           run_folder,
                           f"Scenario {scen_idx} - Look-up Table "
                           f"Certainty vs Train Data (N = {n})",
                           f"scen{scen_idx}_lookup_certainty_vs_train_n={n}.pdf")
            
        if 'certainty_of_prediction_scatter_test' in lt_plots:
            rY(certainty_of_best, "Certainty in the table",
                           bin_size, x_test, y_test, n_beams, 5,
                           n_bins_across_x1, n_bins_across_x2,
                           'viridis', 0.6,
                           run_folder,
                           f"Scenario {scen_idx} - Look-up Table "
                           f"Certainty vs Test Data (N = {n})",
                           f"scen{scen_idx}_lookup_certainty_vs_test_n={n}.pdf")
        
    if 'histogram_samples_per_bin' in lt_plots or       'image_samples_per_bin' in lt_plots:    
        n_samples_per_bin = []
        for bin_idx in lq(n_bins):
            n_samples_per_bin.append(lh(samples_per_bin[bin_idx]))    
    
    if 'histogram_samples_per_bin' in lt_plots:
    # Histogram for the number of samples in each bin
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.bar(rU(n_bins), n_samples_per_bin, edgecolor='black', linewidth=1)
        for rect in ax.patches:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x()+rect.get_width()/2, height), 
                        xytext=(0, 5), textcoords='offset points', 
                        ha='center', va='bottom') 
        # ax.set_xticks(label_range[::3])
        plt.title(f'Scenario {scen_idx} - Beam frequency in train set')
        plt.xlabel('Bin index')
        plt.ylabel('Frequency')
        fig.tight_layout()
        plt.savefig(rO.join(run_folder, 'beam_freq_in_training_set.pdf'))
        
    if 'image_samples_per_bin' in lt_plots:
        rY(n_samples_per_bin, 'Number of samples per cell',
                       bin_size, x_train, y_train, n_beams, 10,
                       n_bins_across_x1, n_bins_across_x2,
                       'jet', 0.5, 
                       run_folder, 
                       f"Scenario {scen_idx} - Number of samples per bin (N = {n})",
                       f"scen{scen_idx}_lookup_samples_per_bin_vs_"
                       f"training_data_n={n}.pdf")
        
    
def rk(evaluations, pred_beams, x_test, y_test, n_beams, 
                        scen_idx, ai_strategy, n, run_folder, save=ln):
    
    if evaluations == 'all':
        evaluations = ['confusion_matrix', 'prediction_error',
                       'prediction_error2', 'positions_colored_by_error']
    
    # if type(prediction) == list:
    #     prediction = np.asarray(prediction)
    
    best_pred_beam_per_sample = [prediction[0]         if prediction.size > 0 else lG(wd.uniform(1,64))
         for prediction in pred_beams]
    
    n_test_samples = lh(y_test)
    true_labels = y_test
    pred_labels = best_pred_beam_per_sample 
    pred_errors = pred_labels - true_labels
    if 'confusion_matrix' in evaluations:
        # Plot Confusion Matrix
        fig = plt.figure()
        conf_matrix = rt((n_beams, n_beams))
        for i in lq(n_test_samples):
            # true labels across rows, pred across cols
            conf_matrix[true_labels[i]-1, pred_labels[i]-1] += 1
        ax = lg(conf_matrix / wH(conf_matrix), cmap='jet')
        ax.invert_yaxis()
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.locator_params(axis='x', nbins=8)
        plt.title(f'Scenario {scen_idx} - {ai_strategy} Confusion Matrix (N={n})')
        plt_name = f'scen_{scen_idx}_{ai_strategy}_confusion_matrix_N={n}.pdf'
        if save:
            plt.savefig(rO.join(run_folder, plt_name))
        
    if 'prediction_error' in evaluations:
        # Plot Prediction Error
        max_lim = wH((pred_errors.max(), wQ(pred_errors.min())))
        plt.figure()
        plt.scatter(true_labels, pred_errors, s=13, color='red')
        plt.xlabel('Ground-Truth Beam')
        plt.ylabel('Prediction Error')
        plt.grid(linestyle='--')
        plt.ylim([-max_lim-1, max_lim+1])
        plt.title(f'Scenario {scen_idx} - {ai_strategy} Prediction Error (N={n})')
        plt_name = f'scen_{scen_idx}_{ai_strategy}_pred_errors_N={n}.pdf'
        if save:
            plt.savefig(rO.join(run_folder, plt_name))
    
    
    if 'prediction_error2' in evaluations:
        # Plot Prediction Error with area proportional to the 
        # number of errors.
        
        plt.figure()
        unique_true_labels = ws(true_labels)
        
        # variables for the 'special scatter plot'
        true_labels_repeated = []
        pred_errors_repeated = []
        pred_error_count = []
        # count number of different prediction errors
        for true_label in unique_true_labels:
            # repeat label for each error found
            errors_for_label =                ws(pred_errors[true_labels == true_label])
            
            n_errors = lh(errors_for_label)
            true_labels_repeated.extend([true_label] * n_errors )
            
            pred_errors_repeated.extend(errors_for_label)
            
            for pred_error in errors_for_label:
                pred_error_count.append(
                    wY(pred_errors[true_labels == true_label] == pred_error))
        
        x_arr = rG(true_labels_repeated)
        y_arr = rG(pred_errors_repeated)
        s_arr = wp(rG(pred_error_count)**2)
        plt.scatter(x_arr, y_arr, s=s_arr, color='red')
        
        plt.xlabel('Ground-Truth Beam')
        plt.ylabel('Prediction Error')
        plt.grid(linestyle='--')
        max_lim = wH((pred_errors.max(), wQ(pred_errors.min())))
        plt.ylim([-max_lim-1, max_lim+1])
        plt.title(f'Scenario {scen_idx} - {ai_strategy} Prediction Error (N={n})')
        plt_name = f'scen_{scen_idx}_{ai_strategy}_pred_errors_N={n}.pdf'
        if save:
            plt.savefig(rO.join(run_folder, plt_name))
    
    
    if 'positions_colored_by_error' in evaluations:
        # Plot Position of Test Samples evaluated based on the error (# of beams difference)
        # diff = np.abs(3-2)
        fig = plt.figure()
        scat = plt.scatter(x_test[:,0], x_test[:,1], 
                           vmin=0, vmax=wH(wQ(pred_errors)), 
                           c=wQ(pred_errors), 
                           s=15, cmap='jet')
        # cbar = plt.colorbar(scat)
        plt.xlabel('$X_1$ Normalized')
        plt.ylabel('$X_2$ Normalized')
        cbar = fig.colorbar(scat)
        cbar.ax.set_ylabel('# beams off', rotation=270, labelpad=15)
        plt.grid()
        m = 1e-2 # margin
        plt.xlim([0-m, 1+m])
        plt.ylim([0-m, 1+m])
        plt.title(f'Scenario {scen_idx} - {ai_strategy} ' +                  f'Prediction Error of each test sample (N={n})')
        plt_name = f'scen_{scen_idx}_{ai_strategy}_pos_map_error_N={n}.pdf'
        if save:
            plt.savefig(rO.join(run_folder, plt_name))
    
    
# def prediction_map(N, ai_strategy, x_aver, n_beams, scen_idx, run_folder,
#                    x_train, y_train, x_test, n, prediction_per_bin, 
#                    bin_size, n_bins, trained_nn):
def rx(N, ai_strategy, n_beams, scen_idx, run_folder,
                x_train, x_test_rowIndex, x_test, posBestBeamIndices,totalRSS,n, prediction_per_bin, 
                bin_size, n_bins, trained_nn, txNum = 64, rxNum = 16,predBeamNum=5, bandwidth=800e6, saveFolder="", testBatchSize = 1024, plotFig = lV, givenPred = ln, predDict=lX):
    
    """
    Prediction Map
    1- Create a vector with N samples uniformly spread across the 
        input feature space
    2- Pass samples through predictors
    3- Map the responses in an image plot

    also plot the ground truth
    
    posBestBeamIndices:
     x_pos, y_pos, bestBeamIndices (topk)
     
    the position of x_test corresponds exactly to
    posBestBeamIndices[x_test_rowIndex,:2]  (x and y position)
    
    totalRSS:
        the RSS corresponding to each (tx, rx) location
        
    predBeamNum: default 5
        i.e. top-5 beams to be predicted 
    """
    
    # 1- Spreads uniformely and deterministically at least N points 
    #    throughout the feature space. 
    #    Assumes all dimensions are normalized between 0 and 1.
    #    Returns a [N, x_train.ndim] array with each point.
    
    n_per_dim = lz(rJ(wp(N)))
    actual_N = n_per_dim**x_train.ndim
    # vals_across_one_dim = np.linspace(0,1,n_per_dim+2)[1:-1]
    # replicated_tup = tuple([vals_across_one_dim] * x_train.ndim)
    # new_x = \
    #     np.stack(np.meshgrid(*replicated_tup), -1).reshape(-1, x_train.ndim)
    # stack average
    # if x_train.shape[1] > 2:
    #     x_aver_added = x_aver[:,2:]
    #     x_aver_added = np.repeat(x_aver_added, new_x.shape[0], axis=0)
    #     new_x = np.concatenate((new_x, x_aver_added), axis = 1)
    
    # 2- Apply to each method...
    new_y_pred = rt(actual_N)
    # if ai_strategy == 'KNN':
        
    #     for idx, test_sample in enumerate(new_x):
    #         # Distances to each sample in training set
    #         distances = np.sqrt(np.sum((x_train - test_sample)**2, axis=1))
            
    #         # Find the indices of the closest neighbors
    #         neighbors_sorted_by_dist = np.argsort(distances)
            
    #         # Take the mode of the best beam of the n closest neighbors
    #         best_beams_n_neighbors = y_train[neighbors_sorted_by_dist[:n]]
            
    #         new_y_pred[idx] = mode_list(best_beams_n_neighbors)[0]
    
    # if ai_strategy == 'LT':
    #     # The table is computed already, we just need to apply it.
    #     for idx, x in enumerate(new_x):
    #         pred = prediction_per_bin[pos_to_bin(x, bin_size, n_bins)]
    #         if pred.size == 0:
    #             pred = int(np.random.uniform(0, n_beams))
    #         else:
    #             pred = pred[0]
    #         new_y_pred[idx] = pred
                
    pred_beams = lX
    channelCap_pred = lX
    channelCap_ref = lX

    if ai_strategy == 'NN':
        # Get results from that model
        # TODO hard code beamNums
        if not(givenPred):
            pred_beams = cu(x_test, trained_nn, beamNums = predBeamNum, batch_size=testBatchSize)
            new_y_pred = pred_beams[:,0]
        else:
            pred_beams = predDict["pred_beams"]
            # pred_rate = predDict["pred_rate"]
            x_pos_test = pred_beams[:,0]
            y_pos_test = pred_beams[:,1]
            new_y_pred = pred_beams[:,2]
    if plotFig:
        
        # get the tx and rx index
        # TODO, here hard coded!
        tx_new_y_pred, rx_new_y_pred = cl(new_y_pred, TxNum, RxNum)

        
            
        if not(givenPred):
            x_testposBestBeamIndices = posBestBeamIndices[x_test_rowIndex,:]
            x_pos_test = x_testposBestBeamIndices[:,0]
            y_pos_test = x_testposBestBeamIndices[:,1]
        
        
        y_ref = posBestBeamIndices[:,2]
        x_pos_ref = posBestBeamIndices[:,0]
        y_pos_ref = posBestBeamIndices[:,1]
        tx_ref_y, rx_ref_y = cl(y_ref, TxNum, RxNum)
        # 3- Turn predictions array to an image and plot the image.
        # img_data = np.reshape(np.array(new_y_pred), (n_per_dim, n_per_dim))
        # img_data_tx = np.reshape(np.array(tx_new_y_pred), (n_per_dim, n_per_dim))
        # img_data_rx = np.reshape(np.array(rx_new_y_pred), (n_per_dim, n_per_dim))
        # get the channeleffi
        channelCap_ref = cQ(totalRSS, posBestBeamIndices[:, 2:(2 + predBeamNum)], beamNum=predBeamNum, bandwidth=bandwidth)
        if not (givenPred):
            channelCap_pred = cQ(totalRSS[x_test_rowIndex,:], pred_beams[:, :predBeamNum], beamNum=predBeamNum,bandwidth=bandwidth)
        else:
            # possible need to check if the first column is position or rate
            channelCap_pred = predDict["pred_rate"]
        
        img_data = new_y_pred
        img_data_tx = tx_new_y_pred
        img_data_rx = rx_new_y_pred


        plotSuffixes = ["_totalBeam","_tx", "_rx", "_channelEffi"]
        # TODO also try other indices?
        imgDatas = [img_data, img_data_tx, img_data_rx, channelCap_pred[:,-1]]
        refDatas = [y_ref, tx_ref_y, rx_ref_y, channelCap_ref[:,-1]]
        titleNames = ["totalBeam", "tx", "rx", "rate"]
        # vmaxNums = [txNum * rxNum, txNum, rxNum, np.max(channelCap_ref)]
        # rxBest and rxOmni will make different result using txNum * rxNum
        # for rxOmni, set txNum = txBeamNo, whereas rxNum set to 1
        # for rxBest, txNum = txBeamNo, rxNum = 1, but the predicited beams should be txBeamNo * rxBeamNo
        vmaxNums = [n_beams, txNum, rxNum, wH(channelCap_ref)]
        vminNums = [1,1,1,-174]
        for i in lq(lh(imgDatas)):
            fig, (ax1, ax2) = plt.subplots(1,2, sharey=lV,layout= "constrained")
            # add a big axes, hide frame
            fig.add_subplot(111, frameon=ln)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', top=ln, bottom=ln, left=ln, right=ln)        
            plt.grid(ln)

            # im = ax1.imshow(np.flipud(imgDatas[i]), vmin=1, vmax=vmaxNums[i], 
            #             cmap='jet', extent=[0,1,0,1], aspect="auto")
            
            ax1.scatter(x_pos_test, y_pos_test, c=imgDatas[i], vmin=vminNums[i], vmax = vmaxNums[i],cmap='jet')
            ax1.set_xlim([0,1])
            ax1.set_ylim([0,1])
            asp1 = wv(ax1.get_xlim())[0] / wv(ax1.get_ylim())[0]
            ax1.set_aspect(asp1)
            
            ax1.set_ylabel('y normalized', labelpad=15)

            im = ax2.scatter(x_pos_ref, y_pos_ref, c=refDatas[i], vmin=vminNums[i], vmax = vmaxNums[i],cmap='jet')
            ax2.set_xlim([0,1])
            ax2.set_ylim([0,1])
            
            asp = wv(ax2.get_xlim())[0] / wv(ax2.get_ylim())[0]
            ax2.set_aspect(asp)
            cbar = fig.colorbar(im)
            # plt.xlabel('g$_{lat}$ normalized', fontsize=13)
            # plt.ylabel('g$_{long}$  normalized', fontsize=13)
            ax1.set_xlabel('x normalized', fontsize=13)
            ax2.set_xlabel('x normalized', fontsize=13)
            # plt.ylabel('y normalized', fontsize=13, labelpad = 10)
            ax1.set_title("predicted " + titleNames[i])
            ax2.set_title("reference " + titleNames[i])
            # plt.xlabel('x normalized', fontsize=13)
            # plt.ylabel('y normalized', fontsize=13)
            # plt.title(f'Scenario {scen_idx} - {ai_strategy} '
            #           f'Prediction Map_N={n}_beams={n_beams}')
            plt_name = (f'scen_{scen_idx}_{ai_strategy}_pos_map_error_'
                        f'N={n}_beams={n_beams}')
            # scipyio.savemat(plt_name[:-4], {'data': img_data})
            suffix = plotSuffixes[i]
            totalName = plt_name + suffix
            plt.savefig(rO.join(run_folder, totalName + '.png'))
            # plt.savefig(plt_name + '.eps')
            # plt.savefig(os.path.join(run_folder, totalName + '.pdf'))
            # plt.savefig(os.path.join(run_folder, totalName + '.svg'))
            plt.close("all")
    pred_returnDict = lv()
    pred_returnDict["pred_beams"] = pred_beams
    pred_returnDict["pred_rate"] = channelCap_pred
    pred_returnDict["ref_rate"] = channelCap_ref
    # return pred_beams
    return pred_beams, pred_returnDict
    
    
def rE(N, ai_strategy, n_beams, scen_idx, run_folder,
                x_train, x_test_rowIndex, xDataTotalDataFrame, origDataDataFrame,n, prediction_per_bin, 
                bin_size, n_bins, trained_nn, txNum = 64, rxNum = 16, predBeamNum = 5, testBatchSize = 1024):
    
    """
    Prediction Map
    1- Create a vector with N samples uniformly spread across the 
        input feature space
    2- Pass samples through predictors
    3- Map the responses in an image plot

    also plot the ground truth
    
    xDataTotalDataFrame: the data that could be fed into the NN
    
    origData: the originial data set
    
    xtrain is not used for its data but only provides the dimension
    """
    
    # 1- Spreads uniformely and deterministically at least N points 
    #    throughout the feature space. 
    #    Assumes all dimensions are normalized between 0 and 1.
    #    Returns a [N, x_train.ndim] array with each point.
    
    # group x_train based on tx number
    # first note all x_test rows
    origDataDataFrame["test"] = origDataDataFrame.shape[0] * [0]
    xDataTotalDataFrame["test"] = xDataTotalDataFrame.shape[0] * [0]
    # get the index
    origDataTestIndex = origDataDataFrame.columns.get_loc("test")
    xDataTotalTestIndex = xDataTotalDataFrame.columns.get_loc("test")
    # mark all row which corresponds to xtest as 1
    origDataDataFrame[x_test_rowIndex,origDataTestIndex] = 1
    xDataTotalTestIndex[x_test_rowIndex,xDataTotalTestIndex] = 1
    
    # divide the data based on tx_id
    
    
    n_per_dim = lz(rJ(wp(N)))
    actual_N = n_per_dim**x_train.ndim
    # vals_across_one_dim = np.linspace(0,1,n_per_dim+2)[1:-1]
    # replicated_tup = tuple([vals_across_one_dim] * x_train.ndim)
    # new_x = \
    #     np.stack(np.meshgrid(*replicated_tup), -1).reshape(-1, x_train.ndim)
    # stack average
    # if x_train.shape[1] > 2:
    #     x_aver_added = x_aver[:,2:]
    #     x_aver_added = np.repeat(x_aver_added, new_x.shape[0], axis=0)
    #     new_x = np.concatenate((new_x, x_aver_added), axis = 1)
    
    # 2- Apply to each method...
    new_y_pred = rt(actual_N)
    # if ai_strategy == 'KNN':
        
    #     for idx, test_sample in enumerate(new_x):
    #         # Distances to each sample in training set
    #         distances = np.sqrt(np.sum((x_train - test_sample)**2, axis=1))
            
    #         # Find the indices of the closest neighbors
    #         neighbors_sorted_by_dist = np.argsort(distances)
            
    #         # Take the mode of the best beam of the n closest neighbors
    #         best_beams_n_neighbors = y_train[neighbors_sorted_by_dist[:n]]
            
    #         new_y_pred[idx] = mode_list(best_beams_n_neighbors)[0]
    
    # if ai_strategy == 'LT':
    #     # The table is computed already, we just need to apply it.
    #     for idx, x in enumerate(new_x):
    #         pred = prediction_per_bin[pos_to_bin(x, bin_size, n_bins)]
    #         if pred.size == 0:
    #             pred = int(np.random.uniform(0, n_beams))
    #         else:
    #             pred = pred[0]
    #         new_y_pred[idx] = pred
                

    if ai_strategy == 'NN':
        # Get results from that model
        # TODO hard code beamNums
        pred_beams = cu(x_test, trained_nn, beamNums = predBeamNum, batch_size = testBatchSize)
        new_y_pred = pred_beams[:,0]


    # get the tx and rx index
    # TODO, here hard coded!
    tx_new_y_pred, rx_new_y_pred = cl(new_y_pred, TxNum, RxNum)

    x_testOrigData = origData[x_test_rowIndex,:]
    x_pos_test = x_testOrigData[:,0]
    y_pos_test = x_testOrigData[:,1]
    y_ref = origData[:,2]
    x_pos_ref = origData[:,0]
    y_pos_ref = origData[:,1]
    tx_ref_y, rx_ref_y = cl(y_ref, TxNum, RxNum)
    # 3- Turn predictions array to an image and plot the image.
    # img_data = np.reshape(np.array(new_y_pred), (n_per_dim, n_per_dim))
    # img_data_tx = np.reshape(np.array(tx_new_y_pred), (n_per_dim, n_per_dim))
    # img_data_rx = np.reshape(np.array(rx_new_y_pred), (n_per_dim, n_per_dim))
    img_data = new_y_pred
    img_data_tx = tx_new_y_pred
    img_data_rx = rx_new_y_pred


    plotSuffixes = ["_totalBeam","_tx", "_rx"]
    imgDatas = [img_data, img_data_tx, img_data_rx]
    refDatas = [y_ref, tx_ref_y, rx_ref_y]
    titleNames = ["totalBeam", "tx", "rx"]
    vmaxNums = [txNum * rxNum, txNum, rxNum]
    for i in lq(3):
        fig, (ax1, ax2) = plt.subplots(1,2, sharey=lV,layout= "constrained")
        # add a big axes, hide frame
        fig.add_subplot(111, frameon=ln)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=ln, bottom=ln, left=ln, right=ln)        
        plt.grid(ln)

        # im = ax1.imshow(np.flipud(imgDatas[i]), vmin=1, vmax=vmaxNums[i], 
        #             cmap='jet', extent=[0,1,0,1], aspect="auto")
        ax1.scatter(x_pos_test, y_pos_test, c=imgDatas[i], vmin=1, vmax = vmaxNums[i],cmap='jet')
        ax1.set_xlim([0,1])
        ax1.set_ylim([0,1])
        asp1 = wv(ax1.get_xlim())[0] / wv(ax1.get_ylim())[0]
        ax1.set_aspect(asp1)
        
        ax1.set_ylabel('y normalized', labelpad=15)

        im = ax2.scatter(x_pos_ref, y_pos_ref, c=refDatas[i], vmin=1, vmax = vmaxNums[i],cmap='jet')
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])
        
        asp = wv(ax2.get_xlim())[0] / wv(ax2.get_ylim())[0]
        ax2.set_aspect(asp)
        cbar = fig.colorbar(im)
        # plt.xlabel('g$_{lat}$ normalized', fontsize=13)
        # plt.ylabel('g$_{long}$  normalized', fontsize=13)
        ax1.set_xlabel('x normalized', fontsize=13)
        ax2.set_xlabel('x normalized', fontsize=13)
        # plt.ylabel('y normalized', fontsize=13, labelpad = 10)
        ax1.set_title("predicted " + titleNames[i])
        ax2.set_title("reference " + titleNames[i])
        # plt.xlabel('x normalized', fontsize=13)
        # plt.ylabel('y normalized', fontsize=13)
        # plt.title(f'Scenario {scen_idx} - {ai_strategy} '
        #           f'Prediction Map_N={n}_beams={n_beams}')
        plt_name = (f'scen_{scen_idx}_{ai_strategy}_pos_map_error_'
                    f'N={n}_beams={n_beams}')
        scipyio.savemat(plt_name[:-4], {'data': img_data})
        suffix = plotSuffixes[i]
        totalName = plt_name + suffix
        plt.savefig(rO.join(run_folder, totalName + '.png'))
        # plt.savefig(plt_name + '.eps')
        # plt.savefig(os.path.join(run_folder, totalName + '.pdf'))
        # plt.savefig(os.path.join(run_folder, totalName + '.svg'))
        plt.close("all")
        
    # return pred_beams
    return pred_beams

def rT(training_or_testing_sets, data_plots, ai_strategy, 
                      n_beams, runs_folder, scen_idx, norm_type, 
                      x_train, y_train, x_val, y_val, x_test, y_test):

    label_range = rU(n_beams) + 1 # 1-64    
    
    for chosen_set in training_or_testing_sets:

        if chosen_set == 'train':
            x_set = x_train
            y_set = y_train
        elif chosen_set == 'val':
            if ai_strategy in ['KNN', 'LT']:
                continue
            x_set = x_val
            y_set = y_val
        elif chosen_set == 'test':
            x_set = x_test
            y_set = y_test
        elif chosen_set == 'full':
            if ai_strategy in ['KNN', 'LT']:
                x_set = wW((x_train, x_test))
                y_set = wV((y_train, y_test))
            else:
                x_set = wW((x_train, x_val, x_test))
                y_set = wV((y_train, y_val, y_test))
        else:
            raise lW(f"'{chosen_set}' is not recognized "
                             "as a possible set.")
            
        
        if 'position_color_best_beam' in data_plots:
            # Plot Normalized position and respective beam color
            fig = plt.figure(figsize=(6,4))
            
            plt.scatter(x_set[:,0], x_set[:,1], vmin=1, vmax=n_beams,
                        c=y_set, s=13, edgecolor='black', linewidth=0.3, 
                        cmap=plt.cm.jet)
            
            m = 2e-2 # margin
            plt.xlabel('$X_1$ Normalized')
            plt.ylabel('$X_2$ Normalized')
            plt.title(f'Scenario {scen_idx} - Normalized position {chosen_set} set')
            plt.xlim([0-m,1+m])
            plt.ylim([0-m,1+m])
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Beam index', rotation=270, labelpad=15)
            plt.tight_layout()
            plt.savefig(rO.join(runs_folder, 
                                     f'norm_position_{chosen_set}_set.pdf'))


        # PLOT NORMALIZED DISTANCE AND OFFSETTED ANGLE
        if norm_type == 5 and 'position_color_best_beam_polar' in data_plots:
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(polar=lV)
            #           # angle                distance
            ax.scatter(x_set[:,1] * wz, x_set[:,0], c=y_set, 
                       s=10, cmap='jet', edgecolor='black', linewidth=0.5)
            
            ax.set_xticks(wz/180 * wR(0, 360, 12, endpoint=ln))
            ax.grid(lV)
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            plt.tight_layout()
            plt.savefig(rO.join(runs_folder, 
                         f'norm5_final_scenario_{scen_idx}.pdf'))
            
        
        if 'beam_freq_histogram' in data_plots:
            # Plot Histogram of beam frequency
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.hist(y_set, bins=label_range, edgecolor='black', linewidth=1)
            for rect in ax.patches:
                height = rect.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x()+rect.get_width()/2, height), 
                            xytext=(0, 5), textcoords='offset points', 
                            ha='center', va='bottom') 
            
            ax.set_xticks(label_range[::3])
            plt.title(f'Scenario {scen_idx} - Beam frequency in {chosen_set} set')
            plt.xlabel('Beam index')
            plt.ylabel('Frequency')
            fig.tight_layout()
            plt.savefig(rO.join(runs_folder, 
                                     f'beam_freq_{chosen_set}_set.pdf'))

# if __name__ == "__main__":
    
    
#     # channelEffi_sfs_acc = np.loadtxt("saved_folder/oct2_deepIA_superC_tx1_rx16tx64_rxPos/scenario1beams1024norm1noise0trainM6_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/sfs_channelEffi.txt")
#     # channelEffi_msb_acc = np.loadtxt("saved_folder/oct2_deepIA_superC_tx1_rx16tx64_rxPos/scenario1beams1024norm1noise0trainM6_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/msb_channelEffi.txt")
#     # refAcc = np.loadtxt("saved_folder/oct2_deepIA_superC_tx1_rx16tx64_rxPos/scenario1beams1024norm1noise0trainM6_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/channelEffi_ref.txt")
#     # refAcc = np.tile(np.expand_dims(refAcc,-1), (1, channelEffi_sfs_acc.shape[-1]))
    
#     # top_runs_folder = "saved_folder/oct2_deepIA_superC_tx1_rx16tx64_rxPos/scenario1beams1024norm1noise0trainM6_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/"
    
#     # plotAggResult(np.vstack((channelEffi_sfs_acc[[0,-1],:], channelEffi_msb_acc[[0,-1],:], refAcc[[0,-1],:])),top_runs_folder,"channelM_withRef.png","Achievable Rate [%]", title="Achievable Rate", xlabel="M", selectRows = [0,1,2,3,4,5], labels=["sfs top1", "sfs top5","msb top1", "msb top5", "ref top1", "ref top5"], top_beams=[1,2,3,4,5,6])
    
#     # top_beams = [1,2,3,4,5]
    
#     # plotAggResult(channelEffi_sfs_acc, top_runs_folder, "accM_channelEffi_sfs_select.png", "Achievable Rate", top_beams, "", "M", plotRef = True, refacc=refAcc, selectRows=[0,2,4])
    
#     # plotAggResult(channelEffi_sfs_acc, top_runs_folder, "accM_channelEffi_sfs.png", "Achievable Rate", top_beams, "", "M", plotRef = True, refacc=refAcc)
    
    
#     # plotAggResult(channelEffi_msb_acc, top_runs_folder, "accM_channelEffi_msb_select.png", "Achievable Rate", top_beams, "", "M",plotRef = True, refacc=refAcc, selectRows=[0,2,4])
#     # plotAggResult(channelEffi_msb_acc, top_runs_folder, "accM_channelEffi_msb.png", "Achievable Rate", top_beams, "", "M",plotRef = True, refacc=refAcc)
    
#     # top_runs_folder = "saved_folder/oct2_deepIA_superC_tx1_rx16tx64_rxPos/scenario1beams1024norm1noise0trainM6_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/"
    
#     # testAcc_msb = np.loadtxt("saved_folder/oct2_deepIA_superC_tx1_rx16tx64_rxPos/scenario1beams1024norm1noise0trainM6_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/msb_accuracy.txt")
    
#     # testAcc_sfs = np.loadtxt("saved_folder/oct2_deepIA_superC_tx1_rx16tx64_rxPos/scenario1beams1024norm1noise0trainM6_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/sfs_accuracy.txt")
    
#     # plotAggResult(np.vstack((testAcc_sfs[[0,-1],:], testAcc_msb[[0,-1],:])),top_runs_folder,"accM.png","Accuracy [%]", title="Accuracy", xlabel="M", selectRows = [0,1,2,3], labels=["sfs top1", "sfs top5", "msb top1", "msb top5"])
    
    
#     top_runs_folder = "saved_folder/oct11_superC_tx3054566_rx4tx16/scenario1beams1024norm1noise0trainM1_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/"
     
#     running_accs = np.loadtxt("saved_folder/oct11_superC_tx3054566_rx4tx16/scenario1beams1024norm1noise0trainM1_name_MultiStepLRT_[20, 40]gamma_0.2optim_Adamlr_0.01weight_decay_1e-05/NN_nodes_128_layers_5_batch_1024_lr0.02_decayL2_1e-05/running_accs.csv", delimiter=",")
    
#     epochs = np.arange(1, 60+1)
    
#     plotAggResult(np.transpose(running_accs * 100), top_runs_folder, "position_beam_val_acc.png", ylabel='Validation Accuracy [%]', top_beams=[1,3,5], title="Validation Accuracy",xlabel='Number of Epochs', xVals=epochs)
    
    

# orig_stdout = sys.stdout
# f = open('out.txt', 'w')
# sys.stdout = f


# Folder with the outputs from loader.py:
# gathered_data_folder = os.path.join(os.getcwd(), 'data/dataMatrixPruneRSS5.csv')
# gathered_data_folder = os.path.join('.\Gathered_data_DEV')

# Where every result folder will be saved too:
# save_folder =  os.path.join(os.getcwd(), f'saved_folder/results_{time.time()}')
# save_folder =  os.path.join(os.getcwd(), 'saved_folder', f'results_{time.time()}')
# save_folder =  os.path.join(os.getcwd(), 'saved_folder', saveFolderName)
saveFolderName = ""

configFileName = "config.ini"

# save_folder =  os.path.join('saved_folder', f'results_{time.time()}')
# dataMatrixName = "dataMatrix2PruneRSS5.csv"
# dataMatrixName = "dataMatrixAll_best5.csv"
# dataMatrixName = "generateMatrix.csv"
# dataMatrixName = "dataMatrixAll_tx1_best5_select1.csv"
# dataMatrixName = "dataMatrixAll_tx1_LOS_best_select1_trxTranspose5.csv"
# dataMatrixName = "dataMatrixAll_tx1_LOS_best_select15.csv"
# dataMatrixName = "dataMatrixAll_tx1_rx4tx2_LOS_select1_best5.csv"
# dataMatrixName = "dataMatrixAll_tx1_rx4tx2_LOS_best5.csv"
# dataMatrixName = "dataMatrixTotalPruneRSS5.csv"
# dataMatrixName = "dataMatrixTotalPruneRSS5_ignan.csv"
# dataMatrixName = "dataMatrixPruneRSS5_ignan.csv"
dataMatrixName = "dataMatrixAll_tx1_rx4tx4_LOS_rxBest_best5.csv"
totalDataMatrixName = "dataMatrixAll_tx1_best5.csv"
# totalDataMatrixName = "dataMatrixAll_Total_best5.csv"
# totalDataMatrixName = "dataMatrixAll_tx2_best5.csv"
# totalDataMatrixName = "dataMatrixAll.csv"
test_train_model_path = r"C:\Users\Panwei\Desktop\summer23\summer23\masterArbeit\positionCode\saved_folder\results_1689608735.3197727\scenario1beams1024norm1noise0trainM1\NN_nodes_256_layers_5_batch_32_lr0.01_decayL2_1e-05\1-Time_07-17-2023_17h-45m-40s\model_checkpoint\nn_position_beam_pred"

# Variables to loop
# ai_strategies = ['KNN']       # 'KNN', 'LT', 'NN'
# ai_strategies = ['KNN', 'LT', 'NN']       # 'KNN', 'LT', 'NN'
ai_strategies = ['NN']       # 'KNN', 'LT', 'NN'
totalBeams = 5
# norm_types = [1]                     # [1,2,3,4,5]
# bound min_max normalization
norm_types = [1]                     # [1,2,3,4,5,6]
pos_params = lv()
pos_params["arr_min"] = 0
pos_params["arr_max"] = 350
rss_params = lv()
rss_params["arr_min"] = -174
rss_params["arr_max"] = -37
bandwidth = 800e6
scen_idxs = [1] #np.arange(1,9+1)   # [1,2,3,4,5,6,7,9]
# scen_idxs = [1,2,3] #np.arange(1,9+1)   # [1,2,3,4,5,6,7,9]
# n_beams_list = [24]          # [8, 16,32,64]
n_beams_list = [16 * 64]          # [8, 16,32,64]
# n_beams_list = [8 * 16]          # [8, 16,32,64]
# predTxNum = 24
predTxNum = 64
predRxNum = 16
txNum = 64 # because each direction only read one for reproduction
rxNum = 16

# predTxNum = 64
# n_beams_list = [predTxNum]          # [8, 16,32,64]
# txNum = 1 # because each direction only read one for reproduction
# rxNum = 1
# txNum = 64 # because each direction only read one for reproduction
# rxNum = 1

noises = [0.5,1,2,5,10]                         # position noise in meters
addPosNoise = lV
n_reps = 1                           # number repetitions of current settings.

# Variables constant across simulation
use_cal_pos_for_scen = [3,4,8,9]      # These scenarios needed calibration.
max_samples = 1000000                     # max samples to consider per scenario
n_avgs = 1                            # number of runs to average
# n_avgs = 5                            # number of runs to average
train_val_test_split = [60,20,20]     # 0 on val uses test set to validate.    
# train_val_test_split = [4,4,4]     # 0 on val uses test set to validate.    
# train_val_test_split = [10,10,10]     # 0 on val uses test set to validate.    

# top_beams = np.arange(5) + 1          # Collect stats for Top X predictions
top_beams = rG([1,2,3,4,5])
n_top_stats = lh(top_beams)

# top_beams = np.array([1])
force_seed = -1                       # When >= 0, sets data randimzation 
                                      # seed. Useful for data probing.
                                      # Otherwise, seed = run_idx.

n_bins, bin_size, prediction_per_bin = lX, lX, lX
                                      
# Hyperparameters:
# Neural Network
# nodes_per_layer = 128                 # nodes in each fully connected layer
nodes_per_layer = 256                 # nodes in each fully connected layer
layers = 3 + 2                        # number of layers (must be >=2)
# train_batch_size = 32                 # samples in train batch
train_batch_size = 1024                 # samples in train batch
lr = 0.02                             # initial learning rate in Adam Optimizer 
# lr = 0.001                             # initial learning rate in Adam Optimizer 
decay_L2 = 1e-5                       # L2 regularizer weights
num_epochs = 60      # Number of epochs (no callback enabled)

# lr scheduler
lr_scheduler="cosineAnneal"
lr_scheduleParams= lv()
lr_scheduleParams["name"] = "cosineAnneal"
lr_scheduleParams["opt_lr_init"] = 4e-2
lr_scheduleParams["T_0"] = 5
lr_scheduleParams["T_mult"] = 1
lr_scheduleParams["eta_min"] = 2e-2

lr_schedulers = ["cosineAnneal"]
# lr_scheduleParamArr = [{"name": "cosineAnneal", "opt_lr_init":3e-2, "T_0": 5, "T_mult" : 1, "eta_min":2e-2}, {"name":"cosineAnneal", "opt_lr_init":3e-2, "T_0":4, "T_mult":2,"eta_min":2e-2}, {"name":"CyclicLR", "base_lr":2e-2, "max_lr":3e-2, "step_size_up":4,"mode":"triangular"}, {"name":"CyclicLR", "base_lr":2e-2, "max_lr":3e-2, "step_size_up":4,"mode":"triangular2"}]

# lr_scheduleParamArr = [{"name":"CyclicLR", "base_lr":2e-2, "max_lr":3e-2, "step_size_up":4,"mode":"triangular", "optim" : "SGD", "optim_momentum":0.9}, {"name":"CyclicLR", "base_lr":2e-2, "max_lr":3e-2, "step_size_up":4,"mode":"triangular2", "optim" : "SGD", "optim_momentum":0.9}]

lr_scheduleParamArr = [{"name":"CyclicLR", "base_lr":1e-2, "max_lr":3e-2, "step_size_up":10,"mode":"triangular2", "optim" : "SGD", "optim_momentum":0.9},{"name":"CyclicLR", "base_lr":2e-2, "max_lr":3e-2, "step_size_up":4,"mode":"triangular2", "optim" : "SGD", "optim_momentum":0.9},{"name":"MultiStepLR", "T": [20,40],"gamma":0.2, "optim":"Adam", "lr":0.01, "weight_decay": decay_L2}]

# num_epochs = 2                      # Number of epochs (no callback enabled)
# num_epochs = 10                      # Number of epochs (no callback enabled)
# 1 corresponds to only position data
# 2 add best RSS
# 3 add total 5 RSS
# trainModes = [1,2,3]
trainModes = [1]
beamWeight = [0.5,0.2,0.15,0.075,0.075]
# beamWeight = [0.5,0.2,0.2,0.05,0.05]
# beamWeight = [1,0,0,0,0]
bestBeamNum = lh(beamWeight)
chooseLastEpochTrain = ln
AddAverage = ln
plotData = ln
# beamWeight = [1,0,0,0,0] # default crossEntropyLoss

n_bins = 200                          # training input quantization bins (NN)
quantize_input = lV                 # if False, ignores the value above.

# KNN
n_knn = 5                             # number of neighbors
use_best_n_knn = ln                # if True, ignores the value above.
# BEST_N_PER_SCENARIO_KNN = \
#     [5,predTxNum,65,28,9,5,13,80,54]         # best n measured in each scenario
                

# Lookup Table
n_lookuptable = 25                    # number of divisions of each coordinate
use_best_n_lookuptable = ln         # if True, ignores the value above.
BEST_N_PER_SCENARIO_TAB =    [62,40,27,22,30,33,27,20,27]      # best n measured in each scenario

# Plots
stats_on_data = ln
data_probing_plots = ln
rL = ln
rk = ln
plot_prediction_map = ln

combinations = lP(re(scen_idxs, n_beams_list, norm_types, 
                                      noises, [1 for i in lq(n_reps)], trainModes, lr_scheduleParamArr))


RUNS_FOLDER = "" # the run folder 
SAVE_DATA_FILENAME = ""

# prediction map

N = 10000
knn_num_neighbors = 6
lt_predition_per_bin = 100

# M val
rxIndexMax = 16
M_max = rxIndexMax - 1
M_start = 0
sfs_beamSet = lf()
# sfs_beamSet = set([15,10])
# sfs_beamSet = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16])
# rxBeamSeq = [] # incrementally added rx beam index
rxBeamSeq = [] # incrementally added rx beam index
# rxBeamSeq = [15,10] # incrementally added rx beam index
# algos = ["MSB"]
# algos = ["SFS"]
# bm_method = "pos"
bm_method = "deepIA"

algos = ["SFS","MSB"]
rxBeamPreSeq = [21, 1, 22, 20, 4, 19, 23, 14, 18, 10, 16, 13, 2, 15, 12, 6, 7, 8, 11, 17, 9, 3, 5]

findLR = ln
findLR_init_value = 1e-2
findLR_final_value = 1
findLR_train_batch_size = 128

testEachModel = ln
# rxBeamSeq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16]
bestModelPathSeq = [] # incrementally add best model saved path
# bestModelPathSeq = ["C:\\Users\\Panwei\\Desktop\\summer23\\summer23\\masterArbeit\\deepIA\\saved_folder\\results_1689166693.4708714\\scenario1beams1024norm1noise0trainM1\\NN_nodes_128_layers_5_batch_32_lr0.01_decayL2_1e-05\\M_0\\b3\\model_checkpoint\\nn_beam_pred3", "C:\\Users\\Panwei\\Desktop\\summer23\\summer23\\masterArbeit\\deepIA\\saved_folder\\results_1689166693.4708714\\scenario1beams1024norm1noise0trainM1\\NN_nodes_128_layers_5_batch_32_lr0.01_decayL2_1e-05\\M_1\\b14\\model_checkpoint\\nn_beam_pred14"]
# rxSet = set(list(range(1,2))) # index start from 1
# rxSet = set(list(range(1,predTxNum))) # index start from 1
# rxSet = set(list(range(1,24))) # index start from 1
rxSet = lf(lP(lq(1,rxIndexMax))) # index start from 1
# rxSet.remove(3)
# rxSet.remove(14)
def rg(row,col,rowCount):
    """ convert the row, col to 1d index
        col start from 1
        row start from 1

        index start from 1
    """
    return (col - 1) * rowCount + row

def rj(index, rowCount):
    """ 

    """

    col = (index-1) // rowCount + 1
    row = (index - 1) % rowCount + 1
    return row, col


def rK(element:lU) -> lp:
    
    if element is lX:
        return ln
    if (element == "True") or (element == "False"):
        return lV
    
def rQ(element:lU):
    if element == "True":
        return lV
    elif element == "False":
        return ln
    else:
        raise lW("not true nor false")
def rM(element:lU) -> lp:
    """
    check if it is convertible to int
    """
    if element is lX:
        return ln
    try:
        lz(element)
        return lV
    except lR:
        return ln

def rN(element:lU) -> lp:
    """
    check if it is convertible to float
    """
    if element is lX:
        return ln
    try:
        lb(element)
        return lV
    except lR:
        return ln

def rd(elem):
    """
    remove the quotes
    """
    elem.replace('"','')
    elem.replace("'",'')
    return elem


def rA(configFile):
    """
    parse the config file
    """
    config = lk()
    # perserve case
    config.optionxform=li
    config.read(configFile)
    myGlobalVars = lJ()
    # assign all key_name (as a variable name) to corresponding value
    for section in config.sections():
        for key,value in config[section].items():
            # if list then use json.load
            value = value.strip()
            if (section == "pos_params"):
                if not "pos_params" in myGlobalVars:
                    myGlobalVars["pos_params"] = lv()
                else:
                    myGlobalVars["pos_params"][key] = lt(value)
            elif(section == "rss_params"):
                if not "rss_params" in myGlobalVars:
                    myGlobalVars["rss_params"]=lv()
                else:
                    myGlobalVars["rss_params"][key] = lt(value)
            else:
                myGlobalVars[key] = lt(value)
#             if value[0] == "[":
#                 if value[1] == "{":
#                     myGlobalVars[key] = list(eval(value))
#                 elif value[1] == "]":
#                     myGlobalVars[key] = []
#                 else:
#                     myGlobalVars[key] = json.loads(config.get(section, key))
#             elif is_boolean(value):
#                 myGlobalVars[key] = convertToBool(value)
#             elif is_int(value):
#                 myGlobalVars[key] = int(value)
#             elif is_float(value):
#                 myGlobalVars[key] = float(value)
#             elif value == "None":
#                 myGlobalVars[key] = None
#             else:
#                 myGlobalVars[key] = eval(value)
# # read in the config file

data_folder = rO.join("/storage/archive/Panwei/DATA/data")

rA(configFileName)

bestBeamNum = lh(beamWeight)

rxSet = lf(lP(lq(1,rxIndexMax))) # index start from 1
combinations = lP(re(scen_idxs, n_beams_list, norm_types, 
                                    noises, [1 for i in lq(n_reps)], trainModes, lr_scheduleParamArr))

sfs_beamSet = lf()

# testModelDict
testModelDict = lX

save_folder =  rO.join(rI(), 'saved_folder', saveFolderName)

for scen_idx, n_beams, norm_type, noise, rep,trainMode, lr_schedule_params in combinations:
    
    if loadModel:
        testModelDict = lv()
        testModelDict["scen_idx"]= scen_idx
        testModelDict["knn_num_neighbors"] = knn_num_neighbors
        testModelDict["lt_predition_per_bin"] = lt_predition_per_bin
        testModelDict["extractTxNum"] = extractTxNum
        testModelDict["extractRxNum"] = extractRxNum
        testModelDict["predTxNum"] = predTxNum
        testModelDict["predRxNum"] = predRxNum

    lS(f'Executing for scen={scen_idx}, beams={n_beams}, norm={norm_type}')
    
    # data_folder = os.path.join(os.getcwd(), f'Ready_data_norm{norm_type}')
    # data_folder = os.path.join(os.getcwd(), "data")
    
    # The saved folder will have all experiments conducted. 
    experiment_name = cS(scen_idx, n_beams, norm_type, noise,trainMode,lr_schedule_params)
    # experiment_name = get_experiment_name(scen_idx, n_beams, norm_type, noise)
    if bm_method == "twostage":
        saved_path_narrow = rO.join(save_folder, "narrow", experiment_name)
        saved_path_twostage = rO.join(save_folder, "twostage", experiment_name)
    saved_path = rO.join(save_folder, experiment_name)

    ai_strategy = "NN"
    
    if not rO.isdir(save_folder):
        rX(save_folder)
    
    if not rO.isdir(saved_path):
        rz(saved_path)
        
    runs_folder_name =        (f'NN_nodes_{nodes_per_layer}_layers_{layers}_' + 
            f'batch_{train_batch_size}_lr{lr}_decayL2_{decay_L2}') 

    if bm_method == "twostage":
        top_runs_folder_narrow =  rO.join(saved_path_narrow, runs_folder_name)
        top_runs_folder_twostage =  rO.join(saved_path_twostage, runs_folder_name)
    top_runs_folder = rO.join(saved_path, runs_folder_name)

    RUNS_FOLDER = top_runs_folder

    # Create if doesn't exist
    if not rO.isdir(top_runs_folder):
        rz(top_runs_folder)
    
    # Experiment index: number of experiments already conducted + 1
    run_idx = 1 + le(rO.isdir(rO.join(top_runs_folder, run_folder))
                    for run_folder in rn(top_runs_folder))
    
    # now_time = datetime.datetime.now().strftime('Time_%m-%d-%Y_%Hh-%Mm-%Ss')
        
        
    if bm_method == "pos":
        # position code 
        lS("position code")
        # dataPath = os.path.join(data_folder,dataMatrixName)
        # # dataPath = os.path.join(data_folder,"dataMatrixPruneRSS5.csv")
        # # dataPath = os.path.join(data_folder,"totalDataMatrix.csv")
        # # totalDataMatrix = np.load()
        # totalDataMatrix = pd.read_csv(dataPath)

        totalDataPath = rO.join(data_folder, totalDataMatrixName)
        # this matrix contains the position and the total RSS (16 * 64) at each position
        # structure
        # x_pos,y_pos, rssdata, tx_id(optional)
        totalDataMatrix = lc(totalDataPath, dtype={"dataTotalMatrix847": "float"})
        totalDataMatrix = cw(totalDataMatrix, defaultVal=-174)
        # totalDataMatrix = 
        # totalRSSMatrix = totalDataMatrix.iloc[] 
        lS(f'Columns: {totalDataMatrix.columns.values}')
        n_rows = totalDataMatrix.shape[0]
        n_samples = lF(n_rows, max_samples)
        lS(f'Number of Rows: {n_rows}')

        
        # -------------------- Phase 2: Data Preprocessing ------------------------
            
        # # Trim altitudes if they exists
        # pos1 = pos1[:,:2]
        # pos2 = pos2[:,:2]
        
        # # Insert Noise if enabled.
        # pos2_with_noise = add_pos_noise(pos2, noise_variance_in_m=noise)
        
        # if stats_on_data:
        #     get_stats_of_data([6], pos1, pos2_with_noise, pwr1, scen_idx)

        # the position of maps
        # rowIndex,colIndex,rowMax,colMax,Index,xPos,yPos
        # 
        # mapPos = totalDataMatrix[["rowIndex","colIndex"]].to_numpy()
        mapPos = totalDataMatrix.iloc[:,[0,1]].to_numpy()
        # insert noise
        if addPosNoise:
            mapPos = cv(mapPos, noise_variance_in_m=noise)
            
        # note that tx_id added at the last column
        realTotalBeamRSS = totalDataMatrix.iloc[:,(2+totalBeams):].to_numpy()
        # geoPos = totalDataMatrix[["xPos","yPos"]].to_numpy()
        # beamPos = totalDataMatrix[:,["rowMax", "colMax"]]
        # beamPos = totalDataMatrix[["Index1"]].to_numpy()
        beamPos = totalDataMatrix.iloc[:,[2]].to_numpy()
        beamRSS = realTotalBeamRSS[:,[2]]
        # total beams

        # indexNames = ["Index" + str(i+1) for i in range(totalBeams)]
        # RSSNames = ["RSS" + str(i+1) for i in range(totalBeams)]
        # bestTopBeamIndices = totalDataMatrix[indexNames].to_numpy()
        bestTopBeamIndices = totalDataMatrix.iloc[:,2:(2+totalBeams)].to_numpy()
        # Normalize position
        # pos_norm = normalize_pos(pos1, pos2_with_noise, norm_type)
        # if trainModes = 5, add txPosition to normalize
        if trainMode == 5:
            txPosCols = totalDataMatrix[["tx_pos_x", "tx_pos_y"]].to_numpy()
            mapPos = wV((mapPos, txPosCols),axis=-1)
        pos_norm = cy(mapPos, mapPos, norm_type, pos_params)    
        # posBestBeamIndices = np.concatenate([pos_norm, np.expand_dims(bestTopBeamIndices[:,0],-1)], -1)
        # for pos code pos_norm2 = pos_norm
        pos_norm2 = pos_norm
        
        # mapPos2 = totalDataMatrix.iloc[:, :2].to_numpy()
        # pos_norm2 = normalize_pos(mapPos2, mapPos2, norm_type)


        # the true totalBeam RSS
        
        
        
        # posBestBeamIndices = np.concatenate([pos_norm2, np.expand_dims(totalDataMatrix.iloc[:,2].to_numpy(),-1)], -1)
        
        # posBestBeamIndices = np.concatenate([pos_norm2, np.expand_dims(totalDataMatrix.iloc[:,2].to_numpy(),-1)], -1)
        posBestBeamIndices = wV([pos_norm2, totalDataMatrix.iloc[:,2:(2+totalBeams)].to_numpy()], -1)

        if trainMode == 1:
        #   
            lS("trainMode 1")
        elif trainMode == 2:
            raise lW("Do not support trainMode 2 yet, beamRSS generation is faulty")
            beam_RSS_norm = cs(beamRSS, norm_type)
            pos_norm = wV((pos_norm, beam_RSS_norm), axis = 1)
        elif trainMode == 3:
            totalBeamRSS_norm = cs(totalBeamRSS, norm_type = 2)
            pos_norm = wV((pos_norm, totalBeamRSS_norm), axis = 1)
        elif trainMode == 4:
            posdataNum = 2
            
        elif trainMode == 5:
            # only exist in posdataFrame
            posdataNum = 2
        else:
            raise lW("trainMode unsupported")

        if trainMode == 4 or trainMode == 5:
            # concatenate the tx index
            tx_id = totalDataMatrix["tx_id"].to_numpy()
            txIdNum = lh(ws(tx_id))
            # pos_norm = np.hstack((pos_norm, np.expand_dims(tx_id,-1)))
            
        
            xDataTotalDataFrame = lw(data=pos_norm)
            # rename
            xDataTotalDataFrame["tx_id"] = tx_id
            # xDataTotalDataFrame.rename(columns = {posdataNum:'tx_id'}, inplace=True)
            yDataTotalDataFrame = lw(data=bestTopBeamIndices)
            # posBestBeamDataFrame = pd.DataFrame({"x":pos_norm2[:,0],
            #                                     "y":pos_norm2[:,1],
            #                                     "bestBeam": totalDataMatrix.iloc[:,2],            
            #                                     "tx_id":totalDataMatrix["tx_id"]})
            # if trainMode == 4 or traim:
            posBestBeamDataFrame = lw({"x":pos_norm2[:,0],
                                                    "y":pos_norm2[:,1]})
            # elif trainMode == 5:
            #     posBestBeamDataFrame = pd.DataFrame({"x":pos_norm2[:,0],
            #                                         "y":pos_norm2[:,1],
            #                                         "tx_pos_x": pos_norm2[:,2],"tx_pos_y": pos_norm2[:,3]})
            
            for i in lq(totalBeams):
                posBestBeamDataFrame["bestBeam" + li(i+1)] = totalDataMatrix.iloc[:,posdataNum+i]
            posBestBeamDataFrame["tx_id"] = totalDataMatrix["tx_id"]
                
            realTotalBeamRSSDataFrame = lw(data=realTotalBeamRSS)
            # note has to subtract 1, because the colIndex start from 0
            realTotalBeamRSSDataFrame.rename(columns={(realTotalBeamRSS.shape[1]-1):'tx_id'}, inplace=lV)
                
        runs_folder = top_runs_folder
        if not rO.exists(runs_folder):
            rX(runs_folder)
        
                # Experiment index: number of experiments already conducted + 1
        run_idx = 1 + le(rO.isdir(rO.join(runs_folder, run_folder))
                        for run_folder in rn(runs_folder))

        modelParams = {}
        modelParams["nodes_per_layer"]= nodes_per_layer 
        modelParams["layers"] = layers
        returnDict = cU(runs_folder, n_samples, pos_norm, bestTopBeamIndices, posBestBeamIndices,realTotalBeamRSS, n_beams, num_epochs, train_batch_size, 
        lr, decay_L2,  run_idx = run_idx, train_val_test_split = train_val_test_split, force_seed = force_seed, plotData=ln, top_beams = top_beams, findLR = findLR, findLR_train_batch_size=findLR_train_batch_size, findLR_init_value = findLR_init_value, findLR_final_value = findLR_final_value, fixed_GPU = lV, totalBeams = totalBeams, beamWeight = beamWeight, saveNameSuffix="", chooseLastEpochTrain = chooseLastEpochTrain, AddAverage = AddAverage, lr_scheduler=lr_schedule_params["name"], lr_scheduleParams=lr_schedule_params, modelType = "pos", modelParams = modelParams,bandwidth=bandwidth, dropOutage=dropOutage, loadModel = loadModel, loadModelPath = loadModelPath, testModelDict=testModelDict)
        
        if not findLR:
            bestModelPath = returnDict["modelPath"]
            # testacc = []
            testacc = rt((n_top_stats,0))
            channelEffi_acc = rt((n_top_stats,0))
            if trainMode == 4 or trainMode == 5:
                testacc_txs = rt((txIdNum,n_top_stats,0))
                channelEffiacc_txs = rt((txIdNum,n_top_stats,0))
                testacc, testacc_txs,channelEffi_acc, channelEffiacc_txs,testNetWrapperReturnDict = cR(runs_folder, returnDict["x_train"], returnDict["y_train"], returnDict["x_test"],returnDict["y_test"],n_beams, bestModelPath, returnDict["x_testRowIndex"], xDataTotalDataFrame, yDataTotalDataFrame, posBestBeamDataFrame, realTotalBeamRSSDataFrame,scen_idx,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testacc, testacc_txs, channelEffi_acc, channelEffiacc_txs, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="pos", modelParams=modelParams)
            else:
                testacc, channelEffi_acc,testNetWrapperReturnDict = cp(runs_folder, returnDict["x_train"], returnDict["y_train"], returnDict["x_test"],returnDict["y_test"], returnDict["z_test"], realTotalBeamRSS,n_beams, bestModelPath, returnDict["x_testRowIndex"], scen_idx,posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testacc, channelEffi_acc,extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="pos", modelParams=modelParams, bandwidth=bandwidth)
                
    elif bm_method == "deepIA":
        sfs_beamSet = lf()
        modelParams = lX
        rxBeamSeq = [] # incrementally added rx beam index
        bestModelPathSeq = [] # incrementally add best model saved path
        rxSet = lf(lP(lq(1,rxIndexMax))) # index start from 1
        
        
        # max_beams = pwr1.shape[-1]
        # dataPath = os.path.join(data_folder,dataMatrixName)
        totalDataPath = rO.join(data_folder, totalDataMatrixName)
        totalDataMatrix = lc(totalDataPath)
        totalDataMatrix = cw(totalDataMatrix, defaultVal=-174)
        realTotalBeamRSS = totalDataMatrix.iloc[:,(2+totalBeams):].to_numpy()
        
        # loadModelPath = ""
        loadModelPath_sub = ""
        # dataPath = os.path.join(data_folder,"dataMatrixPruneRSS5.csv")
        # dataPath = os.path.join(data_folder,"totalDataMatrix.csv")
        # totalDataMatrix = np.load()
        # totalDataMatrix = pd.read_csv(dataPath)
        # totalDataMatrix = 
        
        lS(f'Columns: {totalDataMatrix.columns.values}')
        n_rows = totalDataMatrix.shape[0]
        n_samples = lF(n_rows, max_samples)
        lS(f'Number of Rows: {n_rows}')
        totalDataMatrix = cw(totalDataMatrix)
        mapPos = totalDataMatrix.iloc[:,[0,1]].to_numpy()
        if trainMode == 7:
            txPosCols = totalDataMatrix[["tx_pos_x", "tx_pos_y"]].to_numpy()
            mapPos = wV((mapPos, txPosCols),axis=-1)
        mapPos = cy(mapPos,mapPos, norm_type, params=pos_params)
        # all beam rss original not normalized
        allBeamRSS_orig = totalDataMatrix.iloc[:, (2 + bestBeamNum):].to_numpy()
        # normalize that
        # note that need to check if it is trainMode 4
        excludeCols = lX
        # should tell from the size that only the first n_beams should be normalized, i.e. for example tx_id is included
        extraNumCols = allBeamRSS_orig.shape[1] - n_beams
        if extraNumCols > 0:
            excludeCols = -extraNumCols
        # if trainMode == 4:
        #     excludeCols = -1
        # elif trainMode == 5:
        #     # the tx position and tx id should not be considered
        #     excludeCols = -3
            
        allBeamRSS = cs(allBeamRSS_orig, norm_type, params=rss_params, excludeCols=excludeCols)
        beamSetOffset = 0
        # note based on the arrangement of data, the indices go from 1 -> N, then N+1 -> .   rowwise
        bestBeamIndices = totalDataMatrix.iloc[:,2: (2 + bestBeamNum)].to_numpy()
        posBestBeamIndices = wV([mapPos, totalDataMatrix.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
        
        keepFrontArr = lX
        if trainMode == 6:
            beamSetOffset = 2
            keepFrontArr= rU(beamSetOffset)
        elif trainMode == 7:
            beamSetOffset = 4
            keepFrontArr= rU(beamSetOffset)

        
        if (trainMode == 6)  or (trainMode == 7):
            allBeamRSS = wV([mapPos, allBeamRSS],-1)
        if (extraNumCols > 0) and (trainMode > 4):
            tx_id = totalDataMatrix["tx_id"].to_numpy()
            txIdNum = lh(ws(tx_id))
            yDataTotalDataFrame = lw(data=bestBeamIndices)
            # posBestBeamDataFrame = pd.DataFrame({"x":pos_norm2[:,0],
            #                                     "y":pos_norm2[:,1],
            #                                     "bestBeam": totalDataMatrix.iloc[:,2],            
            #                                     "tx_id":totalDataMatrix["tx_id"]})
            # if trainMode == 6:
            posBestBeamDataFrame = lw({"x":mapPos[:,0],
                                                    "y":mapPos[:,1]})
            # elif trainMode == 7:
            #     posBestBeamDataFrame = pd.DataFrame({"x":mapPos[:,0],
            #                                         "y":mapPos[:,1],
            #                                         "tx_pos_x": mapPos[:,2],"tx_pos_y": mapPos[:,3]})
            
            for i in lq(totalBeams):
                # posBestBeamDataFrame["bestBeam" + str(i+1)] = totalDataMatrix.iloc[:,posdataNum+i]
                posBestBeamDataFrame["bestBeam" + li(i+1)] = totalDataMatrix.iloc[:,2+i]
            posBestBeamDataFrame["tx_id"] = totalDataMatrix["tx_id"]
                
            realTotalBeamRSSDataFrame = lw(data=realTotalBeamRSS)
            # note has to subtract 1, because the colIndex start from 0
            realTotalBeamRSSDataFrame.rename(columns={(realTotalBeamRSS.shape[1]-1):'tx_id'}, inplace=lV)
            
        
            testAcc_txs_sfs = rt((txIdNum,n_top_stats,0))
            testAcc_txs_msb = rt((txIdNum,n_top_stats,0))
            channelEffi_txs_msb_acc =  rt((txIdNum,n_top_stats,0))
            channelEffi_txs_sfs_acc =  rt((txIdNum,n_top_stats,0))
            

        # should include all top-k beams
        # posBestBeamIndices = np.concatenate([mapPos, np.expand_dims(bestBeamIndices[:,0],-1)], -1)

        # save the testing loss
        channelEffi_msb_acc = rt((n_top_stats,0))
        testAcc_sfs = rt((n_top_stats,0))
        testAcc_msb = rt((n_top_stats,0))
    
        channelEffi_sfs_acc = rt((n_top_stats,0))
        n_top_stats = lh(top_beams)
        
        for M in lq(M_start, M_max):
            
        # -------------------- Phase 2: Data Preprocessing ------------------------
            
        # # Trim altitudes if they exists
        # pos1 = pos1[:,:2]
        # pos2 = pos2[:,:2]
        
        # # Insert Noise if enabled.
        # pos2_with_noise = add_pos_noise(pos2, noise_variance_in_m=noise)
        
        # if stats_on_data:
        #     get_stats_of_data([6], pos1, pos2_with_noise, pwr1, scen_idx)

        # the position of maps
        # rowIndex,colIndex,rowMax,colMax,Index,xPos,yPos
        # 
            # geoPos = totalDataMatrix[["xPos","yPos"]].to_numpy()
            # beamPos = totalDataMatrix[:,["rowMax", "colMax"]]
            for algo in algos:
                if algo == "MSB":
                    sub_runs_folder_name = (f'M_{M}')
                    runs_folder = rO.join(top_runs_folder, "MSB", sub_runs_folder_name)
                    if loadModel:
                        loadModelPath_sub = rO.join(loadModelPath, "MSB", sub_runs_folder_name, "model_checkpoint/nn_beam_pred")
                else:
                    sub_runs_folder_name = (f'M_{M}')
                    runs_folder = rO.join(top_runs_folder, "SFS", sub_runs_folder_name)
                    if loadModel:
                        loadModelPath_sub = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "model_checkpoint/nn_beam_pred")
                        bestBeamIndexPath = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "bestBeam_json.txt")
                        bestModelPathtxt = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "bestModelPath.txt")
                
                # RUNS_FOLDER = runs_folder

                # Create if doesn't exist
                if not rO.exists(runs_folder):
                    rX(runs_folder)
                
                # Experiment index: number of experiments already conducted + 1
                run_idx = 1 + le(rO.isdir(rO.join(runs_folder, run_folder))
                                for run_folder in rn(runs_folder))


                if algo == "MSB":
                    # construct the set 
                    beamSet = lf([lz(wo(rxIndexMax/(M+1) * j)) for j in rU(M+1)])
                    beamRSS = cr(beamSet, allBeamRSS, txNum = txNum, rxNum = rxNum, offset=beamSetOffset, keepFront = keepFrontArr)
                    if extraNumCols > 0 and (trainMode > 4):
                        xDataTotalDataFrame = lw(data=beamRSS)
                        xDataTotalDataFrame["tx_id"] = tx_id
                        # xDataTotalDataFrame.rename(columns={(beamRSS.shape[1]-1):'tx_id'},inplace=True)
                    # rename the last column to be tx_id
                    
                    returnDict = cU(runs_folder, n_samples, beamRSS, bestBeamIndices, posBestBeamIndices, realTotalBeamRSS,n_beams, num_epochs, train_batch_size, 
                            lr, decay_L2,  run_idx = run_idx, train_val_test_split = train_val_test_split, force_seed = force_seed, plotData=ln, top_beams = top_beams, findLR = findLR, findLR_train_batch_size=findLR_train_batch_size, findLR_init_value = findLR_init_value, findLR_final_value = findLR_final_value, fixed_GPU = lV, totalBeams = totalBeams, beamWeight = beamWeight, saveNameSuffix="", chooseLastEpochTrain = chooseLastEpochTrain, AddAverage = AddAverage, lr_scheduler=lr_schedule_params["name"], lr_scheduleParams=lr_schedule_params, modelType="deepIA", modelParams = modelParams, bandwidth = bandwidth,dropOutage=dropOutage, loadModel = loadModel, loadModelPath = loadModelPath_sub, testModelDict=testModelDict)
                    if not findLR:
                        bestModelPath = returnDict["modelPath"]
                        if (extraNumCols > 0) and (trainMode > 4):
                            
                            # testAcc_msb, testacc_txs,channelEffi_msb_acc, channelEffiacc_txs = test_netWrapper_multiTx(runs_folder, returnDict["x_train"], returnDict["y_train"], returnDict["x_test"],returnDict["y_test"],n_beams, bestModelPath, returnDict["x_testRowIndex"], xDataTotalDataFrame, yDataTotalDataFrame, posBestBeamDataFrame, realTotalBeamRSSDataFrame,scen_idx,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testAcc_msb, testacc_txs, channelEffi_msb_acc, channelEffiacc_txs, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="deepIA", modelParams=modelParams)
                            
                            testAcc_msb, testAcc_txs_msb,channelEffi_msb_acc, channelEffi_txs_msb_acc,testNetWrapperReturnDict = cR(runs_folder, returnDict["x_train"], returnDict["y_train"], returnDict["x_test"],returnDict["y_test"],n_beams, bestModelPath, returnDict["x_testRowIndex"], xDataTotalDataFrame, yDataTotalDataFrame, posBestBeamDataFrame, realTotalBeamRSSDataFrame,scen_idx,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testAcc_msb, testAcc_txs_msb, channelEffi_msb_acc, channelEffi_txs_msb_acc, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="deepIA", modelParams=modelParams)
                        else:
                            testAcc_msb, channelEffi_msb_acc,testNetWrapperReturnDict = cp(runs_folder, returnDict["x_train"], returnDict["y_train"], returnDict["x_test"],returnDict["y_test"],returnDict["z_test"], realTotalBeamRSS,n_beams, bestModelPath, returnDict["x_testRowIndex"], scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testAcc_msb,channelEffi_msb_acc, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams,modelType="deepIA", modelParams=lX, bandwidth=bandwidth)
                    
                    
                elif algo == "SFS":

                    # run_folder = runs_folder
                    # set accuracy 
                    acc_best = -1
                    bestBeamIndex = -1
                    # path that saves the best model
                    bestModelPath = ""        
                    # to indicate whether the lr find function alreday run for this M-val
                    findLR_run = ln
                    bestReturnDict = {}
                    if not loadModel:
                        for b in rxSet:
                            if findLR:
                                if not findLR_run:
                                    findLR_run = lV
                                    b = rxBeamPreSeq[M]
                                    bestBeamIndex = b
                                else:
                                    break
                            # add b to M temp
                            M_temp = sfs_beamSet.union(lf([b]))
                            # construct data set
                            beamRSS = cr(M_temp, allBeamRSS, txNum = txNum, rxNum = rxNum,offset=beamSetOffset, keepFront = keepFrontArr)
                            
                            if (extraNumCols > 0) and (trainMode > 4):
                                xDataTotalDataFrame = lw(data=beamRSS)
                                # xDataTotalDataFrame.rename(columns={(beamRSS.shape[1]-1):'tx_id'},inplace=True)
                                xDataTotalDataFrame["tx_id"] = tx_id
                            # beamRSS = normalize_rss(beamRSS, norm_type)
                            run_folder = rO.join(runs_folder, f"b{b}")
                            # train and validate
                        
                    # ----------------- Phase 3: Define Path for run --------------------------
                    
                    # We first define the folder where results from this run will be saved
                    # In that folder there will be other runs too, and that will tell us what's
                    # the index of this run. That information is used to shuffle the data 
                    # in a reproducible way. Run 1 uses seed 1, run 2 uses seed 2, etc.
                    
                            n = -1 # ignore this. (for compatibility with other predictors)

                            
                            # Check if there are enough runs. If yes, skip data loading, model
                            # training and testing, and jump to averaging the results.
                            if run_idx > n_avgs:
                                lS('Already enough experiments conducted for ' 
                                    'this case. Either increase n_avgs, or try '
                                    'a different set of parameters. SKIPPING TO the avg. '
                                    'computation!')
                            else:
                                # -------------------- Phase 4: Split Data ------------------------
                                
                                    returnDict = cU(run_folder, n_samples, beamRSS, bestBeamIndices, posBestBeamIndices,realTotalBeamRSS, n_beams,  num_epochs, train_batch_size, 
                                lr, decay_L2,  run_idx = run_idx,train_val_test_split = train_val_test_split, force_seed = force_seed, plotData=ln, top_beams = top_beams, findLR = findLR, findLR_train_batch_size=1024, findLR_init_value = 2e-3, findLR_final_value = 2e-1, fixed_GPU = lV, totalBeams = totalBeams, beamWeight = beamWeight, saveNameSuffix=li(b), chooseLastEpochTrain = chooseLastEpochTrain, AddAverage = AddAverage, lr_scheduler=lr_schedule_params["name"], lr_scheduleParams=lr_schedule_params,modelType="deepIA", modelParams = modelParams, bandwidth = bandwidth, dropOutage=dropOutage, loadModel = loadModel, loadModelPath = loadModelPath, testModelDict=testModelDict)
                                    
                                    acc_temp = returnDict["bestAcc"]
                                    
                                    # test 
                                    if testEachModel:
                                        # pred_beam_model = test_net(returnDict["x_test"], returnDict["model"], batch_size = 1024, beamNums = 5)
                                        cp(run_folder, returnDict["x_train"], returnDict["y_train"], returnDict["x_test"],returnDict["y_test"], returnDict["z_test"], realTotalBeamRSS,n_beams, returnDict["modelPath"], returnDict["x_testRowIndex"], scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, testAcc_sfs, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams= top_beams, modelType="deepIA", modelParams=lX, bandwidth=bandwidth)
                                    if acc_temp > acc_best:
                                        acc_best = acc_temp
                                        bestBeamIndex = b
                                        bestModelPath = returnDict["modelPath"]
                                        bestReturnDict = returnDict
                                        # # TODO
                                        # # temp delete for debug
                                        # break
                            # best beam added
                                    plt.close("all")
                        
                        sfs_beamSet = sfs_beamSet.union([bestBeamIndex])
                        if (lh(rxSet) >= 1):
                            rxSet.remove(bestBeamIndex)
                        rxBeamSeq.append(bestBeamIndex)
                        bestModelPathSeq.append(bestModelPath)
                        lS("best beam set")
                        lS(sfs_beamSet)
                    else:
                        # load the best beam, already provided
                        # rxBeamSeq = np.loadtxt(bestBeamIndexPath)
                        # rxBeamSeq = json.loads(bestBeamIndeexPath)
                        with lH(bestBeamIndexPath) as f:
                            lines = [line.strip() for line in f]
                        
                        rxBeamSeq = lL(lines[0])
                        # beamRSS = getBeamSetRSS(set(np.array(rxBeamSeq).astype(int)), allBeamRSS, txNum = txNum, rxNum = rxNum,offset=beamSetOffset, keepFront = keepFrontArr)
                        beamRSS = cr(lf(rxBeamSeq), allBeamRSS, txNum = txNum, rxNum = rxNum,offset=beamSetOffset, keepFront = keepFrontArr)
                        
                        with lH(bestModelPathtxt) as f:
                            lines = [line.strip() for line in f]
                        bestModel_tf_path = lL(lines[0])[-1]
                        if (extraNumCols > 0) and (trainMode > 4):
                            xDataTotalDataFrame = lw(data=beamRSS)
                            # xDataTotalDataFrame.rename(columns={(beamRSS.shape[1]-1):'tx_id'},inplace=True)
                            xDataTotalDataFrame["tx_id"] = tx_id
                        
                        bestReturnDict = cU(runs_folder, n_samples, beamRSS, bestBeamIndices, posBestBeamIndices,realTotalBeamRSS, n_beams,  num_epochs, train_batch_size, 
                                lr, decay_L2,  run_idx = run_idx,train_val_test_split = train_val_test_split, force_seed = force_seed, plotData=ln, top_beams = top_beams, findLR = findLR, findLR_train_batch_size=1024, findLR_init_value = 2e-3, findLR_final_value = 2e-1, fixed_GPU = lV, totalBeams = totalBeams, beamWeight = beamWeight, saveNameSuffix="", chooseLastEpochTrain = chooseLastEpochTrain, AddAverage = AddAverage, lr_scheduler=lr_schedule_params["name"], lr_scheduleParams=lr_schedule_params,modelType="deepIA", modelParams = modelParams, bandwidth = bandwidth, dropOutage=dropOutage, loadModel = loadModel, loadModelPath = bestModel_tf_path, testModelDict=testModelDict)                        
                    if (not findLR) and (not loadModel):
                        with lH(rO.join(runs_folder, "bestBeam_json.txt"), "w") as f:
                            # json.dump(list(sfs_beamSet),f)
                            lC(rxBeamSeq,f)
                        # np.savetxt(os.path.join(runs_folder, "bestBeam.txt"), rxBeamSeq)
                        wq(rO.join(runs_folder, "bestBeam.txt"), rG(rxBeamSeq).astype(lz), fmt="%i",delimiter=",")
                        
                        with lH(rO.join(runs_folder, "bestModelPath.txt"), "w") as f:
                            lC(bestModelPathSeq, f)
                        # copy the validation from the corresponding path
                        lm(returnDict["position_beam_val_acc_path_png"], runs_folder)
                        # shutil.copy(returnDict["position_beam_val_acc_path_pdf"], runs_folder)
                        lm(returnDict["training_vs_validation_loss_png"], runs_folder)
                        # shutil.copy(returnDict["training_vs_validation_loss_pdf"], runs_folder)
                        lm(returnDict["position_beam_val_channelEffi_png"], runs_folder)
                        lm(returnDict["position_beam_val_channelEffi_withRef_png"], runs_folder)
                        lm(returnDict["position_beam_val_channelEffi_withRef_total_png"], runs_folder)

                        # Load weights of best trained model
                        # trained_model = copy.deepcopy(model)
                        # trained_model = NN_FCN(x_train.shape[-1], n_beams, nodes_per_layer, layers)
                    # test and predict map
                        # beamRSS = getBeamSetRSS(sfs_beamSet, allBeamRSS, txNum = txNum, rxNum = rxNum,offset=beamSetOffset)
                    if (not findLR):
                        if (extraNumCols > 0) and (trainMode > 4):
                            testAcc_sfs, testAcc_txs_sfs,channelEffi_sfs_acc, channelEffi_txs_sfs_acc,testNetWrapperReturnDict = cR(runs_folder, bestReturnDict["x_train"], bestReturnDict["y_train"], bestReturnDict["x_test"],bestReturnDict["y_test"],n_beams, bestReturnDict["modelPath"], bestReturnDict["x_testRowIndex"], xDataTotalDataFrame, yDataTotalDataFrame, posBestBeamDataFrame, realTotalBeamRSSDataFrame,scen_idx,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testAcc_sfs, testAcc_txs_sfs, channelEffi_sfs_acc, channelEffi_txs_sfs_acc, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="deepIA", modelParams=modelParams)
                        else:
                            testAcc_sfs, channelEffi_sfs_acc, testNetWrapperReturnDict = cp(runs_folder, bestReturnDict["x_train"], bestReturnDict["y_train"], bestReturnDict["x_test"],bestReturnDict["y_test"], bestReturnDict["z_test"],realTotalBeamRSS, n_beams, bestReturnDict["modelPath"], bestReturnDict["x_testRowIndex"], scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, testAcc_sfs, channelEffi_sfs_acc,extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams= top_beams, modelType="deepIA", modelParams=lX, bandwidth=bandwidth)
                    
                        # testAcc_sfs, channelEffi_sfs_acc, testNetWrapperReturnDict = test_netWrapper(runs_folder, bestReturnDict["x_train"], bestReturnDict["y_train"], bestReturnDict["x_test"],bestReturnDict["y_test"], bestReturnDict["z_test"],realTotalBeamRSS, n_beams, bestModelPath, bestReturnDict["x_testRowIndex"], scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, testAcc_sfs, channelEffi_sfs_acc,extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams= top_beams, modelType="deepIA", modelParams=None, bandwidth=bandwidth)
                    

        # save the testAcc_sfs
        if not findLR:
            # plt.figure()
            if lh(algos) == 2:
                cF(wW((testAcc_sfs[[0,-1],:], testAcc_msb[[0,-1],:])),top_runs_folder,"accM.png","Accuracy [%]", title="Accuracy", xlabel="M", selectRows = [0,1,2,3], labels=["sfs top1", "sfs top5", "msb top1", "msb top5"])
                
                # plot channel capacity
                cF(wW((channelEffi_sfs_acc[[0,-1],:], channelEffi_msb_acc[[0,-1],:])),top_runs_folder,"channelM.png","Achievable Rate [%]", title="Achievable Rate", xlabel="M", selectRows = [0,1,2,3], labels=["sfs top1", "sfs top5","msb top1", "msb top5"])
                
                # plot channel capacity
                refAcc = testNetWrapperReturnDict["channelEffi_ref"] * 100
                refAcc = wC(wE(refAcc,-1), (1, channelEffi_sfs_acc.shape[-1]))


                cF(wW((channelEffi_sfs_acc[[0,-1],:], channelEffi_msb_acc[[0,-1],:], refAcc[[0,-1],:])),top_runs_folder,"channelM_withRef.png","Achievable Rate [%]", title="Achievable Rate", xlabel="M", selectRows = [0,1,2,3,4,5], labels=["sfs top1", "sfs top5","msb top1", "msb top5", "ref top1", "ref top5"], top_beams=[1,2,3,4,5,6])                
                # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[0,:],"b*-")
                # plt.plot(np.arange(1, testAcc_msb.shape[-1] + 1), testAcc_msb[0,:],"m*-")
                
                # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[4,:],"g*-")
                # plt.plot(np.arange(1, testAcc_msb.shape[-1] + 1), testAcc_msb[4,:],"c*-")
                # plt.xlabel("M")
                # plt.ylabel("Accuracy [%]")
                # plt.title("Accuracy")
                # plt.legend(["sfs top1","msb top1", "sfs top5", "msb top5"])
                # plt.savefig(os.path.join(top_runs_folder, "accM.png"))
                # plt.show()
            lS("sfs accuracy:")
            lS(testAcc_sfs)
            lS("msb accuracy:")
            lS(testAcc_msb)
            wq(rO.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
            
            wq(rO.join(top_runs_folder, "msb_accuracy.txt"), testAcc_msb, fmt='%.2f')
            lS("acc saved at: " + rO.join(top_runs_folder, "sfs_accuracy.txt"))
            
            wq(rO.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
            # separte top-k
            if "SFS" in algos:
                cF(testAcc_sfs, top_runs_folder, "accM_sfs.png", "Accuracy [%]",top_beams,"Accuracy", "M")
                
                cF(channelEffi_sfs_acc, top_runs_folder, "accM_channelEffi_sfs_select.png", "Achievable Rate", top_beams, "", "M", plotRef = lV, refacc=wm(testNetWrapperReturnDict["channelEffi_ref"] * 100), selectRows=[0,2,4])
                
                cF(channelEffi_sfs_acc, top_runs_folder, "accM_channelEffi_sfs.png", "Achievable Rate", top_beams, "", "M", plotRef = lV, refacc=wm(testNetWrapperReturnDict["channelEffi_ref"] * 100))
                
                # plt.figure()
                    
                # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[0,:]*100, 'g*-', lw=2.0, label='Top-1 Accuracy')
                # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[1,:]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
                # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[2,:]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
                # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[3,:]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
                # plt.xlabel("M")
                # plt.ylabel("Accuracy [%]")
                # plt.title("Accuracy")
                # plt.legend()
                # plt.savefig(os.path.join(top_runs_folder, "accM_sfs.png"))
                # plt.show()
                
            # msb
            if "MSB" in algos:
                cF(testAcc_msb, top_runs_folder, "accM_msb.png", "Accuracy [%]",top_beams,"Accuracy", "M")
                cF(channelEffi_msb_acc, top_runs_folder, "accM_channelEffi_msb_select.png", "Achievable Rate", top_beams, "", "M",plotRef = lV, refacc=wm(testNetWrapperReturnDict["channelEffi_ref"] * 100), selectRows=[0,2,4])
                cF(channelEffi_msb_acc, top_runs_folder, "accM_channelEffi_msb.png", "Achievable Rate", top_beams, "", "M",plotRef = lV, refacc=wm(testNetWrapperReturnDict["channelEffi_ref"] * 100))
                
                # plt.figure()
                    
                # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[0,:]*100, 'g*-', lw=2.0, label='Top-1 Accuracy')
                # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[1,:]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
                # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[2,:]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
                # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[3,:]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
                # plt.xlabel("M")
                # plt.ylabel("Accuracy [%]")
                # plt.title("Accuracy")
                # plt.legend()
                # plt.savefig(os.path.join(top_runs_folder, "accM_msb.png"))
                # plt.show()
                
            
            lS("sfs accuracy:")
            lS(testAcc_sfs)
            lS("msb accuracy:")
            lS(testAcc_msb)
            wq(rO.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
            
            wq(rO.join(top_runs_folder, "msb_accuracy.txt"), testAcc_msb, fmt='%.2f')
            lS("acc saved at: " + rO.join(top_runs_folder, "sfs_accuracy.txt"))
            



            lS("sfs channelEffi:")
            lS(testAcc_sfs)
            lS("msb channelEffi:")
            lS(testAcc_msb)
            wq(rO.join(top_runs_folder, "sfs_channelEffi.txt"), channelEffi_sfs_acc, fmt='%.2f')
            
            wq(rO.join(top_runs_folder, "msb_channelEffi.txt"), channelEffi_msb_acc, fmt='%.2f')
            wq(rO.join(top_runs_folder, "channelEffi_ref.txt"), testNetWrapperReturnDict["channelEffi_ref"] * 100, fmt='%.2f')
            lS("channel effi saved at: " + rO.join(top_runs_folder, "sfs_channelEffi.txt"))
            
            # np.savetxt(os.path.join(top_runs_folder, "sfs_channelEffi_txs.txt"), channelEffi_txs_sfs_acc, fmt='%.2f')
            
            # np.savetxt(os.path.join(top_runs_folder, "msb_channelEffi_txs.txt"), channelEffi_txs_msb_acc, fmt='%.2f')
            
            # savetxt cannot save 3d, so save binary
            if (extraNumCols > 0) and (trainMode > 4):
                wP(rO.join(top_runs_folder, "sfs_channelEffi_txs.npy"), channelEffi_txs_sfs_acc)
                
                
                wP(rO.join(top_runs_folder, "msb_channelEffi_txs.npy"), channelEffi_txs_msb_acc)
                
                wP(rO.join(top_runs_folder, "sfs_accuracy_txs.npy"), testAcc_txs_sfs)
                
                wP(rO.join(top_runs_folder, "msb_accuracy_txs.npy"), testAcc_txs_msb)
            
    elif bm_method == "narrowLabel":
    #  try to find the best beam indices for each subregion
    # should be a list of coordinates to first filter the dataset
        sfs_beamSet = lf()
        modelParams = lX
        rxBeamSeq = [] # incrementally added rx beam index
        bestModelPathSeq = [] # incrementally add best model saved path
        rxSet = lf(lP(lq(1,rxIndexMax))) # index 
        squareBoundLen = lh(squareBoundCoords)
        if squareBoundLen % 2:
            raise lW("uneven square bound length")
        totalDataPath = rO.join(data_folder, totalDataMatrixName)
        # totalDataMatrix = pd.read_csv(totalDataPath, dtype={"dataTotalMatrix847": "float"})
        totalDataMatrix = lc(totalDataPath)
        # get list names
        totalDataMatrixColNames = lP(totalDataMatrix.columns.values)
        wideBeamDataPath = rO.join(data_folder, dataMatrixWideBeamName)
        # totalDataMatrix = pd.read_csv(totalDataPath, dtype={"dataTotalMatrix847": "float"})
        wideBeamDataMatrix = lc(wideBeamDataPath)
        # wideBeamDataMatrixColNames = wideBeamDataMatrix.columns
        wideBeamDataMatrixColNames = lP(wideBeamDataMatrix.columns.values)
        
        for i in lq(0,squareBoundLen,2):
            lowBound = squareBoundCoords[i]
            upBound = squareBoundCoords[i+1]
            # select the data
            selectRowIndices = (totalDataMatrix.iloc[:,0] >= lowBound[0]) & (totalDataMatrix.iloc[:,0] <= upBound[0]) & (totalDataMatrix.iloc[:,1] >= lowBound[1]) & (totalDataMatrix.iloc[:,1] <= upBound[1])
            totalDataMatrix_sub = totalDataMatrix.loc[selectRowIndices]
            wideBeamDataMatrix_sub = wideBeamDataMatrix.loc[selectRowIndices]
            lS(f'sub total Columns: {totalDataMatrix_sub.columns.values}')
            n_rows = totalDataMatrix_sub.shape[0]
            n_samples = lF(n_rows, max_samples)
            lS(f'Number of Rows: {n_rows}')
            
            totalDataMatrix_sub = cw(totalDataMatrix_sub, defaultVal=-174)
            wideBeamDataMatrix_sub = cw(wideBeamDataMatrix_sub, defaultVal = -174)
            realTotalBeamRSS = totalDataMatrix_sub.iloc[:,(2+totalBeams):].to_numpy()
            mapPos = totalDataMatrix_sub.iloc[:,[0,1]].to_numpy()
            if trainMode == 7:
                txPosCols = totalDataMatrix_sub[["tx_pos_x", "tx_pos_y"]].to_numpy()
                mapPos = wV((mapPos, txPosCols),axis=-1)
            mapPos = cy(mapPos,mapPos, norm_type, params=pos_params)
            # all beam rss original not normalized
            allBeamRSS_orig = totalDataMatrix_sub.iloc[:, (2 + bestBeamNum):].to_numpy()
            # normalize that
            # note that need to check if it is trainMode 4
            excludeCols = lX
            # should tell from the size that only the first n_beams should be normalized, i.e. for example tx_id is included
            extraNumCols = allBeamRSS_orig.shape[1] - n_beams
            if extraNumCols > 0:
                excludeCols = -extraNumCols
            # if trainMode == 4:
            #     excludeCols = -1
            # elif trainMode == 5:
            #     # the tx position and tx id should not be considered
            #     excludeCols = -3
                
            allBeamRSS = cs(allBeamRSS_orig, norm_type, params=rss_params, excludeCols=excludeCols)
            beamSetOffset = 0
            # note based on the arrangement of data, the indices go from 1 -> N, then N+1 -> .   rowwise
            bestBeamIndices = totalDataMatrix_sub.iloc[:,2: (2 + bestBeamNum)].to_numpy()
            posBestBeamIndices = wV([mapPos, totalDataMatrix_sub.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
            
        

            keepFrontArr = lX
            if trainMode == 6:
                beamSetOffset = 2
                keepFrontArr= rU(beamSetOffset)
            elif trainMode == 7:
                beamSetOffset = 4
                keepFrontArr= rU(beamSetOffset) 
                
            
            #  need to checl
            if (trainMode == 6)  or (trainMode == 7):
                allBeamRSS = wV([mapPos, allBeamRSS],-1)       
                raise lW("Not sure")
            
                # save the testing loss
            channelEffi_msb_acc = rt((n_top_stats,0))
            testAcc_sfs = rt((n_top_stats,0))
            testAcc_msb = rt((n_top_stats,0))
        
            channelEffi_sfs_acc = rt((n_top_stats,0))
            n_top_stats = lh(top_beams)
            boundIndexName = (f'B_{i}')
            rxBeamSeq = [] # incrementally added rx beam index
            bestModelPathSeq = [] # incrementally add best model saved path
            rxSet = lf(lP(lq(1,rxIndexMax))) # index 
            
            for M in lq(M_start, M_max_new):
                sub_runs_folder_name = (f'M_{M}')
                runs_folder = rO.join(top_runs_folder, boundIndexName,"SFS", sub_runs_folder_name)
                # runs_folder = os.path.join(top_runs_folder, boundIndexName,"SFS", sub_runs_folder_name)
                if loadModel:
                    loadModelPath_sub = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "model_checkpoint/nn_beam_pred")
                    bestBeamIndexPath = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "bestBeam_json.txt")
                    bestModelPathtxt = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "bestModelPath.txt")

                # Create if doesn't exist
                if not rO.exists(runs_folder):
                    rX(runs_folder)
                
                # Experiment index: number of experiments already conducted + 1
                run_idx = 1 + le(rO.isdir(rO.join(runs_folder, run_folder))
                                for run_folder in rn(runs_folder))
# run_folder = runs_folder
                    # set accuracy 
                if lV:
                    acc_best = -1
                    bestBeamIndex = -1
                    # path that saves the best model
                    bestModelPath = ""        
                    # to indicate whether the lr find function alreday run for this M-val
                    findLR_run = ln
                    bestReturnDict = {}
                    if not loadModel:
                        for b in rxSet:
                            if findLR:
                                if not findLR_run:
                                    findLR_run = lV
                                    b = rxBeamPreSeq[M]
                                    bestBeamIndex = b
                                else:
                                    break
                            # add b to M temp
                            M_temp = sfs_beamSet.union(lf([b]))
                            # construct data set
                            beamRSS = cr(M_temp, allBeamRSS, txNum = txNum, rxNum = rxNum,offset=beamSetOffset, keepFront = keepFrontArr)
                            
                            if (extraNumCols > 0) and (trainMode > 4):
                                xDataTotalDataFrame = lw(data=beamRSS)
                                # xDataTotalDataFrame.rename(columns={(beamRSS.shape[1]-1):'tx_id'},inplace=True)
                                xDataTotalDataFrame["tx_id"] = tx_id
                            # beamRSS = normalize_rss(beamRSS, norm_type)
                            run_folder = rO.join(runs_folder, f"b{b}")
                            # train and validate
                        
                    # ----------------- Phase 3: Define Path for run --------------------------
                    
                    # We first define the folder where results from this run will be saved
                    # In that folder there will be other runs too, and that will tell us what's
                    # the index of this run. That information is used to shuffle the data 
                    # in a reproducible way. Run 1 uses seed 1, run 2 uses seed 2, etc.
                    
                            n = -1 # ignore this. (for compatibility with other predictors)

                            
                            # Check if there are enough runs. If yes, skip data loading, model
                            # training and testing, and jump to averaging the results.
                            if run_idx > n_avgs:
                                lS('Already enough experiments conducted for ' 
                                    'this case. Either increase n_avgs, or try '
                                    'a different set of parameters. SKIPPING TO the avg. '
                                    'computation!')
                            else:
                                # -------------------- Phase 4: Split Data ------------------------
                                
                                    returnDict = cU(run_folder, n_samples, beamRSS, bestBeamIndices, posBestBeamIndices,realTotalBeamRSS, n_beams,  num_epochs, train_batch_size, 
                                lr, decay_L2,  run_idx = run_idx,train_val_test_split = train_val_test_split, force_seed = force_seed, plotData=ln, top_beams = top_beams, findLR = findLR, findLR_train_batch_size=1024, findLR_init_value = 2e-3, findLR_final_value = 2e-1, fixed_GPU = lV, totalBeams = totalBeams, beamWeight = beamWeight, saveNameSuffix=li(b), chooseLastEpochTrain = chooseLastEpochTrain, AddAverage = AddAverage, lr_scheduler=lr_schedule_params["name"], lr_scheduleParams=lr_schedule_params,modelType="deepIA", modelParams = modelParams, bandwidth = bandwidth, dropOutage=dropOutage, loadModel = loadModel, loadModelPath = loadModelPath, testModelDict=testModelDict)
                                    
                                    acc_temp = returnDict["bestAcc"]
                                    
                                    # test 
                                    if testEachModel:
                                        # pred_beam_model = test_net(returnDict["x_test"], returnDict["model"], batch_size = 1024, beamNums = 5)
                                        cp(run_folder, returnDict["x_train"], returnDict["y_train"], returnDict["x_test"],returnDict["y_test"], returnDict["z_test"], realTotalBeamRSS,n_beams, returnDict["modelPath"], returnDict["x_testRowIndex"], scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, testAcc_sfs, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams= top_beams, modelType="deepIA", modelParams=lX, bandwidth=bandwidth)
                                    if acc_temp > acc_best:
                                        acc_best = acc_temp
                                        bestBeamIndex = b
                                        bestModelPath = returnDict["modelPath"]
                                        bestReturnDict = returnDict
                                        # # TODO
                                        # # temp delete for debug
                                        # break
                            # best beam added
                                    plt.close("all")
                        
                        sfs_beamSet = sfs_beamSet.union([bestBeamIndex])
                        if (lh(rxSet) >= 1):
                            rxSet.remove(bestBeamIndex)
                        rxBeamSeq.append(bestBeamIndex)
                        bestModelPathSeq.append(bestModelPath)
                        lS("best beam set")
                        lS(sfs_beamSet)
                    else:
                        # load best beam 
                        raise lW("not supported load best beam")
                    if (not findLR) and (not loadModel):
                        with lH(rO.join(runs_folder, "bestBeam_json.txt"), "w") as f:
                            # json.dump(list(sfs_beamSet),f)
                            lC(rxBeamSeq,f)
                        # np.savetxt(os.path.join(runs_folder, "bestBeam.txt"), rxBeamSeq)
                        wq(rO.join(runs_folder, "bestBeam.txt"), rG(rxBeamSeq).astype(lz), fmt="%i",delimiter=",")
                        
                        with lH(rO.join(runs_folder, "bestModelPath.txt"), "w") as f:
                            lC(bestModelPathSeq, f)
                        # copy the validation from the corresponding path
                        lm(returnDict["position_beam_val_acc_path_png"], runs_folder)
                        # shutil.copy(returnDict["position_beam_val_acc_path_pdf"], runs_folder)
                        lm(returnDict["training_vs_validation_loss_png"], runs_folder)
                        # shutil.copy(returnDict["training_vs_validation_loss_pdf"], runs_folder)
                        lm(returnDict["position_beam_val_channelEffi_png"], runs_folder)
                        lm(returnDict["position_beam_val_channelEffi_withRef_png"], runs_folder)
                        lm(returnDict["position_beam_val_channelEffi_withRef_total_png"], runs_folder)

                        # Load weights of best trained model
                        # trained_model = copy.deepcopy(model)
                        # trained_model = NN_FCN(x_train.shape[-1], n_beams, nodes_per_layer, layers)
                    # test and predict map
                        # beamRSS = getBeamSetRSS(sfs_beamSet, allBeamRSS, txNum = txNum, rxNum = rxNum,offset=beamSetOffset)
                    if (not findLR):
                        if (extraNumCols > 0) and (trainMode > 4):
                            testAcc_sfs, testAcc_txs_sfs,channelEffi_sfs_acc, channelEffi_txs_sfs_acc,testNetWrapperReturnDict = cR(runs_folder, bestReturnDict["x_train"], bestReturnDict["y_train"], bestReturnDict["x_test"],bestReturnDict["y_test"],n_beams, bestReturnDict["modelPath"], bestReturnDict["x_testRowIndex"], xDataTotalDataFrame, yDataTotalDataFrame, posBestBeamDataFrame, realTotalBeamRSSDataFrame,scen_idx,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testAcc_sfs, testAcc_txs_sfs, channelEffi_sfs_acc, channelEffi_txs_sfs_acc, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="deepIA", modelParams=modelParams)
                        else:
                            testAcc_sfs, channelEffi_sfs_acc, testNetWrapperReturnDict = cp(runs_folder, bestReturnDict["x_train"], bestReturnDict["y_train"], bestReturnDict["x_test"],bestReturnDict["y_test"], bestReturnDict["z_test"],realTotalBeamRSS, n_beams, bestReturnDict["modelPath"], bestReturnDict["x_testRowIndex"], scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, testAcc_sfs, channelEffi_sfs_acc,extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams= top_beams, modelType="deepIA", modelParams=lX, bandwidth=bandwidth)                        
                            
            # find the best sequence
            # attach to the matrix
            totalDataMatrix_sub = wV((totalDataMatrix_sub, wC(rG(rxBeamSeq), (totalDataMatrix_sub.shape[0],1))), axis=-1)
            
            wideBeamDataMatrix_sub = wV((wideBeamDataMatrix_sub, wC(rG(rxBeamSeq), (wideBeamDataMatrix_sub.shape[0],1))), axis=-1)
            if i == 0:
                newColNames = ["beam_" + li(i) for i in lq(1,lh(rxBeamSeq) + 1)]
                totalDataMatrixColNames.extend(newColNames)
                wideBeamDataMatrixColNames.extend(newColNames)
                totalMatrixWithNarrowLabel = totalDataMatrix_sub
                wideBeamMatrixWithNarrowLabel = wideBeamDataMatrix_sub
            else:
                totalMatrixWithNarrowLabel = wV((totalMatrixWithNarrowLabel, totalDataMatrix_sub))
                wideBeamMatrixWithNarrowLabel = wV((wideBeamMatrixWithNarrowLabel, wideBeamDataMatrix_sub))
        # write the totalMatrix to the file
        
        outFileTotalPath = rO.join(data_folder, totalMatrixWithNarrowLabelName)
        wideBeamOutFilePath = rO.join(data_folder, wideBeamMatrixWithNarrowLabelName)
        totalMatrixWithNarrowLabel_df = lw(data=totalMatrixWithNarrowLabel, columns = totalDataMatrixColNames)
        wideBeamMatrixWithNarrowLabel_df = lw(data=wideBeamMatrixWithNarrowLabel, columns = wideBeamDataMatrixColNames)
        # save the data
        totalMatrixWithNarrowLabel_df.to_csv(outFileTotalPath, index=ln)
        wideBeamMatrixWithNarrowLabel_df.to_csv(wideBeamOutFilePath, index=ln)
                    
    elif bm_method == "twostage":
        
        sfs_beamSet = lf()
        modelParams = lX
        rxBeamSeq = [] # incrementally added rx beam index
        bestModelPathSeq = [] # incrementally add best model saved path
        rxSet = lf(lP(lq(1,rxIndexMax))) # index 
        squareBoundLen = lh(squareBoundCoords)
        if squareBoundLen % 2:
            raise lW("uneven square bound length")
        totalDataPath = rO.join(data_folder, totalMatrixWithNarrowLabelName)
        # totalDataMatrix = pd.read_csv(totalDataPath, dtype={"dataTotalMatrix847": "float"})
        totalDataMatrix = lc(totalDataPath)
        # get list names
        totalDataMatrixColNames = lP(totalDataMatrix.columns.values)
        # wide beam
        wideBeamDataPath = rO.join(data_folder, wideBeamMatrixWithNarrowLabelName)
        # totalDataMatrix = pd.read_csv(totalDataPath, dtype={"dataTotalMatrix847": "float"})
        wideBeamDataMatrix = lc(wideBeamDataPath)
        # wideBeamDataMatrixColNames = wideBeamDataMatrix.columns
        wideBeamDataMatrixColNames = lP(wideBeamDataMatrix.columns.values)
        
        # total number of subset of second stage
        # M_max_new
        totalDataMatrix = cw(totalDataMatrix)
        totalDataMatrix["row_no"] = totalDataMatrix.index
        
        mapPos = totalDataMatrix.iloc[:,[0,1]].to_numpy()
        wideMapPos = wideBeamDataMatrix.iloc[:,[0,1]].to_numpy()
        
        if trainMode == 7:
            txPosCols = totalDataMatrix[["tx_pos_x", "tx_pos_y"]].to_numpy()
            mapPos = wV((mapPos, txPosCols),axis=-1)
            wideTXPosCols = wideBeamDataMatrix[["tx_pos_x","tx_pos_y"]].to_numpy()
            wideMapPos = wV((wideMapPos, wideTXPosCols),axis=-1)
            
        mapPos = cy(mapPos,mapPos, norm_type, params=pos_params)
        # get the index where the narrow Label column starts
        totalData_narrowLabelStart = totalDataMatrix.columns.get_loc("beam_1")
        narrowBeamRSS_narrowLabelEnd = totalDataMatrix.columns.get_loc("beam_" + li(wideBeamTargetNums))
        allBeamRSS_orig = totalDataMatrix.iloc[:, (2 + bestBeamNum):totalData_narrowLabelStart].to_numpy()
        allBeamRSS = cs(allBeamRSS_orig, norm_type, params=rss_params)
        
        wideBeamRSS_narrowLabelStart = wideBeamDataMatrix.columns.get_loc("beam_1")
        wideBeamRSS_narrowLabelEnd = wideBeamDataMatrix.columns.get_loc("beam_" + li(wideBeamTargetNums))
        wideBeamRSS = wideBeamDataMatrix.iloc[:, (2 + bestBeamNum):wideBeamRSS_narrowLabelStart].to_numpy()
        wideBeamRSS = cs(allBeamRSS_orig, norm_type, params=rss_params)
        wideBeamNarrowLabel = wideBeamDataMatrix.iloc[:,wideBeamRSS_narrowLabelStart:(wideBeamRSS_narrowLabelEnd+1)].to_numpy()
        wide_realTotalBeamRSS = wideBeamDataMatrix.iloc[:,(2+totalBeams):wideBeamRSS_narrowLabelStart].to_numpy()
        wideposBestBeamIndices = lX
        # wideBeamRSS = wideBeamDataMatrix.iloc[:, (2 + bestBeamNum):(-M_max_new)].to_numpy()
        
        # extra work if txPos included
        if trainMode >= 6:
            # raise Exception("do not support")
            wideBeamRSS = wV([mapPos, wideBeamRSS],-1)
            allBeamRSS = wV([mapPos, allBeamRSS],-1)
        
        # wide_input_subsetIndices 
        
        # construct  pos, widebeamIndice, 
        
        # first train the widebeam
        runs_folder = top_runs_folder
        if not rO.exists(runs_folder):
            rX(runs_folder)
        
                # Experiment index: number of experiments already conducted + 1
        run_idx = 1 + le(rO.isdir(rO.join(runs_folder, run_folder))
                        for run_folder in rn(runs_folder))

        modelParams = {}
        modelParams["nodes_per_layer"]= nodes_per_layer 
        modelParams["layers"] = layers
        wide_n_samples = lF(wideBeamRSS.shape[0], max_samples)
        
        # nbeams = predRxNum
        wideBeamReturnDict = cU(runs_folder, wide_n_samples, wideBeamRSS, wideBeamNarrowLabel, wideposBestBeamIndices,wide_realTotalBeamRSS, predRxNum, num_epochs, train_batch_size, 
        lr, decay_L2,  run_idx = run_idx, train_val_test_split = train_val_test_split, force_seed = force_seed, plotData=ln, top_beams = top_beams, findLR = findLR, findLR_train_batch_size=findLR_train_batch_size, findLR_init_value = findLR_init_value, findLR_final_value = findLR_final_value, fixed_GPU = lV, totalBeams = totalBeams, beamWeight = widebeamNarrowLabelWeight, saveNameSuffix="", chooseLastEpochTrain = chooseLastEpochTrain, AddAverage = AddAverage, lr_scheduler=lr_schedule_params["name"], lr_scheduleParams=lr_schedule_params, modelType = "wideBeam", modelParams = modelParams,bandwidth=bandwidth, dropOutage=dropOutage, loadModel = loadModel, loadModelPath = loadModelPath, testModelDict=testModelDict, checkChannel=ln)
        
        if not findLR:
            wideBestModelPath = wideBeamReturnDict["modelPath"]
            # testacc = []
            testacc = rt((n_top_stats,0))
            channelEffi_acc = rt((n_top_stats,0))
            # if trainMode == 4 or trainMode == 5:
            if (trainMode > 4) and testMultiWideBeam:
                testacc_txs = rt((txIdNum,n_top_stats,0))
                channelEffiacc_txs = rt((txIdNum,n_top_stats,0))
                wideposBestBeamDataFrame = lX
                widerealTotalBeamRSSDataFrame = lX
                
                xDataTotalDataFrame = lw(data=wideBeamRSS)
                # xDataTotalDataFrame.rename(columns={(beamRSS.shape[1]-1):'tx_id'},inplace=True)
                xDataTotalDataFrame["tx_id"] = tx_id
                
                testacc, testacc_txs,channelEffi_acc, channelEffiacc_txs,testNetWrapperReturnDict = cR(runs_folder, wideBeamReturnDict["x_train"], wideBeamReturnDict["y_train"], wideBeamReturnDict["x_test"],wideBeamReturnDict["y_test"],predRxNum, wideBestModelPath, wideBeamReturnDict["x_testRowIndex"], xDataTotalDataFrame, yDataTotalDataFrame, wideposBestBeamDataFrame, widerealTotalBeamRSSDataFrame,scen_idx,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testacc, testacc_txs, channelEffi_acc, channelEffiacc_txs, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="pos", modelParams=modelParams,plotPredFig = ln, testBatchSize = train_batch_size, checkChannel = ln)
            else:
                testacc, channelEffi_acc,testNetWrapperReturnDict = cp(runs_folder, wideBeamReturnDict["x_train"], wideBeamReturnDict["y_train"], wideBeamReturnDict["x_test"],wideBeamReturnDict["y_test"], wideBeamReturnDict["z_test"], wide_realTotalBeamRSS,predRxNum, wideBestModelPath, wideBeamReturnDict["x_testRowIndex"], scen_idx,wideposBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testacc, channelEffi_acc,extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="pos", modelParams=modelParams, bandwidth=bandwidth, plotPredFig = ln, testBatchSize = train_batch_size, checkChannel = ln)      
                
        # narrow beam
        
        sfs_beamSet = lf()
        modelParams = lX
        rxBeamSeq = [] # incrementally added rx beam index
        bestModelPathSeq = [] # incrementally add best model saved path
        rxSet = lf(lP(lq(1,rxIndexMax))) # index 
        squareBoundLen = lh(squareBoundCoords)
        if squareBoundLen % 2:
            raise lW("uneven square bound length")

        # construct dictionary for accessing later
        beamsetToModelDict= lv()
        ModelTobeamsetDict = lv()
        beamsetMatrices = []
        beamsetToIndices = lv()
        indicesToBeamset = lv()
        modelToIndices = lv()
        indicesToModel = lv()
        
        for i in lq(0,squareBoundLen,2):
            lowBound = squareBoundCoords[i]
            upBound = squareBoundCoords[i+1]
            # select the data
            selectRowIndices = (totalDataMatrix.iloc[:,0] >= lowBound[0]) & (totalDataMatrix.iloc[:,0] <= upBound[0]) & (totalDataMatrix.iloc[:,1] >= lowBound[1]) & (totalDataMatrix.iloc[:,1] <= upBound[1])
            totalDataMatrix_sub = totalDataMatrix.loc[selectRowIndices]
            wideBeamDataMatrix_sub = wideBeamDataMatrix.loc[selectRowIndices]
            lS(f'sub total Columns: {totalDataMatrix_sub.columns.values}')
            n_rows = totalDataMatrix_sub.shape[0]
            n_samples = lF(n_rows, max_samples)
            lS(f'Number of Rows: {n_rows}')
            # get the beam set
            column_sub = totalDataMatrix_sub.iloc[0, totalData_narrowLabelStart:(narrowBeamRSS_narrowLabelEnd+1)].to_numpy(dtype=lz)
            
            sfs_beamSet = lf(column_sub)
            totalDataMatrix_sub = cw(totalDataMatrix_sub, defaultVal=-174)
            wideBeamDataMatrix_sub = cw(wideBeamDataMatrix_sub, defaultVal = -174)
            realTotalBeamRSS = totalDataMatrix_sub.iloc[:,(2+totalBeams):].to_numpy()
            mapPos_narrowTrain = totalDataMatrix_sub.iloc[:,[0,1]].to_numpy()
            if trainMode == 7:
                txPosCols = totalDataMatrix_sub[["tx_pos_x", "tx_pos_y"]].to_numpy()
                mapPos_narrowTrain = wV((mapPos_narrowTrain, txPosCols),axis=-1)
            if trainMode > 4:
                tx_id = totalDataMatrix_sub["tx_id"].to_numpy()
            mapPos_narrowTrain = cy(mapPos_narrowTrain,mapPos_narrowTrain, norm_type, params=pos_params)
            # all beam rss original not normalized
            allBeamRSS_orig = totalDataMatrix_sub.iloc[:, (2 + bestBeamNum):totalData_narrowLabelStart].to_numpy()
            # normalize that
            # note that need to check if it is trainMode 4
            excludeCols = lX
            # should tell from the size that only the first n_beams should be normalized, i.e. for example tx_id is included
            extraNumCols = allBeamRSS_orig.shape[1] - n_beams
            if extraNumCols > 0:
                excludeCols = -extraNumCols
            # if trainMode == 4:
            #     excludeCols = -1
            # elif trainMode == 5:
            #     # the tx position and tx id should not be considered
            #     excludeCols = -3
                
            allBeamRSS_narrowTrain = cs(allBeamRSS_orig, norm_type, params=rss_params, excludeCols=excludeCols)
            beamSetOffset = 0
            # note based on the arrangement of data, the indices go from 1 -> N, then N+1 -> .   rowwise
            bestBeamIndices = totalDataMatrix_sub.iloc[:,2: (2 + bestBeamNum)].to_numpy()
            posBestBeamIndices = wV([mapPos_narrowTrain, totalDataMatrix_sub.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
            
            n_samples = lF(totalDataMatrix_sub.shape[0], max_samples)

            keepFrontArr = lX
            if trainMode == 6:
                beamSetOffset = 2
                keepFrontArr= rU(beamSetOffset)
            elif trainMode == 7:
                beamSetOffset = 4
                keepFrontArr= rU(beamSetOffset) 
                
            
            #  need to checl
            if (trainMode == 6)  or (trainMode == 7):
                allBeamRSS_narrowTrain = wV([mapPos_narrowTrain, allBeamRSS_narrowTrain],-1)       
                raise lW("Not sure")
            
                # save the testing loss
            channelEffi_msb_acc = rt((n_top_stats,0))
            testAcc_sfs = rt((n_top_stats,0))
            testAcc_msb = rt((n_top_stats,0))
        
            channelEffi_sfs_acc = rt((n_top_stats,0))
            n_top_stats = lh(top_beams)
            boundIndexName = (f'B_{i}')
            rxBeamSeq = [] # incrementally added rx beam index
            bestModelPathSeq = [] # incrementally add best model saved path
            rxSet = lf(lP(lq(1,rxIndexMax))) # index 
            sfs_set_sub = lf(column_sub)
            # for M in range(M_start, M_max_new):
            sub_runs_folder_name = (f'M_{wideBeamTargetNums}')
            runs_folder = rO.join(top_runs_folder_narrow, boundIndexName,"SFS", sub_runs_folder_name)
                # runs_folder = os.path.join(top_runs_folder, boundIndexName,"SFS", sub_runs_folder_n§ame)
            if loadModel:
                loadModelPath_sub = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "model_checkpoint/nn_beam_pred")
                bestBeamIndexPath = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "bestBeam_json.txt")
                bestModelPathtxt = rO.join(loadModelPath, "SFS", sub_runs_folder_name, "bestModelPath.txt")

            # Create if doesn't exist
            if not rO.exists(runs_folder):
                rX(runs_folder)
            
            # Experiment index: number of experiments already conducted + 1
            run_idx = 1 + le(rO.isdir(rO.join(runs_folder, run_folder))
                            for run_folder in rn(runs_folder))
# run_folder = runs_folder
                # set accuracy 
            if lV:
                acc_best = -1
                bestBeamIndex = -1
                # path that saves the best model
                bestModelPath = ""        
                # to indicate whether the lr find function alreday run for this M-val
                findLR_run = ln
                bestReturnDict = {}
                if not loadModel:
                    # construct data set
                    beamRSS = cr(sfs_set_sub, allBeamRSS_narrowTrain, txNum = txNum, rxNum = rxNum,offset=beamSetOffset, keepFront = keepFrontArr)
                    
                    if (extraNumCols > 0) and (trainMode > 4):
                        xDataTotalDataFrame = lw(data=beamRSS)
                        # xDataTotalDataFrame.rename(columns={(beamRSS.shape[1]-1):'tx_id'},inplace=True)
                        xDataTotalDataFrame["tx_id"] = tx_id
                    # run_folder = runs_folder
                    # # beamRSS = normalize_rss(beamRSS, norm_type)
                    run_folder = rO.join(runs_folder, f"B_{i}")
                    # train and validate
                
            # ----------------- Phase 3: Define Path for run --------------------------
            
            # We first define the folder where results from this run will be saved
            # In that folder there will be other runs too, and that will tell us what's
            # the index of this run. That information is used to shuffle the data 
            # in a reproducible way. Run 1 uses seed 1, run 2 uses seed 2, etc.
            
                    n = -1 # ignore this. (for compatibility with other predictors)

                    
                    # Check if there are enough runs. If yes, skip data loading, model
                    # training and testing, and jump to averaging the results.
                    if run_idx > n_avgs:
                        lS('Already enough experiments conducted for ' 
                            'this case. Either increase n_avgs, or try '
                            'a different set of parameters. SKIPPING TO the avg. '
                            'computation!')
                    else:
                        # -------------------- Phase 4: Split Data ------------------------
                        
                        returnDict = cU(run_folder, n_samples, beamRSS, bestBeamIndices, posBestBeamIndices,realTotalBeamRSS, n_beams,  num_epochs, train_batch_size, 
                        lr, decay_L2,  run_idx = run_idx,train_val_test_split = train_val_test_split, force_seed = force_seed, plotData=ln, top_beams = top_beams, findLR = findLR, findLR_train_batch_size=1024, findLR_init_value = 2e-3, findLR_final_value = 2e-1, fixed_GPU = lV, totalBeams = totalBeams, beamWeight = beamWeight, saveNameSuffix="B_" + li(i), chooseLastEpochTrain = chooseLastEpochTrain, AddAverage = AddAverage, lr_scheduler=lr_schedule_params["name"], lr_scheduleParams=lr_schedule_params,modelType="deepIA", modelParams = modelParams, bandwidth = bandwidth, dropOutage=dropOutage, loadModel = loadModel, loadModelPath = loadModelPath, testModelDict=testModelDict)
                        
                        acc_temp = returnDict["bestAcc"]
                        
                    #         # test 
                    #         if testEachModel:
                    #             # pred_beam_model = test_net(returnDict["x_test"], returnDict["model"], batch_size = 1024, beamNums = 5)
                    #             test_netWrapper(run_folder, returnDict["x_train"], returnDict["y_train"], returnDict["x_test"],returnDict["y_test"], returnDict["z_test"], realTotalBeamRSS,n_beams, returnDict["modelPath"], returnDict["x_testRowIndex"], scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, testAcc_sfs, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams= top_beams, modelType="deepIA", modelParams=None, bandwidth=bandwidth)
                    #         if acc_temp > acc_best:
                    #             acc_best = acc_temp
                    #             bestBeamIndex = b
                        bestModelPath = returnDict["modelPath"]
                        bestReturnDict = returnDict
                    #             # # TODO
                    #             # # temp delete for debug
                    #             # break
                    # # best beam added
                            # plt.close("all")
                    
                    # sfs_beamSet = sfs_beamSet.union([bestBeamIndex])
                    # if (len(rxSet) >= 1):
                    #     rxSet.remove(bestBeamIndex)
                    rxBeamSeq.append(bestBeamIndex)
                    bestModelPathSeq.append(bestModelPath)
                    lS("best beam set")
                    lS(sfs_beamSet)
                    rxBeamSeq = lP(sfs_beamSet)
                    beamsetToModelDict[mc(sfs_beamSet)] = bestModelPath
                    ModelTobeamsetDict[bestModelPath] = sfs_beamSet
                    beamsetMatrices.append(lP(sfs_beamSet))
                    beamsetToIndices[mc(sfs_beamSet)] = i//2
                    indicesToBeamset[i//2] = sfs_beamSet
                    modelToIndices[bestModelPath] = i//2
                    indicesToModel[i//2] = bestModelPath
                else:
                    # load best beam 
                    raise lW("not supported load best beam")
                if (not findLR) and (not loadModel):
                    # with open(os.path.join(runs_folder, "bestBeam_json.txt"), "w") as f:
                    #     # json.dump(list(sfs_beamSet),f)
                    #     json.dump(rxBeamSeq,f)
                    # np.savetxt(os.path.join(runs_folder, "bestBeam.txt"), rxBeamSeq)
                    wq(rO.join(runs_folder, "bestBeam.txt"), rG(rxBeamSeq).astype(lz), fmt="%i",delimiter=",")
                    
                    with lH(rO.join(runs_folder, "bestModelPath.txt"), "w") as f:
                        lC(bestModelPathSeq, f)
                    # copy the validation from the corresponding path
                    # shutil.copy(returnDict["position_beam_val_acc_path_png"], runs_folder)
                    # # shutil.copy(returnDict["position_beam_val_acc_path_pdf"], runs_folder)
                    # shutil.copy(returnDict["training_vs_validation_loss_png"], runs_folder)
                    # # shutil.copy(returnDict["training_vs_validation_loss_pdf"], runs_folder)
                    # shutil.copy(returnDict["position_beam_val_channelEffi_png"], runs_folder)
                    # shutil.copy(returnDict["position_beam_val_channelEffi_withRef_png"], runs_folder)
                    # shutil.copy(returnDict["position_beam_val_channelEffi_withRef_total_png"], runs_folder)

                    # Load weights of best trained model
                    # trained_model = copy.deepcopy(model)
                    # trained_model = NN_FCN(x_train.shape[-1], n_beams, nodes_per_layer, layers)
                # test and predict map
                    # beamRSS = getBeamSetRSS(sfs_beamSet, allBeamRSS_narrowTrain, txNum = txNum, rxNum = rxNum,offset=beamSetOffset)
                if (not findLR):
                    if (extraNumCols > 0) and (trainMode > 4):
                        testAcc_sfs, testAcc_txs_sfs,channelEffi_sfs_acc, channelEffi_txs_sfs_acc,testNetWrapperReturnDict = cR(runs_folder, bestReturnDict["x_train"], bestReturnDict["y_train"], bestReturnDict["x_test"],bestReturnDict["y_test"],n_beams, bestReturnDict["modelPath"], bestReturnDict["x_testRowIndex"], xDataTotalDataFrame, yDataTotalDataFrame, posBestBeamDataFrame, realTotalBeamRSSDataFrame,scen_idx,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testAcc_sfs, testAcc_txs_sfs, channelEffi_sfs_acc, channelEffi_txs_sfs_acc, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="deepIA", modelParams=modelParams)
                    else:
                        testAcc_sfs, channelEffi_sfs_acc, testNetWrapperReturnDict = cp(runs_folder, bestReturnDict["x_train"], bestReturnDict["y_train"], bestReturnDict["x_test"],bestReturnDict["y_test"], bestReturnDict["z_test"],realTotalBeamRSS, n_beams, bestReturnDict["modelPath"], bestReturnDict["x_testRowIndex"], scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, testAcc_sfs, channelEffi_sfs_acc,extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams= top_beams, modelType="deepIA", modelParams=lX, bandwidth=bandwidth)                        
        
        # save the dictionary
        # Save to file
        with lH(rO.join(runs_folder, "beamsetToModel.txt"), 'w') as mr:
            # convert tuple key to string 
            # Convert tuple keys using the custom function
            data_with_custom_keys = {cP(key): value for key, value in beamsetToModelDict.items()}

            my_dict_str = lY(data_with_custom_keys)
            mr.write(my_dict_str)
        
        with lH(rO.join(runs_folder, "ModelTobeamset.txt"), 'w') as mr:
            # convert each value to a string
            ModelTobeamsetStringDict = {key: li(value) for key, value in ModelTobeamsetDict.items()}
            my_dict_str = lY(ModelTobeamsetStringDict)
            mr.write(my_dict_str)
        
########################################################                
########################################################                
########################################################                
########################################################                
        ## test 
        widetestData = wideBeamReturnDict["x_test"]
        # construct the 0-1 vector reference
        beamsets_vector = cn(rG(beamsetMatrices), predRxNum)
        modelParams = {}
        modelParams["nodes_per_layer"]= nodes_per_layer 
        modelParams["layers"] = layers
        if trainMode > 4:
            wideBeamTrainedBeamset,_,_,_ = cR(runs_folder, wideBeamReturnDict["x_train"], wideBeamReturnDict["y_train"], wideBeamReturnDict["x_test"],wideBeamReturnDict["y_test"],predRxNum, wideBestModelPath, wideBeamReturnDict["x_testRowIndex"], xDataTotalDataFrame, yDataTotalDataFrame, wideposBestBeamDataFrame, widerealTotalBeamRSSDataFrame,scen_idx,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testacc, testacc_txs, channelEffi_acc, channelEffiacc_txs, extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="deepIA", modelParams=modelParams,plotPredFig = ln, testBatchSize = train_batch_size, checkChannel = ln)
            
        else:
            wideBeamTrainedBeamset,_,_ = cp(runs_folder, wideBeamReturnDict["x_train"], wideBeamReturnDict["y_train"], wideBeamReturnDict["x_test"],wideBeamReturnDict["y_test"], wideBeamReturnDict["z_test"], wide_realTotalBeamRSS,predRxNum, wideBestModelPath, wideBeamReturnDict["x_testRowIndex"], scen_idx,wideposBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins,testacc, channelEffi_acc,extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams=top_beams, modelType="pos", modelParams=modelParams, bandwidth=bandwidth, plotPredFig = ln, testBatchSize = train_batch_size, checkChannel = ln, simpleOutput = lV)      
        
            # map the beamset to 0-1 matrices
            wideBeam_vec = cn(wideBeamTrainedBeamset, predRxNum)
            # # Calculate Hamming distances and find the row in A with the minimum distance for each row in C
            min_distance_indices = wb(rG([wY(wJ(wt(wideBeam_vec), wt(c_row)), axis=1) for c_row in beamsets_vector]), axis=0)
            
            # construct the beamset
            beamsetModels = rG([indicesToModel[index] for index in min_distance_indices])
            # extend it to a new column
            totalDataMatrix_test = totalDataMatrix.iloc[wideBeamReturnDict["x_testRowIndex"],:]
            totalDataMatrix_test["beamset_id"] = min_distance_indices
            
                        
            totalDataMatrix_test["model_path"] = beamsetModels
            
            # also create a column to store the row number
            totalDataMatrix_test["test_row_no"] = totalDataMatrix_test.index
            
            # # construct the needed data matrix
            # allBeamRSS_orig = totalDataMatrix_test.iloc[:, (2 + bestBeamNum):totalData_narrowLabelStart].to_numpy()
            
            # allBeamRSS_narrowTrain = normalize_rss(allBeamRSS_orig, norm_type, params=rss_params, excludeCols=excludeCols)
            # beamSetOffset = 0
            # bestBeamIndices = totalDataMatrix_test.iloc[:,2: (2 + bestBeamNum)].to_numpy()
            # posBestBeamIndices = np.concatenate([mapPos, totalDataMatrix_test.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
            
            # n_samples = min(totalDataMatrix_test.shape[0], max_samples)

            # keepFrontArr = None
            # if trainMode == 6:
            #     beamSetOffset = 2
            #     keepFrontArr= np.arange(beamSetOffset)
            # elif trainMode == 7:
            #     beamSetOffset = 4
            #     keepFrontArr= np.arange(beamSetOffset) 
                
            n_top_stats = lh(top_beams)
            # boundIndexName = (f'B_{i}')
            # rxBeamSeq = [] # incrementally added rx beam index
            # bestModelPathSeq = [] # incrementally add best model saved path
            # rxSet = set(list(range(1,rxIndexMax))) # index 
            # sfs_set_sub = set(column_sub)
            # for M in range(M_start, M_max_new):
            # sub_runs_folder_name = (f'M_{wideBeamTargetNums}')
            runs_folder = rO.join(top_runs_folder_twostage,"SFS")
                # runs_folder = os.path.join(top_runs_folder, boundIndexName,"SFS", sub_runs_folder_n§ame)
            # if loadModel:
            #     loadModelPath_sub = os.path.join(loadModelPath, "SFS", sub_runs_folder_name, "model_checkpoint/nn_beam_pred")
            #     bestBeamIndexPath = os.path.join(loadModelPath, "SFS", sub_runs_folder_name, "bestBeam_json.txt")
            #     bestModelPathtxt = os.path.join(loadModelPath, "SFS", sub_runs_folder_name, "bestModelPath.txt")
                
                

            # Create if doesn't exist
            if not rO.exists(runs_folder):
                rX(runs_folder)
                            
            
            
            # unique indices
            uniqueIndices = ws(min_distance_indices)
            numBeamset = lh(uniqueIndices)
            # group by the beamset id
            
            xDataGp = totalDataMatrix_test.groupby("beamset_id")
            xDataGpElems = [xDataGp.get_group(x) for x in xDataGp.groups]
            txPosNum = lh(xDataGpElems)
            # origDataGp = origDataDataFrame.groupby("tx_id")
            # origDataGpElems = [origDataGp.get_group(x) for x in origDataGp.groups]
            
            # save the testing loss
            channelEffi_msb_acc = rt((n_top_stats,0))
            channelEffi_sfs_acc = rt((n_top_stats,0))
            testAcc_sfs = rt((n_top_stats,0))
            testAcc_msb = rt((n_top_stats,0))
        
            # save the testing loss
            channelEffi_sfs_acc_test = rt((numBeamset, n_top_stats,0))
            channelEffi_msb_acc_test = rt((numBeamset, n_top_stats,0))
            testAcc_sfs_test = rt((numBeamset, n_top_stats,0))
            testAcc_msb_test = rt((numBeamset, n_top_stats,0))
            
            posBestBeamIndices = wV([mapPos, totalDataMatrix.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
            
            posBestBeamPred = wV([mapPos, totalDataMatrix.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
            
            posChannelRatePred = wV([mapPos, totalDataMatrix.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
            posChannelRateRef = wV([mapPos, totalDataMatrix.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
            
            realTotalBeamRSS = totalDataMatrix.iloc[:,(2+totalBeams):].to_numpy()

            for beamset_index, xData in ls(uniqueIndices,xDataGpElems):
                sfs_set_sub = indicesToBeamset[beamset_index]
                beamRSS_sub = cr(sfs_set_sub, allBeamRSS, txNum = txNum, rxNum = rxNum,offset=beamSetOffset, keepFront = keepFrontArr)
                # find the model path
                model_path = indicesToModel[beamset_index]
                if (extraNumCols > 0) and (trainMode > 4):
                    xDataTotalDataFrame = lw(data=beamRSS)
                    # xDataTotalDataFrame.rename(columns={(beamRSS.shape[1]-1):'tx_id'},inplace=True)
                    xDataTotalDataFrame["tx_id"] = tx_id
                # run_folder = runs_folder
                # # beamRSS = normalize_rss(beamRSS, norm_type)
                run_folder = rO.join(runs_folder, f"beamset_{beamset_index}")
                
                # construct the input, output, and data required for plotting
                # xinput = xData
                # get the row indices
                rowIndices_test = xData["row_no"]

                # construct the needed data matrix
                allBeamRSS_test = allBeamRSS[rowIndices_test,:]
                
                # allBeamRSS = normalize_rss(allBeamRSS_orig, norm_type, params=rss_params, excludeCols=excludeCols)
                # beamSetOffset = 0
                # bestBeamIndices = xData.iloc[:,2: (2 + bestBeamNum)].to_numpy()
                # posBestBeamIndices = np.concatenate([mapPos, xData.iloc[:,2:(2+totalBeams)].to_numpy()], -1)
                
                beamRSS_test = cr(sfs_set_sub, allBeamRSS_test, txNum = txNum, rxNum = rxNum,offset=beamSetOffset, keepFront = keepFrontArr)
                
                n_samples = lF(xData.shape[0], max_samples)

                # keepFrontArr = None
                # if trainMode == 6:
                #     beamSetOffset = 2
                #     keepFrontArr= np.arange(beamSetOffset)
                # elif trainMode == 7:
                #     beamSetOffset = 4
                #     keepFrontArr= np.arange(beamSetOffset) 
                # if trainMode >= 6:
                #     # raise Exception("do not support")
                #     wideBeamRSS = np.concatenate([mapPos, wideBeamRSS],-1)
                #     allBeamRSS = np.concatenate([mapPos, allBeamRSS],-1)
                                    
                # bestBeamIndices_test = xData[:,2:(2+bestBeamNum)].to_numpy()
                bestBeamIndices_test = xData.iloc[:,2:(2+bestBeamNum)].to_numpy()
                posBestBeamIndices_test = wV([mapPos[rowIndices_test,:], totalDataMatrix.iloc[rowIndices_test,2:(2+totalBeams)].to_numpy()], -1)
                realTotalBeamRSS_test = totalDataMatrix.iloc[rowIndices_test,(2+totalBeams):].to_numpy()
                
            
                
                testAcc_sfs, channelEffi_sfs_acc, testNetWrapperReturnDict = cp(run_folder, beamRSS_test, bestBeamIndices_test, beamRSS_test, bestBeamIndices_test, realTotalBeamRSS_test,realTotalBeamRSS, n_beams,model_path, rowIndices_test, scen_idx, posBestBeamIndices,knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, testAcc_sfs, channelEffi_sfs_acc,extractTxNum=1, extractRxNum=1, totalBeams = totalBeams, N = N,predTxNum = predTxNum, predRxNum = predRxNum, n_top_stats = n_top_stats, top_beams= top_beams, modelType="deepIA", modelParams=lX, bandwidth=bandwidth)
                
                # update the predict 
                posBestBeamPred[rowIndices_test, 2:(2+totalBeams)] = testNetWrapperReturnDict["pred_beams"][:,:totalBeams]
                
                posChannelRatePred[rowIndices_test, 2:(2+totalBeams)] = testNetWrapperReturnDict["pred_rate"][:,:totalBeams]
                # posChannelRateRef[:,2:(2+totalBeams)] = testNetWrapperReturnDict["pred_rate"][:,:totalBeams]
                
            
                
                # testacc_txs = np.concatenate((testacc_txs, np.expand_dims(np.zeros((testacc_txs.shape[0], testacc_txs.shape[1])),-1)), axis=-1)
                
                # testacc_txs[tx_index, :, -1] = acc * 100
                
                # testAcc_sfs_test[beamset_index,:,:] = testAcc_sfs
                # channelEffi_sfs_acc_test[beamset_index,:,:] = channelEffi_sfs_acc
                

            # prediction map
            predDict = {"pred_beams": posBestBeamPred, "pred_rate":posChannelRatePred}
            pred_beams,pred_returnDict = rx(N, "NN", n_beams, scen_idx, runs_folder, beamRSS_test, rowIndices_test, beamRSS_test, posBestBeamIndices,realTotalBeamRSS, knn_num_neighbors, lt_predition_per_bin, bin_size,n_bins, lX, txNum = predTxNum, rxNum = predRxNum, predBeamNum=totalBeams, bandwidth=bandwidth,plotFig=lV,testBatchSize=train_batch_size,givenPred = lV, predDict = predDict)
                    
        
        
        
        # # save the testAcc_sfs
        if not findLR:
            ce(algos, testAcc_sfs, testAcc_msb,channelEffi_sfs_acc,channelEffi_msb_acc, top_runs_folder, testNetWrapperReturnDict, top_beams, extraNumCols, trainMode)
            
        #     # plt.figure()
        #     if len(algos) == 2:
        #         plotAggResult(np.vstack((testAcc_sfs[[0,-1],:], testAcc_msb[[0,-1],:])),top_runs_folder,"accM.png","Accuracy [%]", title="Accuracy", xlabel="M", selectRows = [0,1,2,3], labels=["sfs top1", "sfs top5", "msb top1", "msb top5"])
                
        #         # plot channel capacity
        #         plotAggResult(np.vstack((channelEffi_sfs_acc[[0,-1],:], channelEffi_msb_acc[[0,-1],:])),top_runs_folder,"channelM.png","Achievable Rate [%]", title="Achievable Rate", xlabel="M", selectRows = [0,1,2,3], labels=["sfs top1", "sfs top5","msb top1", "msb top5"])
                
        #         # plot channel capacity
        #         refAcc = testNetWrapperReturnDict["channelEffi_ref"] * 100
        #         refAcc = np.tile(np.expand_dims(refAcc,-1), (1, channelEffi_sfs_acc.shape[-1]))


        #         plotAggResult(np.vstack((channelEffi_sfs_acc[[0,-1],:], channelEffi_msb_acc[[0,-1],:], refAcc[[0,-1],:])),top_runs_folder,"channelM_withRef.png","Achievable Rate [%]", title="Achievable Rate", xlabel="M", selectRows = [0,1,2,3,4,5], labels=["sfs top1", "sfs top5","msb top1", "msb top5", "ref top1", "ref top5"], top_beams=[1,2,3,4,5,6])                
        #         # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[0,:],"b*-")
        #         # plt.plot(np.arange(1, testAcc_msb.shape[-1] + 1), testAcc_msb[0,:],"m*-")
                
        #         # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[4,:],"g*-")
        #         # plt.plot(np.arange(1, testAcc_msb.shape[-1] + 1), testAcc_msb[4,:],"c*-")
        #         # plt.xlabel("M")
        #         # plt.ylabel("Accuracy [%]")
        #         # plt.title("Accuracy")
        #         # plt.legend(["sfs top1","msb top1", "sfs top5", "msb top5"])
        #         # plt.savefig(os.path.join(top_runs_folder, "accM.png"))
        #         # plt.show()
        #     print("sfs accuracy:")
        #     print(testAcc_sfs)
        #     print("msb accuracy:")
        #     print(testAcc_msb)
        #     np.savetxt(os.path.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
            
        #     np.savetxt(os.path.join(top_runs_folder, "msb_accuracy.txt"), testAcc_msb, fmt='%.2f')
        #     print("acc saved at: " + os.path.join(top_runs_folder, "sfs_accuracy.txt"))
            
        #     np.savetxt(os.path.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
        #     # separte top-k
        #     if "SFS" in algos:
        #         plotAggResult(testAcc_sfs, top_runs_folder, "accM_sfs.png", "Accuracy [%]",top_beams,"Accuracy", "M")
                
        #         plotAggResult(channelEffi_sfs_acc, top_runs_folder, "accM_channelEffi_sfs_select.png", "Achievable Rate", top_beams, "", "M", plotRef = True, refacc=np.transpose(testNetWrapperReturnDict["channelEffi_ref"] * 100), selectRows=[0,2,4])
                
        #         plotAggResult(channelEffi_sfs_acc, top_runs_folder, "accM_channelEffi_sfs.png", "Achievable Rate", top_beams, "", "M", plotRef = True, refacc=np.transpose(testNetWrapperReturnDict["channelEffi_ref"] * 100))
                
        #         # plt.figure()
                    
        #         # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[0,:]*100, 'g*-', lw=2.0, label='Top-1 Accuracy')
        #         # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[1,:]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
        #         # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[2,:]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
        #         # plt.plot(np.arange(1, testAcc_sfs.shape[-1]+1), testAcc_sfs[3,:]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
        #         # plt.xlabel("M")
        #         # plt.ylabel("Accuracy [%]")
        #         # plt.title("Accuracy")
        #         # plt.legend()
        #         # plt.savefig(os.path.join(top_runs_folder, "accM_sfs.png"))
        #         # plt.show()
                
        #     # msb
        #     if "MSB" in algos:
        #         plotAggResult(testAcc_msb, top_runs_folder, "accM_msb.png", "Accuracy [%]",top_beams,"Accuracy", "M")
        #         plotAggResult(channelEffi_msb_acc, top_runs_folder, "accM_channelEffi_msb_select.png", "Achievable Rate", top_beams, "", "M",plotRef = True, refacc=np.transpose(testNetWrapperReturnDict["channelEffi_ref"] * 100), selectRows=[0,2,4])
        #         plotAggResult(channelEffi_msb_acc, top_runs_folder, "accM_channelEffi_msb.png", "Achievable Rate", top_beams, "", "M",plotRef = True, refacc=np.transpose(testNetWrapperReturnDict["channelEffi_ref"] * 100))
                
        #         # plt.figure()
                    
        #         # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[0,:]*100, 'g*-', lw=2.0, label='Top-1 Accuracy')
        #         # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[1,:]*100, 'b*-', lw=2.0, label='Top-2 Accuracy')
        #         # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[2,:]*100, 'r*-', lw=2.0, label='Top-3 Accuracy')
        #         # plt.plot(np.arange(1, testAcc_msb.shape[-1]+1), testAcc_msb[3,:]*100, 'm*-', lw=2.0, label='Top-5 Accuracy')
        #         # plt.xlabel("M")
        #         # plt.ylabel("Accuracy [%]")
        #         # plt.title("Accuracy")
        #         # plt.legend()
        #         # plt.savefig(os.path.join(top_runs_folder, "accM_msb.png"))
        #         # plt.show()
                
            
        #     print("sfs accuracy:")
        #     print(testAcc_sfs)
        #     print("msb accuracy:")
        #     print(testAcc_msb)
        #     np.savetxt(os.path.join(top_runs_folder, "sfs_accuracy.txt"), testAcc_sfs, fmt='%.2f')
            
        #     np.savetxt(os.path.join(top_runs_folder, "msb_accuracy.txt"), testAcc_msb, fmt='%.2f')
        #     print("acc saved at: " + os.path.join(top_runs_folder, "sfs_accuracy.txt"))
            



        #     print("sfs channelEffi:")
        #     print(testAcc_sfs)
        #     print("msb channelEffi:")
        #     print(testAcc_msb)
        #     np.savetxt(os.path.join(top_runs_folder, "sfs_channelEffi.txt"), channelEffi_sfs_acc, fmt='%.2f')
            
        #     np.savetxt(os.path.join(top_runs_folder, "msb_channelEffi.txt"), channelEffi_msb_acc, fmt='%.2f')
        #     np.savetxt(os.path.join(top_runs_folder, "channelEffi_ref.txt"), testNetWrapperReturnDict["channelEffi_ref"] * 100, fmt='%.2f')
        #     print("channel effi saved at: " + os.path.join(top_runs_folder, "sfs_channelEffi.txt"))
            
        #     # np.savetxt(os.path.join(top_runs_folder, "sfs_channelEffi_txs.txt"), channelEffi_txs_sfs_acc, fmt='%.2f')
            
        #     # np.savetxt(os.path.join(top_runs_folder, "msb_channelEffi_txs.txt"), channelEffi_txs_msb_acc, fmt='%.2f')
            
        #     # savetxt cannot save 3d, so save binary
        #     if (extraNumCols > 0) and (trainMode > 4):
        #         np.save(os.path.join(top_runs_folder, "sfs_channelEffi_txs.npy"), channelEffi_txs_sfs_acc)
                
                
        #         np.save(os.path.join(top_runs_folder, "msb_channelEffi_txs.npy"), channelEffi_txs_msb_acc)
                
        #         np.save(os.path.join(top_runs_folder, "sfs_accuracy_txs.npy"), testAcc_txs_sfs)
                
        #         np.save(os.path.join(top_runs_folder, "msb_accuracy_txs.npy"), testAcc_txs_msb)      

## print result                