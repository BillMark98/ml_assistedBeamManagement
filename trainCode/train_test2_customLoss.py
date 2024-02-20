

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
