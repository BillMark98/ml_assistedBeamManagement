
# used to generate the data as in deepIA paper
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
freq = 24e9
signal_lambda = 3e8 / freq
d0 = 1 # 1 meter
PL_d0 = 20 *np.log10(4 * np.pi * d0/signal_lambda)

arrayN = 10
totalBeamNum = 24
beamAngle = 360 / totalBeamNum
halfBeamAngleBelow = np.ceil(beamAngle/2)  # if (-halfBeamAngle, halfbeamAngle]  -> 0   beamIndex starts from 0
halfBeamAngleUp = np.floor(beamAngle/2)
simulationBound = 25
topkBeams = 5

# the angle at which antenna will point at  - tx codebook
antennaDirectionAngles = np.linspace(0, 2 * np.pi, totalBeamNum + 1)[:-1]

def radToDeg(radians):
    """
    convert radians to degree
    """
    degrees = radians / np.pi * 180
    return degrees

def degToRad(degrees):

    radians = degrees/180 * np.pi
    return radians

def pathLoss(ds, mode = "LOS"):
    """
    given the distance return the path loss in dB  

    Note positive means a pos
    """
    N = len(ds)
    if mode == "LOS":
        # minimum mean squared error path loss exponent
        n_ple = 1.9
        # x_sigma = 1.1
        x_sigma = 0
    else:
        # NLOS mode
        n_ple = 4.5
        x_sigma = 10
    
    # generate X_sigma
    X_sigma = np.random.normal(scale=x_sigma, size = N)
    PL_losses = PL_d0 + 10 * n_ple * np.log10(ds/d0) + X_sigma

    return PL_losses

def getDistance(xPos, yPos):
    """
    return the distance
    """
    return np.sqrt(np.square(xPos) + np.square(yPos))

def getPathLoss(xPos, yPos, mode="LOS"):
    """
    given x,y position, return the path loss
    """
    ds = getDistance(xPos,yPos)
    pls  = pathLoss(ds, mode)
    return pls

def fitAngle(thetas, angleLow=-halfBeamAngleBelow, angleUp=360 - halfBeamAngleBelow):
    """
    given thetas, fit the thetas within the range [angleLow, angleUp]

    thetas in radian, angleLow, angleUp in degree
    """

    thetas_deg = radToDeg(thetas)
    thetas_add = thetas_deg < angleLow
    while(thetas_add.any()):
        thetas_deg[thetas_add] += 360
        thetas_add = thetas_deg < angleLow
    
    thetas_minus = thetas_deg > angleUp
    while(thetas_minus.any()):
        thetas_deg[thetas_minus] -= 360
        thetas_minus = thetas_deg > angleUp
    
    thetas = degToRad(thetas_deg)
    return thetas
        
def sineSinc(thetas, N = arrayN):
    """
    given the thetas calculate  sin(N/2 * thetas)/sin(thetas/2)
    """

    smallTheta = np.abs(np.sin(thetas/2)) < 1.01 * np.finfo(float).eps
    # smallTheta = thetas < 2 * np.finfo(float).eps

    afs = np.abs(np.sin(N/2 * thetas)/(np.sin(thetas/2) + np.finfo(float).eps)) * (1 - smallTheta) + 20 * np.ones(thetas.shape) * smallTheta
    return afs

def antennaGain(thetas, N = arrayN, d = signal_lambda * 1/2, plotGain = False):
    """
    given the array of thetas (angle between tx np.logical_and rx), return the antenna gain
    """
    alphas = np.pi/2 - thetas
    psis = 2 * np.pi/signal_lambda * d * np.cos(alphas)
    # array factor

    afs = sineSinc(psis, N)
    # afs = np.abs(np.sin(N/2 * psis)/(np.sin(psis/2) + np.finfo(float).eps))
    # calculate the log
    afs = 20 * np.log10(afs)
    # for the backlobe 90 - 270, minus 10 db
    thetas_deg = radToDeg(thetas)
    thetas_deg_backlobe = np.logical_or(np.logical_and((thetas_deg <= 270), (thetas_deg >= 90)), np.logical_and((thetas_deg <= -90), (thetas_deg >= -270)))
    afs = afs - 100 * thetas_deg_backlobe

    if plotGain:

        fig = plt.figure(layout='constrained')
        ax1 = fig.add_subplot(1,2,1,projection='polar')
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(thetas/np.pi * 180,afs)
        # ax1.plot(np.linspace(0, 2 * np.pi, 360), data)
        ax1.plot(thetas, afs)
        plt.savefig("./antenna_gain.png")
        plt.show()
    return afs

def getRSS(pls, thetas, N = arrayN, d = signal_lambda * 1/2, plotGain = False):
    """
    given the pls, thetas, calculate the all rss w.r.t to all tx positions, so inparticular each row, consist of txnums
    """
    totalDataNum = len(thetas)
    thetaCopied = np.tile(np.expand_dims(thetas,-1), (1,totalBeamNum))
    antennaAngles = np.tile(antennaDirectionAngles, (totalDataNum,1))
    thetas = thetaCopied - antennaAngles
    # get gains
    antennaGains = antennaGain(thetas)
    rss = np.subtract(antennaGains, np.expand_dims(pls,-1))
    return rss


def getSymmetricIndices(indices, num = topkBeams-1):
    """
    given an array of indices, return the symmetric indices, for example

    3, num=4, then the array is [3,1,2,0,23]   note beamIndex start from 0, and total 24 beams
    """
    currentIndex = indices
    numOfLoop = 1
    offset = 0
    signum = -1
    while num > 0:
        offset = signum * numOfLoop
        addedCol = (currentIndex + offset) % totalBeamNum
        currentIndex = addedCol
        if (numOfLoop > 1):
            addedCol = np.expand_dims(addedCol,-1)
            indices = np.concatenate((indices, addedCol), axis=-1)
        else:
            indices = np.stack((indices, addedCol), axis=-1)

        numOfLoop += 1
        signum *= -1
        num -= 1
    return indices

def getTopkIndices(thetas, topk = topkBeams):
    """
    given array thetas return 
    len(thetas) * topk
    indices, each row:
    [top1, top2, top3, top...., topk]
    """

    # fit angle
    thetas = fitAngle(thetas)
    theta_degs = radToDeg(thetas)
    # topBeams = np.ceil((theta_degs + halfBeamAngleBelow)/beamAngle)
    # use floor beam index from 0 to 23
    topBeams = np.floor((theta_degs + halfBeamAngleBelow)/beamAngle)
    # get symmetric Indices

    topBeams = getSymmetricIndices(topBeams, num = topk - 1)
    return topBeams


def generateDataMatrix(totalNumPts, Bound = simulationBound, topk = topkBeams, totalBeam = totalBeamNum,mode="LOS", fileName = "./data/generateMatrix.csv", plotData = False):
    """
    given totalNumPts, generate datamatrix

    xpos, ypos, topk-beam indices, totalBeam RSS
    """
    n_per_dim = int(np.ceil(np.sqrt(totalNumPts)))
    # bounds
    # generate random int
    # xPos = np.random.randint(-Bound, Bound, totalNumPts)
    # yPos = np.random.randint(-Bound, Bound, totalNumPts)
    xPos = np.random.rand(totalNumPts) * 2 * Bound - Bound
    yPos = np.random.rand(totalNumPts) * 2 * Bound - Bound
    # calculate the radius
    rPos = np.sqrt(np.square(xPos) + np.square(yPos))
    withinCircle = rPos < 1
    randomScale = np.random.rand(totalNumPts) * (Bound/2 - 1) + 1 # to make sure the scalar is within range (1, Bound/2)
    xPos = xPos * 1/rPos * randomScale * withinCircle + xPos * (1 - withinCircle)
    yPos = yPos * 1/rPos * randomScale * withinCircle + yPos * (1 - withinCircle)
    if (plotData):
        plt.figure()
        plt.scatter(xPos, yPos)
        # plot also unit circle
        circleAngles = np.linspace(0, 2*np.pi,1000)
        circleX = np.cos(circleAngles)
        circleY = np.sin(circleAngles)
        plt.plot(circleX, circleY,"r")
        plt.show()
    # xPos = xPos * 1/rPos * withinCircle * withinCircle * 
    # xPos[withinCircle] *= 1/rPos[withinCircle] * randomScale 
    # # xPos = xPos * 1/rPos * withinCircle * withinCircle * 
    # yPos[withinCircle] *= 1/rPos[withinCircle] * randomScale

    # calculate the angle to center
    thetas = np.arctan2(xPos, yPos)

    # generate the best topk beam indices

    topBeams = getTopkIndices(thetas, topk)
    # get the rss
    pls = getPathLoss(xPos,yPos,mode)
    # # get antenna gains
    # antennaGains = antennaGain(thetas)
    rss = getRSS(pls, thetas)
    # check the rss based best beam index
    rss_max_index = np.argmax(rss, axis=1)
    topBeamIndex = topBeams[:,0]
    topBeamIndex = topBeamIndex.astype(int)
    # compare the two index
    indexEqual = np.equal(topBeamIndex, rss_max_index)
    print("index correct: " + str(indexEqual.sum()/len(indexEqual)))

    # calculate the index diff > 2
    indexDiff = np.abs(np.subtract(topBeamIndex, rss_max_index))
    print(np.unique(indexDiff))
    indexDiffLarge = indexDiff > 1
    print("index diff >= 2: " + str(indexDiffLarge.sum()))
    plt.plot(indexDiff)
    # plt.show()
    # combine the matrix
    dataMatrix = np.stack((xPos,yPos), axis=-1)
    dataMatrix = np.concatenate((dataMatrix, topBeams), axis=-1)
    dataMatrix = np.concatenate((dataMatrix, rss), axis=-1)
    np.savetxt(fileName,dataMatrix,delimiter=",")

    
if __name__ == "__main__":
    thetas = np.linspace(-np.pi,np.pi-np.finfo(float).eps,360)
    # thetas = np.linspace(0,2*np.pi,360)
    antennaGain(thetas, plotGain=True)

    # thetas = np.linspace(-np.pi,np.pi-np.finfo(float).eps,360)
    # theta_conv = fitAngle(thetas, -8,352)
    # plt.plot(radToDeg(thetas),radToDeg(theta_conv))
    # plt.show()

    # indices = np.random.randint(0,23,10)
    # symmIndices = getSymmetricIndices(indices)
    # print("symmIndices")
    # print(symmIndices)
    # generateDataMatrix(1000000, plotData=False)
    




# def ArrayFactor(ElementArray, Freq):
#     """
#     Summation of field contributions from each element in array, at frequency freq at theta 0°-95°, phi 0°-360°.
#     Element = xPos, yPos, zPos, ElementAmplitude, ElementPhaseWeight
#     Returns arrayFactor[theta, phi, elementSum]
#     """

#     arrayFactor = np.ones((360, 95))

#     Lambda = 3e8 / Freq

#     for theta in range(95):
#         for phi in range(360):                                                                                                      # For all theta/phi positions
#             elementSum = 1e-9 + 0j

#             for element in ElementArray:                                                                                            # Summation of each elements contribution at theta/phi position.
#                 relativePhase = CalculateRelativePhase(element, Lambda, math.radians(theta), math.radians(phi))                     # Find relative phase for current element
#                 elementSum += element[3] * math.e ** ((relativePhase + element[4]) * 1j)                                            # Element contribution = Amp * e^j(Phase + Phase Weight)

#             arrayFactor[phi][theta] = elementSum.real

#     return arrayFactor

  
# def CalculateRelativePhase(Element, Lambda, theta, phi):
#     """
#     Incident wave treated as plane wave. Phase at element is referred to phase of plane wave at origin.
#     Element = xPos, yPos, zPos, ElementAmplitude, ElementPhaseWeight
#     theta & phi in radians
#     See Eqn 3.1 @ https://theses.lib.vt.edu/theses/available/etd-04262000-15330030/unrestricted/ch3.pdf
#     """
#     phaseConstant = (2 * math.pi / Lambda)

#     xVector = Element[0] * math.sin(theta) * math.cos(phi)
#     yVector = Element[1] * math.sin(theta) * math.sin(phi)
#     zVector = Element[2] * math.cos(theta)

#     phaseOfIncidentWaveAtElement = phaseConstant * (xVector + yVector + zVector)

#     return phaseOfIncidentWaveAtElement
