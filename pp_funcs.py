# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:14:06 2013

@author: pstraus
"""

import numpy
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import re
import matplotlib.pyplot as plt


def getMax(pumpProbe):
    
    maxvec = numpy.zeros(len(pumpProbe.time))
    for index in range(len(pumpProbe.time)):
        maxvec[index] = max(pumpProbe.ppData[index])
    
    return (pumpProbe.time, maxvec)

def gauss(mu, sigma, data):
    gauss_data = 1/(numpy.sqrt(numpy.pi * 2) * sigma) * numpy.e ** (-((data - mu) ** 2)/(2 * sigma **2))
    return gauss_data

def lorentz(mu, gamma, data):
    lorentz_data = gamma/numpy.pi/((data - mu) **2 + gamma ** 2)
    return lorentz_data

def voigt(mu, sigma, gamma, data):
    gauss_data = gauss(mu, sigma, data)
    lorentz_data = lorentz(mu, gamma, data)
    
    gauss_fft = numpy.fft.fft(gauss_data)
    lorentz_fft = numpy.fft.fft(lorentz_data)
    
    voigt_fft = gauss_fft * lorentz_fft
    voigt_data = numpy.real(numpy.fft.fftshift(numpy.fft.ifft(voigt_fft)))
    
    #voigt_data =numpy.fft.ifft(numpy.fft.fftshift(shift(numpy.fft.fft(gauss_data)) * numpy.fft.fftshift(numpy.fft.fft(lorentz_data));
    return voigt_data
    
def build_pp_Axis(centerWavelength, dlambda, offset, *arg):
    centerWavelength = centerWavelength + offset #this is a feature for lazy people.  If you corrected it already, please just use '0' for the offset
    
    dlambda = float(dlambda)
#get specified number of detector elements.  If none is specified, use a default
    try:
        len(arg)
        if arg[0] == "det_elts":
            try:
                numelts = arg[1]
            except:
                numelts = 64
                print "You didn't specify the number of elements.  Defaulting to 64..."
    except:
        numelts = 64
      
    axis = numpy.zeros(numelts)    
    
    for element in range(0, numelts):
        axis[element] = dlambda * (element - float(numelts)/2.0) + centerWavelength  
       
    axis = axis* 1e-7               #convert to cm
    axis = axis ** -1          #convert to cm-1
    axis = numpy.flipud(axis)
    return axis
    
def ppSpline(ppData):
    time = ppData.time
    RawProbeAxis = ppData.probeAxis
    ppData = ppData.ppData
    
    probeAxis = numpy.linspace(RawProbeAxis[0], RawProbeAxis[-1], 400)
    SplineData = numpy.zeros((len(ppData[:,1]),len(probeAxis)))    
    print probeAxis
    
    #The spline function in scipy only works for ascending values of x.  Let's fix that:
    
    if (RawProbeAxis[0] > RawProbeAxis[-1]):
        RawProbeAxis = numpy.flipud(RawProbeAxis)
        probeAxis = numpy.flipud(probeAxis)        
        ppData = numpy.fliplr(ppData)
        flip = True
    else:
        flip = False
    for datasetIndex in range(0, len(ppData[:,1])):
        
        ydata = ppData[datasetIndex,:]                
        tck = interpolate.splrep(RawProbeAxis, ydata)
        #print tck
        SplineData[datasetIndex,:] = interpolate.splev(probeAxis, tck)
    
    if flip == True:
        probeAxis = numpy.flipud(probeAxis)
        
          
    ppSpline = PumpProbe(SplineData, probeAxis, time)  
    return ppSpline

def pseudoVoigt(data, mu, gamma, sigma, frac):
    lorentzData = frac * lorentz(mu, gamma, data)
    gaussian = (1 - frac) * gauss(mu, sigma, data)
    pseudoVoigt = lorentzData + gaussian
    pseudoVoigt = pseudoVoigt/max(pseudoVoigt)
    return pseudoVoigt
        
def TwoGaussians(data, mu1, sigma1, A1, mu2, sigma2, A2):
    gaussian_1 = A1 * gauss(mu1, sigma1, data)
    gaussian_2 = A2 * gauss(mu2, sigma2, data)
    doubleGauss = gaussian_1 + gaussian_2
    return doubleGauss 
    
def TwoVoigts(data, mu1, sigma1, gamma1, A1, mu2, sigma2, gamma2, A2):
    voigt1 = A1 * voigt(mu1, sigma1, gamma1, data)
    voigt2 = A2 * voigt(mu2, sigma2, gamma2, data)
    TwoVoigt = voigt1 + voigt2
    return TwoVoigt
    
def twoPseudoVoigts(data, mu1, gamma1, sigma1, frac1, A1, mu2, gamma2, sigma2, frac2, A2):
    pseudovoigt1 = A1 * pseudoVoigt(data, mu1, gamma1, sigma1, frac1)
    pseudovoigt2 = A2 * pseudoVoigt(data, mu2, gamma2, sigma2, frac2)
    
    return (pseudovoigt1 + pseudovoigt2)    
    
 
def twoLorentzians(data, mu1, gamma1, A1, mu2, gamma2, A2):
    lorentzian1 = A1 * lorentz(mu1, gamma1, data)
    lorentzian2 = A2 * lorentz(mu2, gamma2, data)
    
    return (lorentzian1 + lorentzian2)
    
def LorentzianGuess(pumpProbe, setNumber):
    axis = pumpProbe.probeAxis
    data = pumpProbe.ppData[setNumber,:]
    
    
        #These are the obvious guesses
    mu1 = axis[data == max(data)]
    mu1 = mu1[0]
    mu2 = axis[data == min(data)]
    mu2 = mu2[0]
    A1 = max(data)
    A2 = min(data)
    
    FWHMdata1 = axis[data > max(data/2)]
    FWHMdata2 = axis[data < min(data/2)]

    FWHM1 = max(FWHMdata1) - min(FWHMdata1)        
    FWHM2 = max(FWHMdata2) - min(FWHMdata2)
    gamma1 = FWHM1/2
    gamma2 = FWHM2/2
       
    crudeGuess = {"mu1": mu1, "mu2": mu2, "A1":A1, "A2": A2, "gamma1" : gamma1, "gamma2" : gamma2,}

    return crudeGuess
    
def fitTwoLorentzians(pumpProbe, setNumber, **guesses):
    CrudeGuess = LorentzianGuess(pumpProbe, setNumber)    
    data = pumpProbe.ppData[setNumber,:]
    axis = pumpProbe.probeAxis
    plot = True
    
    for keys, values in guesses.iteritems():
        try:
            CrudeGuess[keys] = guesses[keys]
        except:
            "Guessing %s" % keys
            
    indexDic = {"mu1": 0, "gamma1": 1, "A1" : 2, "mu2" : 3, "gamma2" : 4, "A2" : 5 }
    guess = numpy.zeros(6)    
    
    for k, v in indexDic.iteritems():
        guess[indexDic[k]] = CrudeGuess[k]
    
    fitVars, covMatrix = optimize.curve_fit(twoLorentzians, axis, data, guess)    
    if plot == True:
        fitData = twoLorentzians(axis, *fitVars)        
        plt.plot(axis, fitData, axis, data)    
        
    return fitVars
    
def twoPseudoVoigtGuess(pumpProbe, setNumber):
    axis = pumpProbe.probeAxis
    data = pumpProbe.ppData[setNumber,:]
    
    
        #These are the obvious guesses
    mu1 = axis[data == max(data)]
    mu1 = mu1[0]
    mu2 = axis[data == min(data)]
    mu2 = mu2[0]
    A1 = max(data)
    A2 = min(data)
    
    FWHMdata1 = axis[data > max(data/2)]
    FWHMdata2 = axis[data < min(data/2)]

    FWHM1 = max(FWHMdata1) - min(FWHMdata1)        
    FWHM2 = max(FWHMdata2) - min(FWHMdata2)
   
    gamma1 = FWHM1
    sigma1 = FWHM1/(2 * 2.35482) * 3 

    gamma2 = FWHM2
    sigma2 = FWHM2/(2 * 2.35482) * 3   
    
    frac1 = gamma1/sigma1
    frac2 = gamma2/sigma2
    
    
    crudeGuess = {"mu1": mu1, "mu2": mu2, "A1":A1, "A2": A2, "sigma1": sigma1, "sigma2" : sigma2, "gamma1" : gamma1, "gamma2" : gamma2, "frac1" : frac1, "frac2" : frac2}
    return crudeGuess
    

def fitTwoPseudoVoigts(pumpProbe, setnumber, **guesses): 
    crudeGuess = twoPseudoVoigtGuess(pumpProbe, setnumber)
    plot = True
    
    for key, value in crudeGuess.iteritems():
        try:
            crudeGuess[key] = guesses[key]
        except:
            print "%s Not specified.  Guessing %s = %d" % (key, key, crudeGuess[key] )
    try:
        plot = guesses["plot"]
    except:
        pass
    
    
    
    '''The way the twoPseudoVoigt function interacts with curve_fit, we require that the guesses are input as a list/array
        this dictionary relates the required indices so we can make the guesses correctly.'''
        
    pseudoVoigtKeyIndex =  {"mu1" : 0, "gamma1": 1, "sigma1" : 2, "frac1" : 3, "A1" : 4, "mu2" : 5, "gamma2": 6, "sigma2" : 7, "frac2" : 8, "A2" : 9 }    
    guess = numpy.zeros(10)    

    
    for key, value in pseudoVoigtKeyIndex.iteritems():
        guess[pseudoVoigtKeyIndex[key]] = crudeGuess[key]
        
    '''Actually do the fit'''   
    fitVars, covarianceMatrix = optimize.curve_fit(twoPseudoVoigts, pumpProbe.probeAxis, pumpProbe.ppData[setnumber,:],guess)
    
    #guess = [1, 1, 1, 0.5, 1, 1, 1, 1, .5, 1]
    
    if plot == True:
        fitData = twoPseudoVoigts(pumpProbe.probeAxis, *fitVars)
        plt.plot(pumpProbe.probeAxis, fitData, pumpProbe.probeAxis, pumpProbe.ppData[setnumber,:])
        plt.show()    
    return fitVars
    
def fitTwoVoigts(pumpProbe, setnumber, **guesses):
    d = crudeGuess_twoVoigts(pumpProbe, setnumber)
    plot = True
    
    for k, v in d.iteritems():
        try:
            d[k] = guesses[k]
        except:
            print "\t%s starting position not specified.  Making a guess..." % k
    try:
        plot = guesses["plot"]
    except:
        pass
    
    mu1, sigma1, gamma1, A1, mu2, sigma2, gamma2, A2 = d['mu1'], d['sigma1'],d['gamma1'],d['A1'],d['mu2'],d['sigma2'],d['gamma2'],d['A2']
    guess = [mu1, sigma1, gamma1, A1, mu2, sigma2, gamma2, A2]    
    print d  
    fitVars, covMat = optimize.curve_fit(TwoVoigts, pumpProbe.probeAxis, pumpProbe.ppData[setnumber,:], guess)

    print fitVars
    if plot == True:
        fitdata = TwoVoigts(pumpProbe.probeAxis, *fitVars)
        plt.plot(pumpProbe.probeAxis, fitdata, 'r', pumpProbe.probeAxis, pumpProbe.ppData[setnumber,:],'k')
    return fitVars
    
def crudeGuess_twoVoigts(pumpProbe, setnumber):
    axis = pumpProbe.probeAxis
    data = pumpProbe.ppData[setnumber,:]
    
        #These are the obvious guesses
    mu1 = axis[data == max(data)]
    mu1 = mu1[0]
    mu2 = axis[data == min(data)]
    mu2 = mu2[0]
    A1 = max(data)
    A2 = min(data)
    
    FWHMdata1 = axis[data > max(data/2)]
    FWHMdata2 = axis[data < min(data/2)]

    FWHM1 = max(FWHMdata1) - min(FWHMdata1)        
    FWHM2 = max(FWHMdata2) - min(FWHMdata2)
    
    gamma1 = FWHM1/3
    sigma1 = FWHM1/(2 * 2.35482) * 3 

    gamma2 = FWHM2/3
    sigma2 = FWHM2/(2 * 2.35482) * 3   
    
    crudeGuess = {"mu1": mu1, "mu2": mu2, "A1":A1, "A2": A2, "sigma1": sigma1, "sigma2" : sigma2, "gamma1" : gamma1, "gamma2" : gamma2}
    return crudeGuess
    
def crudeGuess_twoGaussians(pumpProbe, setnumber):
    axis = pumpProbe.probeAxis
    data = pumpProbe.ppData[setnumber,:]
    
    
    #These are the obvious guesses
    mu1 = axis[data == max(data)]
    mu1 = mu1[0]
    mu2 = axis[data == min(data)]
    mu2 = mu2[0]
    A1 = max(data)
    A2 = min(data)
    
    #now let's estimate sigma by the FWHM:
    FWHMdata1 = axis[data > max(data/2)]
    FWHMdata2 = axis[data < min(data/2)]

    FWHM1 = max(FWHMdata1) - min(FWHMdata1)        
    FWHM2 = max(FWHMdata2) - min(FWHMdata2)
    
    sigma1 = FWHM1/2.35482
    sigma2 = FWHM2/2.35482
    
    crudeGuess = {"mu1": mu1, "mu2": mu2, "A1":A1, "A2": A2, "sigma1": sigma1, "sigma2" : sigma2}
    return crudeGuess

def fitAllpVoigt(pumpProbe, start, end):
    
   fitData = numpy.zeros((end - start, len(pumpProbe.probeAxis)))
   aHarm = numpy.zeros(end - start)
   fitVars = numpy.zeros((end - start, 10)) 
   plt.figure()
   for index in range(start, end):
        fitVars = fitTwoPseudoVoigts(pumpProbe,index)
        fitData[index,:] = twoPseudoVoigts(pumpProbe.probeAxis, *fitVars)   
        aHarm[index] = pumpProbe.probeAxis[numpy.argmax(fitData[index,:])] - pumpProbe.probeAxis[numpy.argmin(fitData[index,:])]

   return (fitData, aHarm)


def fitAllGaussian(pumpProbe, start, end):
    
   fitData = numpy.zeros((end - start, len(pumpProbe.probeAxis)))
   aHarm = numpy.zeros(end - start)
   fitVars = numpy.zeros((end - start, 6)) 
   #plt.figure()
   for index in range(start, end):
        fitVars = fitTwoGaussians(pumpProbe,index)
        fitData[index,:] = TwoGaussians(pumpProbe.probeAxis, *fitVars)   
        aHarm[index] = pumpProbe.probeAxis[numpy.argmax(fitData[index,:])] - pumpProbe.probeAxis[numpy.argmin(fitData[index,:])]
   plt.contour(pumpProbe.probeAxis, pumpProbe.time, fitData)
   plt.show()
   #plt.figure()
   #plt.plot(pumpProbe.time, aHarm)
   #plt.show()
   return (fitData, aHarm)    
    
    
def fitTwoGaussians(pumpProbe, setnumber, **guesses):
    '''You can enter the guesses in a dictionary.  "plot" determines if the function plots the result'''
    d = crudeGuess_twoGaussians(pumpProbe, setnumber)
    plot = True  
    
    for k, v in d.iteritems():
        try:
            d[k] = guesses[k]
        except:
            print "%s Not specified.  Making a crude estimate..." % k
            
    try:
        plot = guesses["plot"]
    except:
        pass
   
    mu1, mu2, A1, A2, sigma1, sigma2 = d["mu1"], d["mu2"], d["A1"], d["A2"], d["sigma1"], d["sigma2"]
    print "\n\t Initial guesses are:\n"
    print "\t %s%d %s%d %s%d %s%d %s%d %s%d" % ("mu1: ", mu1, "mu2: ", mu2, "A1: ", A1, "A2: ",  A2, "sigma1: ", sigma1, "sigma2: " , sigma2)

    guess = [mu1, sigma1, A1, mu2, sigma2, A2]
    
    fitVars, covMat = optimize.curve_fit(TwoGaussians, pumpProbe.probeAxis, pumpProbe.ppData[setnumber,:], guess)
    if plot == True:    
        fitData = TwoGaussians(pumpProbe.probeAxis, *fitVars)
        plt.plot(pumpProbe.probeAxis, fitData,'b', pumpProbe.probeAxis, pumpProbe.ppData[setnumber,:],'r')
        
    
    return fitVars
    
    
    
class PumpProbe:
    #standard init.  supply the probe axis and the data
    """PumpProbe is a simple class that pulls together several pieces of data about a pump probe spectrum.
    The key variables are:
    
            -probeAxis  -- This corroborates the indices of the pump probe data 
                            with the frequency at which is was measured
            -ppData     -- shorthand for pump probe data.  Contains the pump probe 
                            spectrum at each measured time point.  Time is represented by row number. 
            -time       -- Tells us what waiting time each row of ppData was measured at               """
                            
                            
    def __init__(self, ppData, axis, time):
        self.probeAxis = axis
        self.ppData = ppData
        self.time = time
#This is a class method that lets us import the class directly from the file. 
    @classmethod
    def fromfile(cls, BaseName, numFiles):
        append = "-T-0.00fs.dat"
        dlambda = 9.54
        offset = 19
    
        for kk in range(1, numFiles + 1):
            fileName = '%s%d%s' % (BaseName, kk, append)
            if (kk > 1):
                ppFile = ppFile + numpy.genfromtxt(fileName, skip_header = 1)
            elif kk == 1:
                ppFile = numpy.genfromtxt(fileName, skip_header=1)
                time = ppFile[:,0]
        #time = ppFile[:,0]
        length = numpy.size(ppFile[1,:])
        ppData = ppFile[:,1:length]
        ppData = ppData/numFiles
        #ppData = numpy.fliplr(ppData)    
        ## Gonna want to pull the center wavelength out of the file so I can generate an axis
        
        fileName = '%s%d%s' % (BaseName, 1, append)    
        tempFile = open(fileName)
        tempString = tempFile.readline()
        tempFile.close()
        tempString = tempString.strip('\n')
        tmp = re.search("=", tempString)   
    
        tempString = tempString[tmp.start() + 2:] 
        #print tempString
        CenterWavelength = float(tempString)
        probeAxis = build_pp_Axis(CenterWavelength, dlambda, offset )    
    
        pp = cls(ppData, probeAxis, time)
        return pp
        
    @classmethod
    def fromfile_loner(cls, fileName,dLambda, offset):
        
        ppFile = numpy.genfromtxt(fileName, skip_header=1)
        time = ppFile[:,0]
        #time = ppFile[:,0]
        length = numpy.size(ppFile[1,:])
        ppData = ppFile[:,1:length]
        #ppData = ppData/numFiles
        #ppData = numpy.fliplr(ppData)    
        ## Gonna want to pull the center wavelength out of the file so I can generate an axis
        
        #fileName = '%s%d%s' % (BaseName, 1, append)    
        tempFile = open(fileName)
        tempString = tempFile.readline()
        tempFile.close()
        tempString = tempString.strip('\n')
        tmp = re.search("=", tempString)   
    
        tempString = tempString[tmp.start() + 2:] 
        #print tempString
        CenterWavelength = float(tempString)
        probeAxis = build_pp_Axis(CenterWavelength, dLambda, offset )    
    
        pp = cls(ppData, probeAxis, time)
        return pp
    
    def plot(self, index):
        plt.plot(self.probeAxis, self.ppData[index,:])
        time = int(self.time[index])
        titlestring = "%s%d%s" % ("Pump Probe data at T = ", time, " fs" )        
        print titlestring        
        plt.title(titlestring)
        plt.show()
        return
        
    def contour(self, *args):
        '''Contour plot of all the pump probe data.  Argument specifies number of contours.  Default is 60.'''
        try:
            numcontours = args[0]
        except:
            numcontours = 60
            
        plt.contour(self.probeAxis, self.time, self.ppData, numcontours)
        plt.colorbar()
        plt.draw()
        plt.show()
        return
        
        
    def Spline(self, *args):
        '''interpolates the pump probe data using cubic splines.  argument 
        allows you to choose how many points to spline into.  Default is 400.'''
        try:
            splinepts = args[0]        
        except:
            splinepts = 400
        time = self.time
        RawProbeAxis = self.probeAxis
        ppData = self.ppData
    
        probeAxis = numpy.linspace(RawProbeAxis[0], RawProbeAxis[-1], splinepts)
        SplineData = numpy.zeros((len(ppData[:,1]),len(probeAxis)))    
        print probeAxis
    
    #The spline function in scipy only works for ascending values of x.  Let's fix that:
    
        if (RawProbeAxis[0] > RawProbeAxis[-1]):
            RawProbeAxis = numpy.flipud(RawProbeAxis)
            probeAxis = numpy.flipud(probeAxis)        
            ppData = numpy.fliplr(ppData)
            flip = True
        else:
            flip = False
        for datasetIndex in range(0, len(ppData[:,1])):
        
            ydata = ppData[datasetIndex,:]                
            tck = interpolate.splrep(RawProbeAxis, ydata)
        #print tck
            SplineData[datasetIndex,:] = interpolate.splev(probeAxis, tck)
    
        if flip == True:
            probeAxis = numpy.flipud(probeAxis)
        
          
        ppSpline = PumpProbe(SplineData, probeAxis, time)  
        return ppSpline        
    
        
    
    def fit(self, datasetIndex, **kwargs):
        axis = self.probeAxis
        data = self.ppData[datasetIndex,:]
        
        try:
            func = kwargs["func"]
        except:
            func = "TwoGaussians"
        
        
        if func == "TwoGaussians":
            fitDictionary
        
        
    
if __name__ == "__main__":
    from sys import argv
    import matplotlib.pyplot as plt
    mu = argv[1]
    sigma = argv[2]
    data = argv[3]
    gauss_data = gauss(mu, sigma, data)
    plt.plot(data,gauss_data)