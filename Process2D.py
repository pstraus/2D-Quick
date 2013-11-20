# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:46:00 2013

@author: pstraus
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.signal as sig
import re
import scipy.io


def SaveTwoDataMat(fnameBase, rData, nrData):
    fdict = {"rProbeAxis" : rData.probeAxis, "nrProbeAxis" : nrData.probeAxis, "rTauAxis" : rData.tauAxis, "nrTauAxis" : nrData.tauAxis, "nrDataTime": nrData.data, "rDataTime" : rData.data}
    scipy.io.savemat(fnameBase, fdict)
    
def SaveThreeData(fnameBase, finalData,rData, nrData, parameters):

    
    fdict = {"probeAxis" : finalData.probeAxis, "tauAxis" : finalData.tauAxis, "finalData" : finalData.data, "rData" : rData.data, "nrData" : nrData.data, "finalPhase" : parameters.finalPhase}
    scipy.io.savemat(fnameBase,fdict)


def FT2fft(data, tZero):
    
     resolution = 1
     c = float(2.9978e8)
     c_cm = c * 100
     probeAxis = data.probeAxis
     current_res = 1/( abs(probeAxis[0] - probeAxis[-1]) )
     current_res_wavenumbers = current_res/c_cm
     print "res in wavenumbers = %d" % current_res_wavenumbers
     zeroPadFactor = current_res_wavenumbers / resolution
     zeroPadFactor = int(zeroPadFactor)
     if zeroPadFactor > 1:
         zeroPad = zeroPadFactor * len(probeAxis)  
     else:
         zeroPad = None
         
     fftData = np.fft.fft(data.data , n = zeroPad, axis = -1)
     plt.figure()
     plt.contour(fftData)
     plt.show()
     
def FT3(data):
       
    c = 3e8 ;
    c_cm = c*100;
    
    ProbeAxis = data.probeAxis
    TimeAxis = data.tauAxis
    TimeAxis = TimeAxis * 1e-15           #fs to s.  Much easier for everything to work in Hz.
    dT = abs(TimeAxis[0] - TimeAxis[1])  
    print "abs. max of FT3 Data %d" % (np.abs(data.data)).max()
    #if max(TimeAxis) < 0
    #  TimeAxis =  flipud(TimeAxis);
    #end
    #% Determine the Nyquist frequency
    #FNy = 2/abs(TimeAxis(1) - TimeAxis(2))/15 ; 

    #% determine the minimum frequency

    #Fmin = 1/abs(TimeAxis(1) - TimeAxis(end)) * 100 ;
    #freqstep = (FNy - Fmin)/200;


    Fmin =(ProbeAxis.min()) * c_cm ;
    FNy = (ProbeAxis.max()) * c_cm  ;

    print Fmin
    print FNy
    FreqAxis = np.linspace(Fmin,FNy,400); #Fmin:freqstep:FNy;
    
    PumpAxis = FreqAxis / c_cm;         #Puts corresponding frequencies in wavenumbers
    #print PumpAxis
    CoeffMatrix = np.exp( 2j * np.pi * np.outer(FreqAxis, TimeAxis))
    print "abs. max of CoeffMatrix: %d" % CoeffMatrix.max()

    FinalData = np.dot(CoeffMatrix, data.data) * dT
    #plt.figure()
    #plt.contour(FinalData)
    #plt.colorbar()    
    #plt.show()
    FinalDataSet = Data2D(FinalData,PumpAxis, ProbeAxis)
    '''probeAxis = data.probeAxis
    TimeAxis = data.tauAxis
    print TimeAxis
    print probeAxis
    try:
        resolution = float(unicode(self.ResBox.text()))
    except:
        resolution = 1
        
    resolution = float(resolution)
    ScanTime = np.abs(TimeAxis[0] - TimeAxis[-1]) #This is in fs
    ScanTime = float(ScanTime)
    print ScanTime
    ScanRes = 1/ScanTime  #resolution of the scan in fHz
    ScanRes = ScanRes * 1e15 # resolution in Hz
    ScanRes = 3e8/ScanRes * 100 # convert to wavenumbers
    print ScanRes
    zeropadding= ScanRes/resolution * len(TimeAxis)
    
    datafft = np.fft.fftshift(np.fft.ifft(data.data, n= zeropadding, axis = 0), axes = 0)
    plt.figure()    
    plt.contour(datafft, 60)
    plt.colorbar()
    plt.show()
    FinalDataSet = Data2D(1,1,1)'''
    return FinalDataSet    
    
def FT2(data, RawProbeAxis, tZero):
    #Performs the FT back to the time domain
    #Detailed explanation goes here
    c = 2.9979e08 ;  #speed of light in m/s
    c_cm = c* 100;
    
    
    Taxis = data.probeAxis
    Taxis = Taxis - tZero *1e-15;
    dT = abs(Taxis[0] - Taxis[1]) ; #Absolute value because I don't want to think about positivity.
    #if Taxis(1) < Taxis(2)
    #    Taxis = fliplr(Taxis * -1);
    #end
    
    #fstep = 1/5*abs(RawProbeAxis(1) - RawProbeAxis(2)) ;

    ProbeAxis = np.linspace(RawProbeAxis[0], RawProbeAxis[-1], 400)   ; #make a new probe axis with more points.  This makes the data appear smoother
    #ProbeAxis = fliplr(ProbeAxis);
    ProbeAxisHz = c_cm * ProbeAxis                                    #%The other FT was done in Hz (even though the corresponding axis was not)!

    CoeffMatrix = np.exp( -2j * np.pi * np.outer(Taxis,  ProbeAxisHz ) ) ; 
    ProcessedData = dT * np.dot(data.data, CoeffMatrix)  ;
    FinalProduct = Data2D(ProcessedData, data.tauAxis, ProbeAxis)
    return FinalProduct
    
    
def FT1(data, tZero, NyquistMultiplier, TransformTime):
    c = 2.9979 * 10 **8# speed of light in m/s     
    c_cm = c * 100        
    tZero = tZero * 10 ** -15 
    
    ProbeAxis = data.probeAxis

    dOmega = BuildDOmega( ProbeAxis )
    #We want to take the integral of our function against exp(+2*pi*i*w*t) over omega.
    #This is equilvalent to a matrix with columns of the factor for each time
    #point t and each row is the frequency dependence at that time
    ProbeFreq =  c_cm * ProbeAxis                # %Probe axis in hz
    #TimeLength =  min(ProbeFreq)**-1 * 200     #transform N times the nyquist frequency to get a good reproduction of the signal.    
    TimeLength = TransformTime/1e15
    #To determine the time points we need, we must find the Nyquist frequency
    #for the data we collected.  This is two times the minimum sampling rate to
    #find measure the frequencies that we collected in the time domain.  Let's
    #then multiply this number by four to get a reasonable timestep.  That is,
    #four times the maximum frequency we have.  Note that we're in wavenumbers
    
    SampTime = max(ProbeFreq)**-1 / NyquistMultiplier
    # Convert to seconds/sample with Nyquist adjustment
    #SampTime = 2;
    numSamples = TimeLength/SampTime
    Taxis = np.linspace(0 + tZero, TimeLength + tZero, numSamples)     

    #Taxis = Taxis*1e-15;
    #print "Taxis Size: %d" % Taxis.shape()
    #print "ProbeFreq size: %d" ProbeFreq.shape()    
    CoeffMatrix = np.exp(1j * 2 * np.pi *  np.outer(ProbeFreq,  Taxis ))  
      
    Tdata =  np.dot( np.dot(data.data, dOmega), CoeffMatrix)  
    timeDataObject = Data2D(Tdata,data.tauAxis, Taxis)
    return timeDataObject
    
def BuildDOmega( ProbeAxis):
    c_nm = 2.9979 * (10 **17);
    DeltaLambda = np.abs(ProbeAxis[0] ** -1 - ProbeAxis[1] ** -1) ; #dOmega in femto Hz.

    dOmega = np.zeros(len(ProbeAxis))
    for element in range(0, len(ProbeAxis)):
        lambda1 = ProbeAxis[element] ** -1 + DeltaLambda/2 
        lambda2 = ProbeAxis[element] ** -1 - DeltaLambda/2 
    
        nu1 = 2 * np.pi * 1/lambda1 * c_nm    
        nu2 = 2 * np.pi * 1/lambda2 * c_nm    
    
        dOmega[element] = np.abs(nu1 - nu2)
    dOmega = np.diag(dOmega)
    return dOmega


def relPhase2D(data1, data2, *args):
    #print data1.zeroSlice()
    try:
        phaseGuess = args[0];
    except:
        phaseGuess = 0;
        
    rData = (data1.zeroSlice())
    nrData = (data2.zeroSlice())
    #print rData
    #plt.plot(data1.probeAxis,data1.zeroSlice(),'b', data2.probeAxis, data2.zeroSlice(),'r')
    #plt.show()    
    fitVars, cov_matrix = optimize.curve_fit(phasefitfunc, rData, nrData, [phaseGuess,1])
    #rDataPhased = np.exp(2 * np.pi * 1j * rel_phase) * rData
    #rel_phase = fitVars[0]
    #amplitude = fitVars[1]

    #plt.figure()    
    #plt.plot(data1.probeAxis, nrData,'r', data1.probeAxis, phasefitfunc(rData, *fitVars),'b')
    #plt.show()    
    return fitVars
    
def phasefitfunc(data, phase, amplitude):
    return amplitude * np.real((np.exp(2j * np.pi * phase) * sig.hilbert(data)))

def phaseRtoNR(RData, phase, amplitude):
    for index in range(0, len(RData[:,0])):
        RData[index,:] = phasefitfunc(RData[index,:], phase, amplitude)
    return RData
    
def import2D(fileName):
    '''It imports a 2D spectrum given its filename'''
      
    #Opens the file and just plucks out the center wavelength   
    tempFile = open(fileName)
    tempString = tempFile.readline()
    tempFile.close()
    tempString = tempString.strip('\n')
    tmp = re.search("=", tempString)   
    
    tempString = tempString[tmp.start() + 2:] 
    print "Center Wavelength is %s" % tempString
    CenterWavelength = float(tempString)     
   
    #reopens the file and loads all the data
    RawDataSet = np.genfromtxt(fileName, skip_header = 1)
    #print RawDataSet
    tauAxis = RawDataSet[:,0] 
    rawData = RawDataSet[:,1:]
    rawAxis = np.linspace(1,len(rawData[1,:]),len(rawData[1,:]))
      
    #print rawData
    Raw = RawData(rawData, tauAxis, rawAxis, CenterWavelength)
    return Raw
    
def splitRawData(RawData, parameters):
    rDataStartIndex = np.argmin(RawData.tauAxis[0:len(RawData.tauAxis)] * RawData.tauAxis[0:len(RawData.tauAxis)])
    
    #line below for debugging    
    #print rDataStartIndex
    
    rDataStartIndex = rDataStartIndex + 1
    nrDataTauAxis = RawData.tauAxis[0:rDataStartIndex]
    rDataTauAxis = RawData.tauAxis[(rDataStartIndex):]
    
    nrData = RawData.data[0:rDataStartIndex,:]
    rData = RawData.data[(rDataStartIndex):,:]
    
    ProbeAxis = buildProbeAxis(RawData.centerWavelength, parameters.dLambda, offset=parameters.offset, det_elts=parameters.det_elts)
      
    nrDataSet = Data2D(nrData, nrDataTauAxis, ProbeAxis)
    rDataSet = Data2D(rData, rDataTauAxis, ProbeAxis)   
    #nrDataSet.contour2D()
    return (nrDataSet, rDataSet)
    
def buildProbeAxis(centerWavelength, dlambda, **kwargs):
#get specified number of detector elements.  If none is specified, use a default
    defaults = {"det_elts": 64, "offset": 19}

    for k, v in defaults.iteritems():
        try:
            defaults[k] = kwargs[k]
        except:
            pass
    
    numelts = defaults["det_elts"]
    offset = defaults["offset"]
    
    centerWavelength = centerWavelength + offset #this is a feature for lazy people.  If you corrected it already, please just use '0' for the offset
    dlambda = float(dlambda)
      
    axis = np.zeros(numelts)    
    
    for element in range(0, numelts):
        axis[element] = dlambda * (element - float(numelts)/2.0) + centerWavelength  
       
    axis = axis* 1e-7               #convert to cm
    axis = axis ** -1          #convert to cm-1
    axis = np.flipud(axis)
    return axis     
    
class Data2D(object):
    '''The point of this class is to contain a 2D dataset and its respective axes, irrespective of domain'''
    def __init__(self, data, tauAxis, probeAxis):
        
        self.data = data
        self.probeAxis = probeAxis
        self.tauAxis = tauAxis
        
    def plotProbe(self, tauSlice):
        '''Plots the probeAxis spectrum from given slice of the tauAxis
        Useful for determining the rephasing and nonrephasing relative Phase'''
        
        plt.plot( self.probeAxis, self.zeroSlice  )
        
    def contour2D(self, contours=60):
        '''Super basic plotting for 2D spectra.  Can implement more advanced plotting options outside the class.'''
        print self.data        
        plt.contour(self.probeAxis, self.tauAxis, self.data, contours)
        plt.colorbar()
        plt.show()
        
    def zeroSlice(self):
        '''return the slice of the spectrum at tau = zero waiting time'''
        sliceIndex = np.argmin((self.tauAxis * self.tauAxis))
        return self.data[sliceIndex, :]
        
    def Spline(self, *args):
        '''interpolates the pump probe data using cubic splines.  argument 
        allows you to choose how many points to spline into.  Default is 400.'''
        try:
            splinepts = args[0]        
        except:
                splinepts = 400
        
        RawProbeAxis = self.probeAxis
        data = self.data
    
        NewProbeAxis = np.linspace(RawProbeAxis[0], RawProbeAxis[-1], splinepts)
        SplineData = np.zeros((len(data[:,1]),len(NewProbeAxis)))    
        print NewProbeAxis
    
    #The spline function in scipy only works for ascending values of x.  Let's fix that:
    
        if (RawProbeAxis[0] > RawProbeAxis[-1]):
            RawProbeAxis = np.flipud(RawProbeAxis)
            NewProbeAxis = np.flipud(NewProbeAxis)        
            data = np.fliplr(data)
            flip = True
        else:
            flip = False
        
        for datasetIndex in range(0, len(data[:,1])):
        
            ydata = data[datasetIndex,:]                
            tck = interpolate.splrep(RawProbeAxis, ydata)
            #print tck
            SplineData[datasetIndex,:] = interpolate.splev(NewProbeAxis, tck)
    
        if flip == True:
            NewProbeAxis = np.flipud(NewProbeAxis)
        
          
        dataSpline = Data2D(SplineData, self.tauAxis, NewProbeAxis)  
        return dataSpline        

class RawData(Data2D):
    def __init__(self, data, tauAxis, probeAxis, centerWavelength):
        self.centerWavelength = centerWavelength
        super(RawData,self).__init__(data, tauAxis, probeAxis)
        
class Parameters2D:
    def __init__(self, dLambda, offset, tZero):
        self.dLambda = dLambda
        self.offset = offset
        self.tZero = tZero
        self.det_elts = 64
        self.finalPhase = 0
        self.rel_phase = 0
        self.centerWavelength = 0
        self.old_multiplier = 0
        self.ppIndex = 0
        self.ppLoad = False
