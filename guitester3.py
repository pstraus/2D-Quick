# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:13:35 2013

@author: pstraus
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:37:23 2013

@author: pstraus
"""

from PyQt4 import QtGui
import sys
#import numpy as np
import Process2D as p2d
import matplotlib.pyplot as plt
from GuiCode import Ui_MainWindow
import numpy as np
import pp_funcs as pp
import pickle as pickle

class killer(QtGui.QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(QtGui.QMainWindow, self).__init__()        
        self.__initVars__()        
        self.setupUi(self)
        self.actionExit.triggered.connect(QtGui.qApp.quit)
        self.actionLoad_File.triggered.connect(self.Loader)
        
        self.actionFT1_Data.triggered.connect(self._saveFT1Data)
        self.actionFT2_Data.triggered.connect(self._saveFT2Data)        
        self.actionLoad_Parameter_File.triggered.connect(self._loadParameters)        
        
        
        self.actionFinal_Data.triggered.connect(self._saveFinalData)
        self.actionSave_Parameters.triggered.connect(self._saveParameters)
        self.RawDataPlotButton.clicked.connect(self.plotter)
        self.FT1PlotButton.clicked.connect(self.plotFT1)
        
    def __initVars__(self):
        
        #tauDefault = np.linspace(1,5,5)
        #probeDefault = np.linspace(1,5,5)
        #dataDefault = np.zeros((5,5))
        self.rawData = p2d.RawData([], [], [],2000)
        self.rDataRaw = p2d.Data2D(1,1,1)
        self.rDataRawPhased = p2d.Data2D(1,1,1)        
        self.nrDataRaw = p2d.Data2D(1,1,1)
        self.rDataTime = p2d.Data2D(1,1,1)
        self.nrDataTime = p2d.Data2D(1,1,1)
        self.rDataTau = p2d.Data2D(1,1,1)
        self.nrDataTau = p2d.Data2D(1,1,1)
        self.rData = p2d.Data2D(1,1,1)
        self.nrData = p2d.Data2D(1,1,1)
        self.finalData = p2d.Data2D(1,1,1)
        self.ppProjection = pp.PumpProbe(1,1,1)
        self.ppLoad = pp.PumpProbe(1,1,1)
        self.Parameters = p2d.Parameters2D(9.45, 19, 0)
        
    def doFT1(self):
        '''Fourier transforms along the probe axis into the time domain.  We can then filter out unwanted artifacts, particularly those due to the local oscillator.'''
        TransformTime = float(self.TransformTime.text())
        NyquistMult = self.NyquistMult.value()
        tZero = unicode(self.FT1tZero.text())
        print tZero
        tZero = float(tZero)
        self.tZeroBox.setText(self.FT1tZero.text())
        self.Parameters.tZero = tZero
        self.Parameters.dLambda = float(self.dLambdaBox.text())
        
        phaseGuess = float(self.phaseGuess.text())     
        
        fitVars = p2d.relPhase2D(self.rDataRaw, self.nrDataRaw, phaseGuess)
        self.rDataRawPhased.data = self.rDataRaw.data
        self.rDataRawPhased.probeAxis = self.rDataRaw.probeAxis
        self.rDataRawPhased.tauAxis = self.rDataRaw.tauAxis
        self.RelativePhaseBox.setText(str(fitVars[0]))
        self.Parameters.rel_phase = fitVars[0]
        #make a copy of rephasing data so we can phase it while keeping the raw data intact        
        #self.rDataRawPhased.data = p2d.phaseRtoNR(self.rDataRaw.data, *fitVars)
        self.RelPhasePlot.axes.plot(self.rDataRaw.probeAxis, self.rDataRaw.zeroSlice(), 'b',self.rDataRaw.probeAxis, self.nrDataRaw.zeroSlice()* np.exp(2j * np.pi * fitVars[0]),'g')
        #self.RelPhasePlot.axes.plot(self.rDataRaw.probeAxis, self.nrDataRaw.zeroSlice())
        self.RelPhasePlot.draw()        
        #plt.figure()
        #plt.plot(self.rDataRawPhased.probeAxis, self.rDataRawPhased.zeroSlice(),self.nrDataRaw.probeAxis, self.nrDataRaw.zeroSlice())
        #self.rData.data = p2d.phasefitfunc(self.rData.data, *fitVars)
        self.nrDataTime = p2d.FT1(self.nrDataRaw, tZero, NyquistMult, TransformTime)
        self.rDataTime = p2d.FT1(self.rDataRaw, tZero, NyquistMult, TransformTime)
        plt.figure()        
        plt.plot(self.nrDataTime.probeAxis, self.nrDataTime.zeroSlice(), self.rDataTime.probeAxis, self.rDataTime.zeroSlice()* np.exp(-2j * np.pi * fitVars[0]) * fitVars[1])
        plt.show()
    
    def doFT2(self):
        '''We are basically just calling a function that is the inverse of the doFT1 action.'''
        self.ft2Indicator.setText("Please Wait")        
        tZero = self.Parameters.tZero
        self.nrDataTau = p2d.FT2(self.nrDataTime, self.nrDataRaw.probeAxis, tZero)
        self.rDataTau = p2d.FT2(self.rDataTime, self.rDataRaw.probeAxis, tZero)
         
        self.ft2Indicator.setText("FT2 Complete!") 
        #plt.figure()
        #self.nrDataTau.contour2D()
        #plt.figure()
        #self.rDataTau.contour2D()
        #print self.nrDataTau.data
        
    def doFT3(self):
        
        self.rData = p2d.FT3(self.rDataTau)
        self.rData.data = self.rData.data * np.exp(1j * np.pi * self.Parameters.rel_phase)
        self.nrData = p2d.FT3(self.nrDataTau)
        self.nrData.data = self.nrData.data * np.exp(-1j * np.pi * self.Parameters.rel_phase)
        self.finalData.probeAxis = self.rData.probeAxis  #getAxis information
        self.finalData.tauAxis = self.rData.tauAxis
       
        self.finalData.data = (self.rData.data + self.nrData.data)
        print np.abs(self.finalData.data).max()
        print np.abs(self.nrData.data).max()
        print np.abs(self.rData.data).max()
        
        self.PlotFinalData()
        
        pptmp = np.sum(np.real(self.finalData.data * np.exp(2j * np.pi * self.Parameters.finalPhase)), axis = 0)  
        pptmp = pptmp/max(np.abs(pptmp))     
        self.ppProjection = pp.PumpProbe(pptmp, self.finalData.probeAxis, 0)     
        self.Update_pp_Plot()        
        self.RelPhaseEdit.setText(str(self.Parameters.rel_phase))       
        self.RelPhaseScroll.setValue(int(self.Parameters.rel_phase * 100))
        #plt.figure()
        #plt.contour(self.rData.probeAxis, self.rData.tauAxis,np.real(self.rData.data * np.exp(-2j * np.pi * self.Parameters.finalPhase)), 70)
        #plt.title("Rephasing Data")        
        #plt.colorbar()        
        #plt.show()
        #plt.figure()
        #plt.contour(self.nrData.probeAxis, self.nrData.tauAxis, np.real(self.nrData.data* np.exp(-2j * np.pi * self.Parameters.finalPhase)), 70)
        #plt.colorbar()
        #plt.title("Non Rephasing Data")
           #self.FinalDataPlot.contour(self.finalData.probeAxis,self.finalData.tauAxis, self.finalData.Data)
        #plt.figure()
        #plt.contour(self.finalData.probeAxis,self.finalData.tauAxis, np.real(self.finalData.data* np.exp(-2j * np.pi * self.Parameters.finalPhase)), 70)
        #plt.title("Purely absorptive spectrum")        
        #plt.show()
       
    def plotter(self):
        #self.rawData.contour2D()
        plt.contour(self.rawData.data, 60)
        plt.colorbar()
        plt.show()
    def plotFT1(self):
        plt.figure()
        plt.plot(self.nrDataTime.probeAxis, self.nrDataTime.zeroSlice(), self.rDataTime.probeAxis, self.rDataTime.zeroSlice())
        plt.show()
        
    def updatePhaseBox(self):
        print unicode(self.PhaseBox.size())
        self.PhaseBox.setText(str(float(self.PhaseSlider.value())/100))
        self.Parameters.finalPhase = float(self.PhaseSlider.value())/100
        print self.Parameters.finalPhase
        self.PlotFinalData()
        
        pptmp = np.sum(np.real(self.finalData.data * np.exp(2j * np.pi * self.Parameters.finalPhase)), axis = 0)  
        pptmp = pptmp/max(np.abs(pptmp))     
        self.ppProjection = pp.PumpProbe(pptmp, self.finalData.probeAxis, 0)     
        self.Update_pp_Plot()
        
    def updatePhase(self):
        self.PhaseSlider.setValue(int(100 * float(unicode(self.PhaseBox.text()))))
        self.Parameters.finalPhase = float(self.PhaseSlider.value())/100
        #print self.Parameters.finalPhase  
        
        pptmp = np.sum(np.real(self.finalData.data * np.exp(2j * np.pi * self.Parameters.finalPhase)), axis = 0) 
        pptmp = pptmp/max(np.abs(pptmp))     
        self.ppProjection = pp.PumpProbe(pptmp, self.finalData.probeAxis, 0)        
        self.PlotFinalData()
        self.Update_pp_Plot()
        
    def PlotNR(self):
        plt.figure()
        plt.contour(self.nrData.probeAxis, self.nrData.tauAxis, np.real(self.nrData.data* np.exp(2j * np.pi * self.Parameters.finalPhase)), 70)
        plt.colorbar()
        plt.title("Non Rephasing Data")  
        plt.show()
        
    def PlotR(self):
        plt.figure()
        plt.contour(self.rData.probeAxis, self.rData.tauAxis,np.real(self.rData.data * np.exp(2j * np.pi * self.Parameters.finalPhase)), 70)
        plt.title("Rephasing Data")        
        plt.colorbar()        
        plt.show()
        
    def Update_pp_Plot(self):
        
        if self.Parameters.ppLoad == False:        
            self.ppPlot.axes.plot(self.ppProjection.probeAxis, self.ppProjection.ppData)
        else:
            index = self.Parameters.ppIndex
            ppPlot = self.ppLoad.ppData[index,:]/np.max(self.ppLoad.ppData[index,:])
            self.ppPlot.axes.plot(self.ppProjection.probeAxis, self.ppProjection.ppData, 'b', self.ppLoad.probeAxis, ppPlot,'r')
        self.ppPlot.draw()
    
    
    def PlotFinalData(self):
        #plt.figure()
        finalData = self.finalData.data * np.exp(2j * np.pi * self.Parameters.finalPhase)
        #plt.contour(self.finalData.probeAxis,self.finalData.tauAxis, np.real(finalData), 70)
        #plt.title("Purely absorptive spectrum")        
        #plt.show()
        self.FinalDataPlot.axes.contour(self.finalData.probeAxis,self.finalData.tauAxis, np.real(finalData), 70)
        #self.FinalDataPlot.axes.colorbar()
        #self.FinalDataPlot.figure.title("Purely absorptive spectrum")        
        self.FinalDataPlot.draw()
    
    def PlotFinalSeparate(self):
        plt.figure()
        plt.contour(self.finalData.probeAxis, self.finalData.tauAxis, np.real(np.exp(2j * np.pi * self.Parameters.finalPhase)*self.finalData.data), 80)
        plt.title("Purely Absorptive 2DIR Spectrum")
        plt.colorbar()
        plt.show()
        
    def Loader(self):
        fname = QtGui.QFileDialog.getOpenFileName(self,'Open file', 'C:/Users/pstraus/Documents')
        #print fname
        fname = str(fname)
        
        self.rawData = p2d.import2D(fname)
        #self.rawData.data[10,:] = 0.5 * self.rawData.data[11,:] + 0.5 * self.rawData.data[9,:]        
        #self.rawData.data[14,:] = 0.5 * self.rawData.data[15,:] + 0.5 * self.rawData.data[13,:]        

        self.rawData.data[:, 9] = 0.5 * self.rawData.data[:, 10] + 0.5 * self.rawData.data[:,8]        
        self.rawData.data[:, 13] = 0.5 * self.rawData.data[:,14] + 0.5 * self.rawData.data[:,12]        
        
        self.nrDataRaw, self.rDataRaw = p2d.splitRawData(self.rawData, self.Parameters)
        
        #Make sure that the tau data is a non-empty set
        if self.rawData.tauAxis != []:
            self.FileIndicator.setText("File %s Loaded successfully!" % fname)
            
        #get the default MC offset...
        self.set_MC_Offset()
        self.Parameters.centerWavelength = self.rawData.centerWavelength
        
        #Guess a good amount for the first FT
        TransformTime = int((min(self.rDataRaw.probeAxis) * 3e10) ** -1 * 1e15)* 230 #default transform time
                
        self.TransformTime.setText(str(TransformTime))
        #self.rDataRaw.data[]        
        #self.rDataRaw = self.rDataRaw.Spline()
        #self.nrDataRaw = self.nrDataRaw.Spline()
    def set_MC_Offset(self):
        offset = float(unicode(self.MC_Offset.text()))
        self.Parameters.offset = offset
    
    def _update_dLambda(self):
        self.Parameters.dLambda = float(self.dLambdaBox.text())
        
    def Change_tZero(self):
        
        tZero = unicode(self.tZeroBox.text())
        tZero = float(tZero)
        self.FT1tZero.setText(self.tZeroBox.text())
        self.Parameters.tZero = tZero
        #self.Parameters.dLambda = float(self.dLambdaBox.text())
        #fitVars = p2d.relPhase2D(self.rDataRaw, self.nrDataRaw)
        #self.rDataRawPhased.data = self.rDataRaw.data
        #self.rDataRawPhased.probeAxis = self.rDataRaw.probeAxis
        #self.rDataRawPhased.tauAxis = self.rDataRaw.tauAxis
        #self.RelativePhaseBox.setText(str(fitVars[0]))
        #self.Parameters.rel_phase = fitVars[0]
        #make a copy of rephasing data so we can phase it while keeping the raw data intact        
        #self.rDataRawPhased.data = p2d.phaseRtoNR(self.rDataRaw.data, *fitVars)
        
        #plt.figure()
        #plt.plot(self.rDataRawPhased.probeAxis, self.rDataRawPhased.zeroSlice(),self.nrDataRaw.probeAxis, self.nrDataRaw.zeroSlice())
        #self.rData.data = p2d.phasefitfunc(self.rData.data, *fitVars)
        self.nrDataTime = p2d.FT1(self.nrDataRaw, tZero, 3, float(self.TransformTime.text()))
        self.rDataTime = p2d.FT1(self.rDataRaw, tZero, 3, float(self.TransformTime.text()))
        
        self.doFT2()
        self.doFT3()
            
    def _update_tZero(self):
        cycle = self.Parameters.centerWavelength
        cycle_time = cycle/(3e8 * 1e9) * 1e15;
        tZero = float(self.tZeroBox.text())
        
        #need to keep track of the values for the tZero slider for the jumps to make sense.
        # could also keep track of original time zero.  this seems easier.        
        oldmult = self.Parameters.old_multiplier
        multiplier = self.tZeroSlider.value()
        self.Parameters.old_multiplier = multiplier
        
        multiplier = multiplier - oldmult
        tZero = tZero + multiplier*cycle_time;
        self.tZeroBox.setText(str(tZero))
        self.Change_tZero()        

    def loadPP(self):
        fname = QtGui.QFileDialog.getOpenFileName(self,'Open file', 'C:/Users/pstraus/Documents')
        #print fname
        fname = str(fname)
        self.ppLoad = pp.PumpProbe.fromfile_loner(fname, self.Parameters.dLambda, self.Parameters.offset)        
        self.Parameters.ppLoad = True       
        self.Update_pp_Plot()       
        
    def _update_ppIndex(self):
        try:
            self.Parameters.ppIndex = self.ppSpinBox.value()
            ppTime = self.ppLoad.time[self.ppSpinBox.value()]
            self.ppTimeLabel.setText(str(int(ppTime)))
            self.Update_pp_Plot()        
        except:
            print "You didn't load a pump-probe!"
            
            
    def _saveFinalData(self):
        fname = str(QtGui.QFileDialog.getSaveFileName(self, "Save File", ""))
        p2d.SaveThreeData(fname, self.finalData,self.rData, self.nrData, self.Parameters)
        
    def _saveFT1Data(self):
        fname = str(QtGui.QFileDialog.getSaveFileName(self, "Save FT1 Data", ""))
        p2d.SaveTwoDataMat(fname, self.rDataTime, self.nrDataTime)
        
    def _saveFT2Data(self):
        fname = str(QtGui.QFileDialog.getSaveFileName(self,"Save FT2 Data", ""))
        p2d.SaveTwoDataMat(fname, self.rDataTau, self.nrDataTau)
    
    def _saveParameters(self):
        fname = str(QtGui.QFileDialog.getSaveFileName(self, "Save the Parameters", ".params"))
        dumpfile = open(fname,'w')
        pickle.dump(self.Parameters,dumpfile)
        dumpfile.close()
        
    def _loadParameters(self):
        fname = str(QtGui.QFileDialog.getOpenFileName(self, "Open the Parameters"))
        dumpfile = open(fname,'r')
        self.Parameters = pickle.load(dumpfile)
        self.MC_Offset.setText(str(self.Parameters.offset))
        self.FT1tZero.setText(str(self.Parameters.tZero))
        self.dLambdaBox.setText(str(self.Parameters.dLambda))
        self.PhaseBox.setText(str(self.Parameters.finalPhase))
        self.tZeroBox.setText(str(self.Parameters.tZero))
        self.RelativePhaseBox.setText(str(self.Parameters.rel_phase))
        
        #Since the slider will start from the middle now        
        self.Parameters.old_multiplier = 0
        self.Parameters.ppLoad = False
    def _plotRtau(self):
        plt.figure()
        plt.contour(self.rDataTau.probeAxis, self.rDataTau.tauAxis, self.rDataTau.data, 60)
        plt.show()
        
    def _plotNRtau(self):
        plt.figure()
        plt.contour(self.nrDataTau.probeAxis, self.nrDataTau.tauAxis, self.nrDataTau.data, 60)
        plt.show()
        
        
    def _plotRtauAbs(self):
        plt.figure()
        plt.contour(self.rDataTau.probeAxis, self.rDataTau.tauAxis, np.abs(self.rDataTau.data), 60)
        plt.show()
        
    def _plotNRtauAbs(self):
        plt.figure()
        plt.contour(self.nrDataTau.probeAxis, self.nrDataTau.tauAxis, np.abs(self.nrDataTau.data), 60)
        plt.show()     
    
    def _updateRelPhase(self):
        newval = float(unicode(self.RelPhaseEdit.text()))
        delta = newval - self.Parameters.rel_phase        
        self.Parameters.rel_phase = newval        
        self.RelPhaseScroll.setValue(int(100*self.Parameters.rel_phase))        
        self.rData.data = self.rData.data * np.exp(1j * np.pi * delta)
        self.nrData.data = self.nrData.data * np.exp(-1j * np.pi * delta)
       
        self.finalData.data = (self.rData.data + self.nrData.data)
        self.PlotFinalData()      
        pptmp = np.sum(np.real(self.finalData.data * np.exp(2j * np.pi * self.Parameters.finalPhase)), axis = 0)  
        pptmp = pptmp/max(np.abs(pptmp))     
        self.ppProjection = pp.PumpProbe(pptmp, self.finalData.probeAxis, 0)     
        self.Update_pp_Plot()        
        
    def _updateRelPhaseSlider(self):
        newval = float((self.RelPhaseScroll.value()))/100
        #current = self.Parameters.rel_phase
        #newval = current + delta
        self.RelPhaseEdit.setText(unicode(str((newval))))
        
        delta = newval - self.Parameters.rel_phase        
        
        self.Parameters.rel_phase = newval
              
        
        self.rData.data = self.rData.data * np.exp(1j * np.pi * delta)
        self.nrData.data = self.nrData.data * np.exp(-1j * np.pi * delta)
       
        self.finalData.data = (self.rData.data + self.nrData.data)
        self.PlotFinalData()      
        pptmp = np.sum(np.real(self.finalData.data * np.exp(2j * np.pi * self.Parameters.finalPhase)), axis = 0)  
        pptmp = pptmp/max(np.abs(pptmp))     
        self.ppProjection = pp.PumpProbe(pptmp, self.finalData.probeAxis, 0)     
        self.Update_pp_Plot()
 
        
def main():
    app = QtGui.QApplication(sys.argv)
    Window = killer()
    Window.show()    
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()