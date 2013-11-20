# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:37:23 2013

@author: pstraus
"""

from PyQt4 import QtGui
import sys
#import numpy as np
import Process2D as p2d

from test1 import Ui_MainWindow

class killer(QtGui.QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(QtGui.QMainWindow, self).__init__()        
        self.__initVars__()        
        self.setupUi(self)
        self.actionExit.triggered.connect(QtGui.qApp.quit)
        self.actionLoad_File.triggered.connect(self.Loader)
        
        self.pushButton.clicked.connect(self.plotter)
    
    def __initVars__(self):
        
        #tauDefault = np.linspace(1,5,5)
        #probeDefault = np.linspace(1,5,5)
        #dataDefault = np.zeros((5,5))
        self.rawData = p2d.RawData([], [], [],2000)
        self.rDataRaw = p2d.Data2D(1,1,1)
        self.nrDataRaw = p2d.Data2D(1,1,1)
        self.rDataTime = p2d.Data2D(1,1,1)
        self.nrDataTime = p2d.Data2D(1,1,1)
        self.rDataTau = p2d.Data2D(1,1,1)
        self.nrDataTau = p2d.Data2D(1,1,1)
        self.rData = p2d.Data2D(1,1,1)
        self.nrData = p2d.Data2D(1,1,1)
        self.finalData = p2d.Data2D(1,1,1)
        
        self.Parameters = p2d.Parameters2D(9.45, 0, 0)
        
    def doFT1(self):
        #STuff
        pass
    
    def plotter(self):
        self.rawData.contour2D()
        
    def Loader(self):
        fname = QtGui.QFileDialog.getOpenFileName(self,'Open file', 'C:/Users/pstraus/Documents')
        print fname
        fname = str(fname)
        self.rawData = p2d.import2D(fname)
        self.nrDataRaw, self.rDataRaw = p2d.splitRawData(self.rawData, self.Parameters)
        if self.rawData.tauAxis != []:
            self.label.setText("File %s Loaded successfully!" % fname)
        
def main():
    app = QtGui.QApplication(sys.argv)
    Window = killer()
    Window.show()    
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()