# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test3.ui'
#
# Created: Mon Nov 18 21:45:40 2013
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(984, 863)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.RawDataPlotButton = QtGui.QPushButton(self.centralwidget)
        self.RawDataPlotButton.setGeometry(QtCore.QRect(10, 50, 81, 21))
        self.RawDataPlotButton.setObjectName(_fromUtf8("RawDataPlotButton"))
        self.FileIndicator = QtGui.QLabel(self.centralwidget)
        self.FileIndicator.setGeometry(QtCore.QRect(10, 10, 951, 31))
        self.FileIndicator.setObjectName(_fromUtf8("FileIndicator"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 90, 961, 711))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.FT1 = QtGui.QWidget()
        self.FT1.setObjectName(_fromUtf8("FT1"))
        self.FT1Button = QtGui.QPushButton(self.FT1)
        self.FT1Button.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.FT1Button.setObjectName(_fromUtf8("FT1Button"))
        self.FT1PlotButton = QtGui.QPushButton(self.FT1)
        self.FT1PlotButton.setGeometry(QtCore.QRect(10, 70, 75, 23))
        self.FT1PlotButton.setObjectName(_fromUtf8("FT1PlotButton"))
        self.label_2 = QtGui.QLabel(self.FT1)
        self.label_2.setGeometry(QtCore.QRect(210, 10, 51, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.RelativePhaseBox = QtGui.QLineEdit(self.FT1)
        self.RelativePhaseBox.setGeometry(QtCore.QRect(250, 450, 71, 20))
        self.RelativePhaseBox.setObjectName(_fromUtf8("RelativePhaseBox"))
        self.label = QtGui.QLabel(self.FT1)
        self.label.setGeometry(QtCore.QRect(160, 450, 81, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.FT1tZero = QtGui.QLineEdit(self.FT1)
        self.FT1tZero.setGeometry(QtCore.QRect(270, 10, 71, 20))
        self.FT1tZero.setObjectName(_fromUtf8("FT1tZero"))
        self.NyquistMult = QtGui.QSpinBox(self.FT1)
        self.NyquistMult.setGeometry(QtCore.QRect(270, 60, 42, 22))
        self.NyquistMult.setProperty("value", 15)
        self.NyquistMult.setObjectName(_fromUtf8("NyquistMult"))
        self.label_4 = QtGui.QLabel(self.FT1)
        self.label_4.setGeometry(QtCore.QRect(170, 60, 91, 21))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.TransformTime = QtGui.QLineEdit(self.FT1)
        self.TransformTime.setGeometry(QtCore.QRect(270, 120, 41, 20))
        self.TransformTime.setObjectName(_fromUtf8("TransformTime"))
        self.label_5 = QtGui.QLabel(self.FT1)
        self.label_5.setGeometry(QtCore.QRect(180, 120, 81, 21))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_6 = QtGui.QLabel(self.FT1)
        self.label_6.setGeometry(QtCore.QRect(320, 120, 46, 21))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.phaseGuess = QtGui.QLineEdit(self.FT1)
        self.phaseGuess.setGeometry(QtCore.QRect(250, 420, 71, 20))
        self.phaseGuess.setObjectName(_fromUtf8("phaseGuess"))
        self.label_9 = QtGui.QLabel(self.FT1)
        self.label_9.setGeometry(QtCore.QRect(200, 420, 46, 13))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.RelPhasePlot = MatplotlibWidget(self.FT1)
        self.RelPhasePlot.setGeometry(QtCore.QRect(110, 510, 701, 121))
        self.RelPhasePlot.setObjectName(_fromUtf8("RelPhasePlot"))
        self.tabWidget.addTab(self.FT1, _fromUtf8(""))
        self.FT2 = QtGui.QWidget()
        self.FT2.setObjectName(_fromUtf8("FT2"))
        self.FT2Button = QtGui.QPushButton(self.FT2)
        self.FT2Button.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.FT2Button.setObjectName(_fromUtf8("FT2Button"))
        self.ft2Indicator = QtGui.QLabel(self.FT2)
        self.ft2Indicator.setGeometry(QtCore.QRect(20, 90, 201, 61))
        self.ft2Indicator.setObjectName(_fromUtf8("ft2Indicator"))
        self.pushButton = QtGui.QPushButton(self.FT2)
        self.pushButton.setGeometry(QtCore.QRect(270, 102, 101, 21))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_2 = QtGui.QPushButton(self.FT2)
        self.pushButton_2.setGeometry(QtCore.QRect(270, 140, 101, 23))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_3 = QtGui.QPushButton(self.FT2)
        self.pushButton_3.setGeometry(QtCore.QRect(410, 100, 75, 23))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_4 = QtGui.QPushButton(self.FT2)
        self.pushButton_4.setGeometry(QtCore.QRect(410, 140, 75, 23))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.tabWidget.addTab(self.FT2, _fromUtf8(""))
        self.FT3 = QtGui.QWidget()
        self.FT3.setObjectName(_fromUtf8("FT3"))
        self.FT3Button = QtGui.QPushButton(self.FT3)
        self.FT3Button.setGeometry(QtCore.QRect(0, 20, 75, 23))
        self.FT3Button.setObjectName(_fromUtf8("FT3Button"))
        self.PhaseSlider = QtGui.QScrollBar(self.FT3)
        self.PhaseSlider.setGeometry(QtCore.QRect(0, 90, 160, 16))
        self.PhaseSlider.setMinimum(-100)
        self.PhaseSlider.setMaximum(100)
        self.PhaseSlider.setSingleStep(5)
        self.PhaseSlider.setPageStep(50)
        self.PhaseSlider.setOrientation(QtCore.Qt.Horizontal)
        self.PhaseSlider.setObjectName(_fromUtf8("PhaseSlider"))
        self.tZeroSlider = QtGui.QScrollBar(self.FT3)
        self.tZeroSlider.setGeometry(QtCore.QRect(0, 130, 160, 16))
        self.tZeroSlider.setMinimum(-100)
        self.tZeroSlider.setMaximum(100)
        self.tZeroSlider.setPageStep(5)
        self.tZeroSlider.setOrientation(QtCore.Qt.Horizontal)
        self.tZeroSlider.setObjectName(_fromUtf8("tZeroSlider"))
        self.ppPlot = MatplotlibWidget(self.FT3)
        self.ppPlot.setGeometry(QtCore.QRect(170, 500, 711, 151))
        self.ppPlot.setObjectName(_fromUtf8("ppPlot"))
        self.pushButton_5 = QtGui.QPushButton(self.FT3)
        self.pushButton_5.setGeometry(QtCore.QRect(170, 460, 121, 23))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.FinalDataPlot = MatplotlibWidget(self.FT3)
        self.FinalDataPlot.setGeometry(QtCore.QRect(290, 10, 481, 361))
        self.FinalDataPlot.setObjectName(_fromUtf8("FinalDataPlot"))
        self.PhaseBox = QtGui.QLineEdit(self.FT3)
        self.PhaseBox.setGeometry(QtCore.QRect(190, 90, 61, 20))
        self.PhaseBox.setObjectName(_fromUtf8("PhaseBox"))
        self.tZeroBox = QtGui.QLineEdit(self.FT3)
        self.tZeroBox.setGeometry(QtCore.QRect(190, 130, 61, 20))
        self.tZeroBox.setObjectName(_fromUtf8("tZeroBox"))
        self.Separate_Plot = QtGui.QPushButton(self.FT3)
        self.Separate_Plot.setGeometry(QtCore.QRect(630, 380, 131, 31))
        self.Separate_Plot.setObjectName(_fromUtf8("Separate_Plot"))
        self.nrPlot = QtGui.QPushButton(self.FT3)
        self.nrPlot.setGeometry(QtCore.QRect(810, 70, 121, 23))
        self.nrPlot.setObjectName(_fromUtf8("nrPlot"))
        self.rPlot = QtGui.QPushButton(self.FT3)
        self.rPlot.setGeometry(QtCore.QRect(810, 100, 121, 23))
        self.rPlot.setObjectName(_fromUtf8("rPlot"))
        self.ppSpinBox = QtGui.QSpinBox(self.FT3)
        self.ppSpinBox.setGeometry(QtCore.QRect(380, 460, 42, 22))
        self.ppSpinBox.setObjectName(_fromUtf8("ppSpinBox"))
        self.label_7 = QtGui.QLabel(self.FT3)
        self.label_7.setGeometry(QtCore.QRect(330, 460, 51, 16))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.label_8 = QtGui.QLabel(self.FT3)
        self.label_8.setGeometry(QtCore.QRect(470, 460, 46, 13))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.ppTimeLabel = QtGui.QLabel(self.FT3)
        self.ppTimeLabel.setGeometry(QtCore.QRect(530, 460, 46, 13))
        self.ppTimeLabel.setText(_fromUtf8(""))
        self.ppTimeLabel.setObjectName(_fromUtf8("ppTimeLabel"))
        self.tabWidget.addTab(self.FT3, _fromUtf8(""))
        self.tab_4 = QtGui.QWidget()
        self.tab_4.setObjectName(_fromUtf8("tab_4"))
        self.tabWidget.addTab(self.tab_4, _fromUtf8(""))
        self.MC_Offset = QtGui.QLineEdit(self.centralwidget)
        self.MC_Offset.setGeometry(QtCore.QRect(190, 50, 51, 20))
        self.MC_Offset.setObjectName(_fromUtf8("MC_Offset"))
        self.MC_Offset_label = QtGui.QLabel(self.centralwidget)
        self.MC_Offset_label.setGeometry(QtCore.QRect(120, 50, 61, 21))
        self.MC_Offset_label.setObjectName(_fromUtf8("MC_Offset_label"))
        self.dLambdaBox = QtGui.QLineEdit(self.centralwidget)
        self.dLambdaBox.setGeometry(QtCore.QRect(340, 50, 51, 20))
        self.dLambdaBox.setObjectName(_fromUtf8("dLambdaBox"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(280, 50, 71, 20))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 984, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuSave_Spectrum = QtGui.QMenu(self.menuFile)
        self.menuSave_Spectrum.setObjectName(_fromUtf8("menuSave_Spectrum"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_File = QtGui.QAction(MainWindow)
        self.actionLoad_File.setObjectName(_fromUtf8("actionLoad_File"))
        self.actionSave_Parameters = QtGui.QAction(MainWindow)
        self.actionSave_Parameters.setObjectName(_fromUtf8("actionSave_Parameters"))
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.actionFinal_Data = QtGui.QAction(MainWindow)
        self.actionFinal_Data.setObjectName(_fromUtf8("actionFinal_Data"))
        self.actionFT1_Data = QtGui.QAction(MainWindow)
        self.actionFT1_Data.setObjectName(_fromUtf8("actionFT1_Data"))
        self.actionFT2_Data = QtGui.QAction(MainWindow)
        self.actionFT2_Data.setObjectName(_fromUtf8("actionFT2_Data"))
        self.actionRaw_Data = QtGui.QAction(MainWindow)
        self.actionRaw_Data.setObjectName(_fromUtf8("actionRaw_Data"))
        self.actionLoad_Parameter_File = QtGui.QAction(MainWindow)
        self.actionLoad_Parameter_File.setObjectName(_fromUtf8("actionLoad_Parameter_File"))
        self.menuSave_Spectrum.addAction(self.actionFinal_Data)
        self.menuSave_Spectrum.addAction(self.actionFT1_Data)
        self.menuSave_Spectrum.addAction(self.actionFT2_Data)
        self.menuSave_Spectrum.addAction(self.actionRaw_Data)
        self.menuFile.addAction(self.actionLoad_File)
        self.menuFile.addAction(self.actionLoad_Parameter_File)
        self.menuFile.addAction(self.actionSave_Parameters)
        self.menuFile.addAction(self.menuSave_Spectrum.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.FT1Button, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.doFT1)
        QtCore.QObject.connect(self.FT2Button, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.doFT2)
        QtCore.QObject.connect(self.FT3Button, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.doFT3)
        QtCore.QObject.connect(self.PhaseSlider, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), MainWindow.updatePhaseBox)
        QtCore.QObject.connect(self.PhaseBox, QtCore.SIGNAL(_fromUtf8("editingFinished()")), MainWindow.updatePhase)
        QtCore.QObject.connect(self.MC_Offset, QtCore.SIGNAL(_fromUtf8("editingFinished()")), MainWindow.set_MC_Offset)
        QtCore.QObject.connect(self.nrPlot, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.PlotNR)
        QtCore.QObject.connect(self.rPlot, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.PlotR)
        QtCore.QObject.connect(self.Separate_Plot, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.PlotFinalSeparate)
        QtCore.QObject.connect(self.dLambdaBox, QtCore.SIGNAL(_fromUtf8("editingFinished()")), MainWindow._update_dLambda)
        QtCore.QObject.connect(self.tZeroBox, QtCore.SIGNAL(_fromUtf8("editingFinished()")), MainWindow.Change_tZero)
        QtCore.QObject.connect(self.tZeroSlider, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), MainWindow._update_tZero)
        QtCore.QObject.connect(self.ppSpinBox, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), MainWindow._update_ppIndex)
        QtCore.QObject.connect(self.pushButton_5, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.loadPP)
        QtCore.QObject.connect(self.pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow._plotRtau)
        QtCore.QObject.connect(self.pushButton_2, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow._plotNRtau)
        QtCore.QObject.connect(self.pushButton_3, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow._plotRtauAbs)
        QtCore.QObject.connect(self.pushButton_4, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow._plotNRtauAbs)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Paul\'s 2D Processing Code", None))
        self.RawDataPlotButton.setText(_translate("MainWindow", "Plot Raw Data", None))
        self.FileIndicator.setText(_translate("MainWindow", "No File Loaded", None))
        self.FT1Button.setText(_translate("MainWindow", "Do FT1", None))
        self.FT1PlotButton.setText(_translate("MainWindow", "Plot FT1", None))
        self.label_2.setText(_translate("MainWindow", "Time Zero:", None))
        self.RelativePhaseBox.setText(_translate("MainWindow", "0", None))
        self.label.setText(_translate("MainWindow", "Relative Phase", None))
        self.FT1tZero.setText(_translate("MainWindow", "0", None))
        self.label_4.setText(_translate("MainWindow", "Nyquist Multiplier:", None))
        self.TransformTime.setText(_translate("MainWindow", "6000", None))
        self.label_5.setText(_translate("MainWindow", "Transform Time:", None))
        self.label_6.setText(_translate("MainWindow", "fs", None))
        self.phaseGuess.setText(_translate("MainWindow", "0", None))
        self.label_9.setText(_translate("MainWindow", "Guess:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FT1), _translate("MainWindow", "FT1", None))
        self.FT2Button.setText(_translate("MainWindow", "FT2 Button", None))
        self.ft2Indicator.setText(_translate("MainWindow", "You haven\'t done anything yet!", None))
        self.pushButton.setText(_translate("MainWindow", "Plot Rephasing", None))
        self.pushButton_2.setText(_translate("MainWindow", "Plot NonRephasing", None))
        self.pushButton_3.setText(_translate("MainWindow", "abs. rephasing", None))
        self.pushButton_4.setText(_translate("MainWindow", "abs. NR", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FT2), _translate("MainWindow", "FT2", None))
        self.FT3Button.setText(_translate("MainWindow", "Do FT3", None))
        self.pushButton_5.setText(_translate("MainWindow", "Load a Pump Probe", None))
        self.PhaseBox.setText(_translate("MainWindow", "0", None))
        self.Separate_Plot.setText(_translate("MainWindow", "Plot in separate Window", None))
        self.nrPlot.setText(_translate("MainWindow", "Plot Non-Rephasing", None))
        self.rPlot.setText(_translate("MainWindow", "Plot Rephasing", None))
        self.label_7.setText(_translate("MainWindow", "ppIndex:", None))
        self.label_8.setText(_translate("MainWindow", "ppTime:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FT3), _translate("MainWindow", "FT3", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Post Processing", None))
        self.MC_Offset.setText(_translate("MainWindow", "19", None))
        self.MC_Offset_label.setText(_translate("MainWindow", "Mon. Offset:", None))
        self.dLambdaBox.setText(_translate("MainWindow", "9.45", None))
        self.label_3.setText(_translate("MainWindow", "dLambda:", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuSave_Spectrum.setTitle(_translate("MainWindow", "Save Spectrum", None))
        self.actionLoad_File.setText(_translate("MainWindow", "Load File", None))
        self.actionSave_Parameters.setText(_translate("MainWindow", "Save Parameters", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q", None))
        self.actionFinal_Data.setText(_translate("MainWindow", "Final Data", None))
        self.actionFT1_Data.setText(_translate("MainWindow", "FT1 Data", None))
        self.actionFT2_Data.setText(_translate("MainWindow", "FT2 Data", None))
        self.actionRaw_Data.setText(_translate("MainWindow", "Raw Data", None))
        self.actionLoad_Parameter_File.setText(_translate("MainWindow", "Load Parameter File", None))

from matplotlibwidget import MatplotlibWidget
