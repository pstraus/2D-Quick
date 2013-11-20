# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test1.ui'
#
# Created: Tue Jul 30 14:01:45 2013
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
        MainWindow.resize(800, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 50, 81, 21))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 111, 31))
        self.label.setObjectName(_fromUtf8("label"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(190, 20, 351, 321))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.FT1 = QtGui.QWidget()
        self.FT1.setObjectName(_fromUtf8("FT1"))
        self.lineEdit = QtGui.QLineEdit(self.FT1)
        self.lineEdit.setGeometry(QtCore.QRect(240, 10, 71, 21))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.pushButton_2 = QtGui.QPushButton(self.FT1)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_3 = QtGui.QPushButton(self.FT1)
        self.pushButton_3.setGeometry(QtCore.QRect(260, 260, 75, 23))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.label_2 = QtGui.QLabel(self.FT1)
        self.label_2.setGeometry(QtCore.QRect(180, 10, 51, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.tabWidget.addTab(self.FT1, _fromUtf8(""))
        self.FT2 = QtGui.QWidget()
        self.FT2.setObjectName(_fromUtf8("FT2"))
        self.tabWidget.addTab(self.FT2, _fromUtf8(""))
        self.FT3 = QtGui.QWidget()
        self.FT3.setObjectName(_fromUtf8("FT3"))
        self.tabWidget.addTab(self.FT3, _fromUtf8(""))
        self.tab_4 = QtGui.QWidget()
        self.tab_4.setObjectName(_fromUtf8("tab_4"))
        self.tabWidget.addTab(self.tab_4, _fromUtf8(""))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_File = QtGui.QAction(MainWindow)
        self.actionLoad_File.setObjectName(_fromUtf8("actionLoad_File"))
        self.actionSave_Parameters = QtGui.QAction(MainWindow)
        self.actionSave_Parameters.setObjectName(_fromUtf8("actionSave_Parameters"))
        self.actionSave_Spectrum = QtGui.QAction(MainWindow)
        self.actionSave_Spectrum.setObjectName(_fromUtf8("actionSave_Spectrum"))
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.menuFile.addAction(self.actionLoad_File)
        self.menuFile.addAction(self.actionSave_Parameters)
        self.menuFile.addAction(self.actionSave_Spectrum)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.pushButton_2, QtCore.SIGNAL(_fromUtf8("clicked()")), MainWindow.doFT1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pushButton.setText(_translate("MainWindow", "Plot Raw Data", None))
        self.label.setText(_translate("MainWindow", "No File Loaded", None))
        self.lineEdit.setText(_translate("MainWindow", "0", None))
        self.pushButton_2.setText(_translate("MainWindow", "Do FT1", None))
        self.pushButton_3.setText(_translate("MainWindow", "Plot FT1", None))
        self.label_2.setText(_translate("MainWindow", "Time Zero:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FT1), _translate("MainWindow", "FT1", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FT2), _translate("MainWindow", "FT2", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FT3), _translate("MainWindow", "FT3", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Post Processing", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionLoad_File.setText(_translate("MainWindow", "Load File", None))
        self.actionSave_Parameters.setText(_translate("MainWindow", "Save Parameters", None))
        self.actionSave_Spectrum.setText(_translate("MainWindow", "Save Spectrum", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q", None))

