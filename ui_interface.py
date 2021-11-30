# Form implementation generated from reading ui file 'ui_main.ui'
#
# Created by: PyQt6 UI code generator 6.2.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 829)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.header_frame = QtWidgets.QFrame(self.centralwidget)
        self.header_frame.setStyleSheet("background-color: rgb(23, 124, 193);")
        self.header_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.header_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.header_frame.setObjectName("header_frame")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.header_frame)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_header = QtWidgets.QLabel(self.header_frame)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_header.setFont(font)
        self.label_header.setStyleSheet("color: rgb(245, 247, 250);\n"
"font-weight: bold;\n"
"")
        self.label_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_header.setObjectName("label_header")
        self.gridLayout_6.addWidget(self.label_header, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.header_frame, 0, 0, 1, 1)
        self.main_frame = QtWidgets.QFrame(self.centralwidget)
        self.main_frame.setStyleSheet("background-color: #12232E;\n"
"color: white;")
        self.main_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.main_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.main_frame.setObjectName("main_frame")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.main_frame)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.output_frame = QtWidgets.QFrame(self.main_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_frame.sizePolicy().hasHeightForWidth())
        self.output_frame.setSizePolicy(sizePolicy)
        self.output_frame.setMinimumSize(QtCore.QSize(640, 0))
        self.output_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.output_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.output_frame.setObjectName("output_frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.output_frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.output_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.verticalLayout_5.addWidget(self.output_frame)
        self.prediction_frame = QtWidgets.QFrame(self.main_frame)
        self.prediction_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.prediction_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.prediction_frame.setObjectName("prediction_frame")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.prediction_frame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.prediction_table = QtWidgets.QTableWidget(self.prediction_frame)
        self.prediction_table.setAutoFillBackground(False)
        self.prediction_table.setStyleSheet("::item{\n"
"background-color: #1F3647;\n"
"}")
        self.prediction_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.prediction_table.setAutoScroll(True)
        self.prediction_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.prediction_table.setAlternatingRowColors(False)
        self.prediction_table.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.prediction_table.setShowGrid(True)
        self.prediction_table.setWordWrap(False)
        self.prediction_table.setCornerButtonEnabled(False)
        self.prediction_table.setRowCount(6)
        self.prediction_table.setObjectName("prediction_table")
        self.prediction_table.setColumnCount(3)
        item = QtWidgets.QTableWidgetItem()
        self.prediction_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.prediction_table.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.prediction_table.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.prediction_table.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.prediction_table.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.prediction_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.prediction_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.prediction_table.setHorizontalHeaderItem(2, item)
        self.prediction_table.horizontalHeader().setVisible(True)
        self.prediction_table.horizontalHeader().setCascadingSectionResizes(False)
        self.prediction_table.horizontalHeader().setHighlightSections(True)
        self.prediction_table.horizontalHeader().setMinimumSectionSize(27)
        self.prediction_table.horizontalHeader().setStretchLastSection(True)
        self.prediction_table.verticalHeader().setVisible(False)
        self.prediction_table.verticalHeader().setMinimumSectionSize(27)
        self.gridLayout_3.addWidget(self.prediction_table, 0, 0, 1, 1)
        self.verticalLayout_5.addWidget(self.prediction_frame)
        self.verticalLayout_5.setStretch(0, 4)
        self.gridLayout_4.addLayout(self.verticalLayout_5, 0, 0, 1, 1)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setSpacing(7)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.form_frame = QtWidgets.QFrame(self.main_frame)
        self.form_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.form_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.form_frame.setObjectName("form_frame")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.form_frame)
        self.gridLayout_8.setVerticalSpacing(15)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.detect_button = QtWidgets.QPushButton(self.form_frame)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.detect_button.setFont(font)
        self.detect_button.setStyleSheet("background-color: #177CC1;")
        self.detect_button.setObjectName("detect_button")
        self.horizontalLayout.addWidget(self.detect_button)
        self.stop_button = QtWidgets.QPushButton(self.form_frame)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.stop_button.setFont(font)
        self.stop_button.setStyleSheet("background-color: rgb(38,66,86);")
        self.stop_button.setObjectName("stop_button")
        self.horizontalLayout.addWidget(self.stop_button)
        self.gridLayout_8.addLayout(self.horizontalLayout, 4, 0, 1, 1)
        self.import_layout = QtWidgets.QVBoxLayout()
        self.import_layout.setSpacing(5)
        self.import_layout.setObjectName("import_layout")
        self.label_import = QtWidgets.QLabel(self.form_frame)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_import.setFont(font)
        self.label_import.setObjectName("label_import")
        self.import_layout.addWidget(self.label_import)
        self.button_import = QtWidgets.QLineEdit(self.form_frame)
        self.button_import.setStyleSheet("background-color: #1F3647;")
        self.button_import.setReadOnly(False)
        self.button_import.setPlaceholderText("")
        self.button_import.setClearButtonEnabled(True)
        self.button_import.setObjectName("button_import")
        self.import_layout.addWidget(self.button_import)
        self.gridLayout_8.addLayout(self.import_layout, 1, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_mode = QtWidgets.QLabel(self.form_frame)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_mode.setFont(font)
        self.label_mode.setObjectName("label_mode")
        self.verticalLayout.addWidget(self.label_mode)
        self.combo_mode = QtWidgets.QComboBox(self.form_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.combo_mode.setFont(font)
        self.combo_mode.setObjectName("combo_mode")
        self.combo_mode.addItem("")
        self.combo_mode.addItem("")
        self.combo_mode.addItem("")
        self.verticalLayout.addWidget(self.combo_mode)
        self.gridLayout_8.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(5)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.form_frame)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.checkbox_frame = QtWidgets.QFrame(self.form_frame)
        self.checkbox_frame.setAutoFillBackground(False)
        self.checkbox_frame.setStyleSheet("background-color: #1F3647;")
        self.checkbox_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.checkbox_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.checkbox_frame.setObjectName("checkbox_frame")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.checkbox_frame)
        self.gridLayout_7.setSpacing(6)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.class_1 = QtWidgets.QCheckBox(self.checkbox_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.class_1.setFont(font)
        self.class_1.setStyleSheet("::indicator {\n"
"    border: 1px solid grey;\n"
"    background-color: #fff;\n"
"}\n"
"\n"
"::indicator:checked {\n"
"    image: url(\"assets/checked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}\n"
"::indicator:unchecked {\n"
"    image: url(\"assets/unchecked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}")
        self.class_1.setChecked(True)
        self.class_1.setObjectName("class_1")
        self.gridLayout_7.addWidget(self.class_1, 1, 0, 1, 1)
        self.class_4 = QtWidgets.QCheckBox(self.checkbox_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.class_4.setFont(font)
        self.class_4.setStyleSheet("::indicator {\n"
"    border: 1px solid grey;\n"
"    background-color: #fff;\n"
"}\n"
"\n"
"::indicator:checked {\n"
"    image: url(\"assets/checked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}\n"
"::indicator:unchecked {\n"
"    image: url(\"assets/unchecked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}")
        self.class_4.setChecked(True)
        self.class_4.setObjectName("class_4")
        self.gridLayout_7.addWidget(self.class_4, 4, 0, 1, 1)
        self.class_2 = QtWidgets.QCheckBox(self.checkbox_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.class_2.setFont(font)
        self.class_2.setStyleSheet("::indicator {\n"
"    border: 1px solid grey;\n"
"    background-color: #fff;\n"
"}\n"
"\n"
"::indicator:checked {\n"
"    image: url(\"assets/checked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}\n"
"::indicator:unchecked {\n"
"    image: url(\"assets/unchecked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}")
        self.class_2.setChecked(True)
        self.class_2.setObjectName("class_2")
        self.gridLayout_7.addWidget(self.class_2, 2, 0, 1, 1)
        self.class_0 = QtWidgets.QCheckBox(self.checkbox_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.class_0.setFont(font)
        self.class_0.setStyleSheet("::indicator {\n"
"    border: 1px solid grey;\n"
"    background-color: #fff;\n"
"}\n"
"\n"
"::indicator:checked {\n"
"    image: url(\"assets/checked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}\n"
"::indicator:unchecked {\n"
"    image: url(\"assets/unchecked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}")
        self.class_0.setChecked(True)
        self.class_0.setObjectName("class_0")
        self.gridLayout_7.addWidget(self.class_0, 0, 0, 1, 1)
        self.class_3 = QtWidgets.QCheckBox(self.checkbox_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.class_3.setFont(font)
        self.class_3.setStyleSheet("::indicator {\n"
"    border: 1px solid grey;\n"
"    background-color: #fff;\n"
"}\n"
"\n"
"::indicator:checked {\n"
"    image: url(\"assets/checked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}\n"
"::indicator:unchecked {\n"
"    image: url(\"assets/unchecked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}")
        self.class_3.setChecked(True)
        self.class_3.setObjectName("class_3")
        self.gridLayout_7.addWidget(self.class_3, 3, 0, 1, 1)
        self.class_5 = QtWidgets.QCheckBox(self.checkbox_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.class_5.setFont(font)
        self.class_5.setStyleSheet("::indicator {\n"
"    border: 1px solid grey;\n"
"    background-color: #fff;\n"
"}\n"
"\n"
"::indicator:checked {\n"
"    image: url(\"assets/checked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}\n"
"::indicator:unchecked {\n"
"    image: url(\"assets/unchecked.png\");\n"
"    height: 1em;\n"
"    width: 1em;\n"
"}")
        self.class_5.setChecked(True)
        self.class_5.setObjectName("class_5")
        self.gridLayout_7.addWidget(self.class_5, 5, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.checkbox_frame)
        self.gridLayout_8.addLayout(self.verticalLayout_4, 3, 0, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_detection = QtWidgets.QLabel(self.form_frame)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_detection.setFont(font)
        self.label_detection.setObjectName("label_detection")
        self.verticalLayout_3.addWidget(self.label_detection)
        self.button_detection = QtWidgets.QLineEdit(self.form_frame)
        self.button_detection.setStyleSheet("background-color: #1F3647;")
        self.button_detection.setReadOnly(True)
        self.button_detection.setPlaceholderText("")
        self.button_detection.setClearButtonEnabled(True)
        self.button_detection.setObjectName("button_detection")
        self.verticalLayout_3.addWidget(self.button_detection)
        self.gridLayout_8.addLayout(self.verticalLayout_3, 2, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_8.addItem(spacerItem, 5, 0, 1, 1)
        self.verticalLayout_7.addWidget(self.form_frame)
        self.gridLayout_4.addLayout(self.verticalLayout_7, 0, 1, 1, 1)
        self.gridLayout_4.setColumnStretch(0, 2)
        self.gridLayout.addWidget(self.main_frame, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        self.menuMain = QtWidgets.QMenu(self.menubar)
        self.menuMain.setObjectName("menuMain")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuMain.addAction(self.actionExit)
        self.menubar.addAction(self.menuMain.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_header.setText(_translate("MainWindow", "DISTRIBUTION LINE DETECTION"))
        item = self.prediction_table.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "New Row"))
        item = self.prediction_table.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "New Row"))
        item = self.prediction_table.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "New Row"))
        item = self.prediction_table.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "New Row"))
        item = self.prediction_table.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "New Row"))
        item = self.prediction_table.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ID"))
        item = self.prediction_table.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Name"))
        item = self.prediction_table.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Occurence"))
        self.detect_button.setText(_translate("MainWindow", "Detect"))
        self.stop_button.setText(_translate("MainWindow", "Save"))
        self.label_import.setText(_translate("MainWindow", "Import"))
        self.label_mode.setText(_translate("MainWindow", "Mode"))
        self.combo_mode.setItemText(0, _translate("MainWindow", "Camera"))
        self.combo_mode.setItemText(1, _translate("MainWindow", "Image"))
        self.combo_mode.setItemText(2, _translate("MainWindow", "Video"))
        self.label_2.setText(_translate("MainWindow", "Components"))
        self.class_1.setText(_translate("MainWindow", "HV Bushing"))
        self.class_4.setText(_translate("MainWindow", "Radiator Fins"))
        self.class_2.setText(_translate("MainWindow", "LV Bushing"))
        self.class_0.setText(_translate("MainWindow", "Transformer Tank"))
        self.class_3.setText(_translate("MainWindow", "Arrester"))
        self.class_5.setText(_translate("MainWindow", "Output Fuse"))
        self.label_detection.setText(_translate("MainWindow", "Detection"))
        self.menuMain.setTitle(_translate("MainWindow", "Main"))
        self.actionExit.setText(_translate("MainWindow", "Quit"))
