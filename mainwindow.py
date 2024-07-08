import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDate
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QFileDialog, QTableView, QVBoxLayout, QDialog, QCalendarWidget, QPushButton
from PyQt5 import *
from v4 import Ui_Dialog
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QVBoxLayout, QWidget
from PyQt5.QtCore import QUrl
import os
import folium
from PyQt5.QtWebEngineWidgets import QWebEngineView


class MainWindow(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.stackedWidget.setCurrentIndex(0)

        self.initCsvTableView()

        self.pushButton_10.clicked.connect(self.packageData)




    def initCsvTableView(self):
        # Assume Widget1 is a QWidget
        self.csvTableView = QTableView()
        self.csvTableLayout = QVBoxLayout(self.tableWidget)  # Add to Widget1's layout
        self.csvTableLayout.addWidget(self.csvTableView)

    def displayCsvInTable(self, filename):
        try:
            # Use pandas to read the CSV file
            df = pd.read_csv(filename)
            model = QStandardItemModel(df.shape[0], df.shape[1])
            model.setHorizontalHeaderLabels(df.columns)

            # Fill data
            for row in df.itertuples():
                for col_index, value in enumerate(row[1:]):
                    item = QStandardItem(str(value))
                    model.setItem(row.Index, col_index, item)

            # Set up the table view
            self.csvTableView.setModel(model)
            self.csvTableView.resizeColumnsToContents()
            self.csvTableView.setAlternatingRowColors(True)
            self.csvTableView.horizontalHeader().setStretchLastSection(True)
        except Exception as e:
            print("Error reading file:", e)



    def click_index_0(self):
        self.stackedWidget.setCurrentIndex(0)
    def click_index_1(self):
        self.stackedWidget.setCurrentIndex(1)

    def click_index_2(self):
        self.stackedWidget.setCurrentIndex(2)

    def click_index_3(self):
        self.stackedWidget.setCurrentIndex(3)

    def click_index_4(self):
        self.stackedWidget.setCurrentIndex(4)

    def addcsv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "./", "CSV Files (*.csv)")
        if filename:
            self.displayCsvInTable(filename)


    def packageData(self):
        # 读取控件内容并封装成字典
        data = {
            'station_id': self.lineEdit.text(),
            'pile_id': self.lineEdit_1.text(),
            'fault_type': self.lineEdit_2.text(),
            'model_id': self.lineEdit_3.text(),
            'start_time': self.dateEdit.date().toString('yyyy-MM-dd'),
            'end_time': self.dateEdit_2.date().toString('yyyy-MM-dd')
        }
        self.queryFaults(data)

    def queryFaults(self, data):
        # 在这里实现您的查询逻辑
        # 例如，打印数据或发送到数据库
        print(data)




