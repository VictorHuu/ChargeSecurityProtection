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
#from PyQt5.QtWebEngineWidgets import QWebEngineView


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
    def displayCsv(self, df):
        try:
            # Use pandas to read the CSV file
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
        directory=QFileDialog.getExistingDirectory(self,"Select directory that contains CSV files")
        #filename, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "./", "CSV Files (*.csv)")
        if directory:
            mdf=self.merge_data(directory)
            cdf=self.cleanse_data(mdf)
            self.displayCsv(cdf)
    def cleanse_data(self,merged_df):
        # 数据偏移纠正
        merged_df["bmscurrent"] = merged_df["bmscurrent"] * (-1)
        merged_df["maxcurrent"] = merged_df["maxcurrent"] * (-1)
        # 数据去重
        merged_df = merged_df.drop_duplicates()
        # 按时间和站名排序
        merged_df['samptime'] = pd.to_datetime(merged_df['samptime'], unit='ms')
        merged_df['fessamptime'] = pd.to_datetime(merged_df['fessamptime'], unit='ms')
        merged_df = merged_df.sort_values(by=['station_name', 'stake_name', 'samptime'])
        merged_df = merged_df.reset_index(drop=True)
        return merged_df
    def merge_data(self,folder_path):
        df_list = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                # 如果你的CSV文件没有表头，你可以手动设置一个
                header_list = [
                    "station_code",
                    "station_name",
                    "stake_name",
                    "tag",
                    "datatype",
                    "chargingpilenum",
                    "chargingpileinterfacenum",
                    "conncetflag",
                    "charginggunflag",
                    "electroniclockstatus",
                    "dcoutputcontactorstatus",
                    "workstatus",
                    "outputvoltage",
                    "outputcurrent",
                    "demandvoltage",
                    "demandcurrent",
                    "chargemode",
                    "soc",
                    "batterytype",
                    "minimumbatterytemperature",
                    "maximumbatterytemperature",
                    "cumulativechargetime",
                    "estimatedfullchargetime",
                    "maximumbatteryvoltage",
                    "minimumbatteryvoltage",
                    "totalactivepower",
                    "electricityconsumptionamount",
                    "servicefee",
                    "chargingtype",
                    "useridentification",
                    "tariffmodelnumber",
                    "servicechargemodelnumber",
                    "batch",
                    "status",
                    "samptime",
                    "msgnum",
                    "bmsvoltage",
                    "bmscurrent",
                    "guntemperature1",
                    "guntemperature2",
                    "guntemperature3",
                    "guntemperature4",
                    "maxtemperature",
                    "maxbmsvoltage",
                    "maxmonomervoltage",
                    "maxcurrent",
                    "ratedtotalvoltage",
                    "currentvoltage",
                    "ratedcapacity",
                    "nominalenergy",
                    "stateofcharge",
                    "powertype",
                    "powerparameter",
                    "resultsfeedback",
                    "fessamptime",
                    "dt"]
                df = pd.read_csv(file_path, header=None, names=header_list)
                df_list.append(df)
        merged_df = pd.concat(df_list, ignore_index=True)
        return merged_df
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




