import sys
import os
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDate, QUrl
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QFileDialog, QTableView, QVBoxLayout, QDialog, QPushButton, QLineEdit, QFormLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from v5 import Ui_Dialog
import folium


class MainWindow(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.stackedWidget.setCurrentIndex(0)

        self.initCsvTableView()
        self.initMapView()

        # self.pushButton_7.clicked.connect(self.packageData)
        self.pushButton_8.clicked.connect(self.load_map_data)
        self.pushButton_9.clicked.connect(self.add_marker)

    def initCsvTableView(self):
        self.csvTableView = QTableView()
        self.csvTableLayout = QVBoxLayout(self.tableWidget)  # Add to Widget1's layout
        self.csvTableLayout.addWidget(self.csvTableView)

    def displayCsv(self, df):
        try:
            model = QStandardItemModel(df.shape[0], df.shape[1])
            model.setHorizontalHeaderLabels(df.columns)
            for row in df.itertuples():
                for col_index, value in enumerate(row[1:]):
                    item = QStandardItem(str(value))
                    model.setItem(row.Index, col_index, item)
            self.csvTableView.setModel(model)
            self.csvTableView.resizeColumnsToContents()
            self.csvTableView.setAlternatingRowColors(True)
            self.csvTableView.horizontalHeader().setStretchLastSection(True)
        except Exception as e:
            print("Error reading file:", e)

    def displayCsvInTable(self, filename):
        try:
            df = pd.read_csv(filename)
            model = QStandardItemModel(df.shape[0], df.shape[1])
            model.setHorizontalHeaderLabels(df.columns)
            for row in df.itertuples():
                for col_index, value in enumerate(row[1:]):
                    item = QStandardItem(str(value))
                    model.setItem(row.Index, col_index, item)
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
        directory = QFileDialog.getExistingDirectory(self, "Select directory that contains CSV files")
        if directory:
            mdf = self.merge_data(directory)
            cdf = self.cleanse_data(mdf)
            self.displayCsv(cdf)

    def cleanse_data(self, merged_df):
        merged_df["bmscurrent"] = merged_df["bmscurrent"] * (-1)
        merged_df["maxcurrent"] = merged_df["maxcurrent"] * (-1)
        merged_df = merged_df.drop_duplicates()
        merged_df['samptime'] = pd.to_datetime(merged_df['samptime'], unit='ms')
        merged_df['fessamptime'] = pd.to_datetime(merged_df['fessamptime'], unit='ms')
        merged_df = merged_df.sort_values(by=['station_name', 'stake_name', 'samptime'])
        merged_df = merged_df.reset_index(drop=True)
        return merged_df

    def merge_data(self, folder_path):
        df_list = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                header_list = [
                    "station_code", "station_name", "stake_name", "tag", "datatype", "chargingpilenum",
                    "chargingpileinterfacenum", "conncetflag", "charginggunflag", "electroniclockstatus",
                    "dcoutputcontactorstatus", "workstatus", "outputvoltage", "outputcurrent", "demandvoltage",
                    "demandcurrent", "chargemode", "soc", "batterytype", "minimumbatterytemperature",
                    "maximumbatterytemperature", "cumulativechargetime", "estimatedfullchargetime",
                    "maximumbatteryvoltage", "minimumbatteryvoltage", "totalactivepower",
                    "electricityconsumptionamount", "servicefee", "chargingtype", "useridentification",
                    "tariffmodelnumber", "servicechargemodelnumber", "batch", "status", "samptime", "msgnum",
                    "bmsvoltage", "bmscurrent", "guntemperature1", "guntemperature2", "guntemperature3",
                    "guntemperature4", "maxtemperature", "maxbmsvoltage", "maxmonomervoltage", "maxcurrent",
                    "ratedtotalvoltage", "currentvoltage", "ratedcapacity", "nominalenergy", "stateofcharge",
                    "powertype", "powerparameter", "resultsfeedback", "fessamptime", "dt"
                ]
                df = pd.read_csv(file_path, header=None, names=header_list)
                df_list.append(df)
        merged_df = pd.concat(df_list, ignore_index=True)
        return merged_df

    def packageData(self):
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
        print(data)

    def initMapView(self):
        self.map_view = QWebEngineView()
        self.map_layout = QVBoxLayout(self.widget)  # Use self.widget from your UI
        self.map_layout.addWidget(self.map_view)
        self.map_data = None
        self.display_map()  # Show initial map

    def load_map_data(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择充电站数据文件", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.map_data = pd.read_csv(file_path)
                self.display_map()
                self.save_map_to_file()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", f"无法加载文件: {e}")

    def display_map(self):
        offline_map_path = 'E:/firstgold/mapdemo/tiles/hang/{z}/{x}/{y}.png'
        m = folium.Map(location=[30.2741, 120.1551], zoom_start=12, tiles=None)
        folium.TileLayer(
            tiles=offline_map_path,
            attr='My Offline Map',
            name='Local Tiles',
            overlay=False,
            control=True
        ).add_to(m)
        if self.map_data is not None:
            def get_color(rating):
                if rating > 8:
                    return 'green'
                elif rating > 6:
                    return 'orange'
                elif rating > 4:
                    return 'red'
                else:
                    return 'darkred'

            for index, row in self.map_data.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(f"{row['name']}<br>{row['description']}", max_width=300),
                    icon=folium.Icon(color=get_color(row['rating']))
                ).add_to(m)
        m.save('offline_map.html')
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath('offline_map.html')))

    def add_marker(self):
        dialog = MarkerDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name, description, latitude, longitude, rating = dialog.get_data()
            new_marker = pd.DataFrame([{
                'name': name,
                'description': description,
                'latitude': float(latitude),
                'longitude': float(longitude),
                'rating': int(rating)
            }])
            if self.map_data is None:
                self.map_data = new_marker
            else:
                self.map_data = pd.concat([self.map_data, new_marker], ignore_index=True)
            self.display_map()
            self.save_map_to_file()

    def save_map_to_file(self):
        offline_map_path = 'tiles/{z}/{x}/{y}.png'
        m = folium.Map(location=[30.2741, 120.1551], zoom_start=12, tiles=None)
        folium.TileLayer(
            tiles=offline_map_path,
            attr='My Offline Map',
            name='Local Tiles',
            overlay=False,
            control=True
        ).add_to(m)
        if self.map_data is not None:
            def get_color(rating):
                if rating > 8:
                    return 'green'
                elif rating > 6:
                    return 'orange'
                elif rating > 4:
                    return 'red'
                else:
                    return 'darkred'

            for index, row in self.map_data.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(f"{row['name']}<br>{row['description']}", max_width=300),
                    icon=folium.Icon(color=get_color(row['rating']))
                ).add_to(m)
        m.save('offline_map.html')


class MarkerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加标注")

        self.name_input = QLineEdit()
        self.description_input = QLineEdit()
        self.latitude_input = QLineEdit()
        self.longitude_input = QLineEdit()
        self.rating_input = QLineEdit()

        form_layout = QFormLayout()
        form_layout.addRow("名称:", self.name_input)
        form_layout.addRow("描述:", self.description_input)
        form_layout.addRow("纬度:", self.latitude_input)
        form_layout.addRow("经度:", self.longitude_input)
        form_layout.addRow("评分:", self.rating_input)

        self.submit_button = QPushButton("提交")
        self.submit_button.clicked.connect(self.accept)
        form_layout.addRow(self.submit_button)

        self.setLayout(form_layout)

    def get_data(self):
        return (
            self.name_input.text(), self.description_input.text(), self.latitude_input.text(),
            self.longitude_input.text(), self.rating_input.text()
        )


