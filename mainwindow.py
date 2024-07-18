import sys
import os
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDate, QUrl, Qt, QRectF, QRect, QSize, QDateTime
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QBrush, QColor, QFont, QPen
from PyQt5.QtWidgets import QFileDialog, QTableView, QVBoxLayout, QDialog, QPushButton, QLineEdit, QFormLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from v5 import Ui_Dialog
import folium
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice, QLineSeries, QDateTimeAxis, QValueAxis


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

        self.pushButton_10.clicked.connect(self.packageData)




    def initCsvTableView(self):
        self.csvTableView = QTableView()
        self.csvTableLayout = QVBoxLayout(self.tableWidget)  # Add to Widget1's layout
        self.csvTableLayout.addWidget(self.csvTableView)
        # 用于展示饼图（从pickle转化而来）
        self.pickleTableView = QChartView()
        self.pickleTableLayout = QVBoxLayout(self.widget_2_1)  # Add to widget_2_1(1st Pie Chart)'s layout
        self.pickleTableLayout.addWidget(self.pickleTableView)
        # Battery Pie Chart
        self.pickle_battery_TableView = QChartView()
        self.pickle_battery_TableLayout = QVBoxLayout(self.widget_2_2)  # Add to widget_2_2(2nd Pie Chart)'s layout
        self.pickle_battery_TableLayout.addWidget(self.pickle_battery_TableView)
        self.station_time_series_TableView = QChartView()
        self.station_time_series_TableLayout = QVBoxLayout(self.widget_2_3)  # Add to widget_2_3(3rd Pie Chart)'s layout
        self.station_time_series_TableLayout.addWidget(self.station_time_series_TableView)

        self.plot_dict = {}
    # 一个将df转化为model的工具类
    def dataframe_to_model(self, df):
        model = QStandardItemModel()

        # Set horizontal header labels
        model.setHorizontalHeaderLabels(df.columns)

        # Populate data
        for row_idx, row_data in df.iterrows():
            items = [QStandardItem(str(field)) for field in row_data]
            model.appendRow(items)

        return model
    def zoom_free_2_1(self):
        # zoom out by 50%
        current_size = self.widget_2_1.size()
        if self.widget_2_1_zoom_out:
            new_size = QSize(current_size.width() * 2, current_size.height() * 2)
        else:
            new_size = QSize(current_size.width() // 2, current_size.height() // 2)
        self.widget_2_1.resize(new_size)
        self.widget_2_1_zoom_out = not self.widget_2_1_zoom_out
    def zoom_free_2_2(self):
        # zoom out by 50%
        current_size = self.widget_2_2.size()
        if self.widget_2_2_zoom_out:
            new_size = QSize(current_size.width() * 2, current_size.height() * 2)
            self.widget_2_2.setGeometry(QRect(80, 60,new_size.width(),new_size.height()))
        else:
            new_size = QSize(current_size.width() // 2, current_size.height() // 2)
            self.widget_2_2.setGeometry(QRect(430, 70, new_size.width(), new_size.height()))
        self.widget_2_2.resize(new_size)
        self.widget_2_2_zoom_out = not self.widget_2_2_zoom_out
    def zoom_free_2_3(self):
        # zoom out by 50%
        current_size = self.widget_2_3.size()
        if self.widget_2_3_zoom_out:
            new_size = QSize(current_size.width() * 2, current_size.height() * 2)
            self.widget_2_3.setGeometry(QRect(80, 60,new_size.width(),new_size.height()))
        else:
            new_size = QSize(current_size.width() // 2, current_size.height() // 2)
            self.widget_2_3.setGeometry(QRect(430, 70, new_size.width(), new_size.height()))
        self.widget_2_3.resize(new_size)
        self.widget_2_3_zoom_out = not self.widget_2_3_zoom_out

    def on_slice_clicked(self):
        slice = self.sender()
        if slice.isLabelVisible():
            slice.setLabelVisible(False)
            # zoom out by 50%
            current_size = self.widget_2_3.size()
            if self.widget_2_3_zoom_out:
                pass
            else:
                new_size = QSize(current_size.width() // 2, current_size.height() // 2)
                self.widget_2_3.setGeometry(QRect(30, 300, new_size.width(), new_size.height()))
            self.widget_2_3.resize(new_size)
            self.widget_2_3_zoom_out = True
        else:
            slice.setLabelVisible(True)
            # zoom out by 50%
            current_size = self.widget_2_3.size()
            if self.widget_2_3_zoom_out:
                new_size = QSize(current_size.width() * 2, current_size.height() * 2)
                self.widget_2_3.setGeometry(QRect(30, 300, new_size.width(), new_size.height()))
            else:
                pass
            self.widget_2_3.resize(new_size)
            self.widget_2_3_zoom_out = False
            # Cache  needed
            if slice.label() in self.plot_dict:
                hourly_work=self.plot_dict[slice.label()]
            else:
                hourly_work = self.choose_a_station(self.data, slice.label())
                self.plot_dict[slice.label()] = hourly_work

            self.displayPlot(hourly_work, self.station_time_series_TableView, heading=slice.label()+"'s Charging Stack's working status")
        if slice.isExploded():
            slice.setExploded(False)
        else:
            slice.setExploded(True)
    # 用于展示饼图（Experimental）
    def displayPieChart(self,df,chart_view,heading,order):
        model=self.dataframe_to_model(df)
        series = QPieSeries()
        # Read data from the model
        for row in range(model.rowCount()):
            category = model.item(row, 0).text()
            value = float(model.item(row, 1).text())
            series.append(category, value)
        for slice in series.slices():
            slice.setLabelColor(QColor(Qt.red))
            slice.setLabelFont(QFont('Arial', 25))
            slice.hovered.connect(self.on_slice_clicked)
            # connect a function with a parameter
            if order == 1:
                slice.clicked.connect(self.zoom_free_2_1)
            elif order ==2:
                slice.clicked.connect(self.zoom_free_2_2)
        chart = QChart()
        chart.addSeries(series)
        chart.setTitleBrush(QBrush(Qt.black))
        chart.setTitleFont(QFont('Impact', 500, QFont.Bold))
        chart.setTitle(heading)
        chart.setBackgroundPen(QPen(QColor('black'), 5))
        # 800*600 is so small,I want a bigger one!
        #chart.setPlotArea(QRectF(-100, 20, 600, 400))
        # The text along the legends can wrap around
        chart.legend().setBackgroundVisible(True)
        chart.legend().setBorderColor(QColor(Qt.darkGreen))
        chart.legend().setMaximumWidth(400)
        chart.legend().setBrush(QBrush(QColor(128, 255, 255, 255)))
        chart.legend().setLabelBrush(QBrush(Qt.red))
        chart.legend().setAlignment(Qt.AlignRight)
        chart.legend().setFont(QFont('Courier', 25))
        chart.legend().setLabelColor(Qt.darkRed)
        chart.setTheme(QChart.ChartThemeBrownSand)
        ## User can drag
        chart.setAcceptHoverEvents(True)
        chart.setAcceptTouchEvents(True)
        #Background color
        #RGB style(blue)
        # A dark green
        chart.setBackgroundBrush(QBrush(QColor(0, 255, 0, 127)))
        #The pie must be colorful ,not blue-like
        # Legend must be aligned vertically

        # set the size of pie chart not the size of the window
        chart.setAnimationOptions(QChart.SeriesAnimations)
        # Great! but besides the animation,the physical body size of it needs to be adjusted
        #chart_view = QChartView(chart)
        chart_view.setChart(chart)
        #chart_view.setRenderHint(QChartView.Antialiasing)

        return chart_view

    def displayCsv(self, df,view=None):
        try:
            if view is None:
                view = self.csvTableView
            # Use pandas to read the CSV file
            model = QStandardItemModel(df.shape[0], df.shape[1])
            model.setHorizontalHeaderLabels(df.columns)

            # Fill data
            for row in df.itertuples():
                for col_index, value in enumerate(row[1:]):
                    item = QStandardItem(str(value))
                    model.setItem(row.Index, col_index, item)

            # Set up the table view
            view.setModel(model)
            view.resizeColumnsToContents()
            view.setAlternatingRowColors(True)
            view.horizontalHeader().setStretchLastSection(True)
        except Exception as e:
            print("Error reading file:", e)
    def displayCsvInTable(self, df):
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

    def click_index_0(self):
        self.stackedWidget.setCurrentIndex(0)

    def click_index_1(self):
        self.stackedWidget.setCurrentIndex(1)

    def click_index_2(self):
        self.stackedWidget.setCurrentIndex(2)
        print("Hello world!")
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 1000)
        #File
        cleaned_pickle,_ = QFileDialog.getOpenFileName(self, "Open Pickle File", "./", "Pickle Files (*.pickle)")
        print("*"+cleaned_pickle+"*")
        data = pd.read_pickle(cleaned_pickle)
        counts = data.groupby(['station_name']).size().reset_index(name='counts')
        counts_battery_name,counts_battery_size=self.battery_proportion(data)
        # converts to a DataFrame
        counts_battery = pd.DataFrame({'batterytype': counts_battery_name, 'counts': counts_battery_size})
        #self.displayCsv(counts,self.pickleTableView)
        self.displayPieChart(counts,self.pickleTableView,"Count Statistics Pie Chart for Charging Stations",
                             order=1)
        self.displayPieChart(counts_battery, self.pickle_battery_TableView,heading="Battery Type Pie Chart",
                             order=2)
        self.data=data
    def mousePressEvent(self, event):
        #Check whether the pointer hovers over widget_2_3
        if event.buttons() == Qt.LeftButton:
            # zoom out by 50%
            current_size = self.widget_2_3.size()
            if self.widget_2_3_zoom_out:
                new_size = QSize(current_size.width() * 2, current_size.height() * 2)
            else:
                new_size = QSize(current_size.width() // 2, current_size.height() // 2)
            self.widget_2_3.resize(new_size)
            self.widget_2_3_zoom_out = not self.widget_2_3_zoom_out
            event.accept()
        if event.buttons() == Qt.RightButton:
            # zoom out by 50%
            current_size = self.widget_2_4.size()
            if self.widget_2_4_zoom_out:
                new_size = QSize(current_size.width() * 2, current_size.height() * 2)
            else:
                new_size = QSize(current_size.width() // 2, current_size.height() // 2)
            self.widget_2_4.resize(new_size)
            self.widget_2_4_zoom_out = not self.widget_2_4_zoom_out
            event.accept()
    def displayPlot(self,series,view,heading):
        # Series to DataFrame
        df=pd.DataFrame(series)
        # Draw a 2d plot
        series = QLineSeries()
        chart = QChart()
        chart.setTitle(heading)
        # Read data from DataFrame
        for row in df.itertuples():
            y = row.working
            # Convert the Timestamp to number
            #x = mdates.date2num(row.Index.strftime("%Y-%m-%d %H:%M:%S"),"yyyy-MM-dd HH:mm:ss")
            x=QDateTime.fromString(row.Index.strftime("%Y-%m-%d %H:%M:%S"),"yyyy-MM-dd HH:mm:ss")
            series.append(x.toMSecsSinceEpoch(), y)
        chart.addSeries(series)
        chart.setBackgroundPen(QPen(QColor("green")))
        chart.setBackgroundVisible(True)
        chart.setBackgroundBrush(QBrush(QColor(0, 255, 0, 127)))
        axisX=QDateTimeAxis()
        axisX.setFormat("dd-MM-yyyy hh:MM:ss")
        axisX.setTitleText("Datetime")
        chart.addAxis(axisX,Qt.AlignBottom)
        series.attachAxis(axisX)

        axisY = QValueAxis()
        axisY.setLabelFormat("%d")
        axisY.setTitleText("No. of Data Point /h")
        chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisY)

        pen=QPen(QColor("blue"))
        pen.setWidth(2)
        series.setPen(pen)

        chart.setTitle(heading)
        view.setChart(chart)
        return view
    def choose_a_station(self,data,station_name):
        filtered_df = data[(data['station_name'] == station_name) & (data['stake_name'].str.startswith("1"))]
        # if data_clean dir doesn't exist,create it
        if not os.path.exists("./data_clean"):
            os.makedirs("./data_clean")
        output_path = "./data_clean/filtered_df.pickle"
        filtered_df.to_pickle(output_path)

        filtered_df.set_index('samptime', inplace=True)

        df = filtered_df.copy()
        # 创建一个表示工作状态的虚拟列，值为1
        df['working'] = 1

        # 以每小时为单位对工作状态进行重采样，计算每小时的工作时长
        hourly_work = df['working'].resample('h').sum()
        return hourly_work
    def battery_proportion(self,data):
        # 电池类型与编号的映射
        battery_types = {
            1: '铅酸电池',
            2: '镍氢电池',
            3: '磷酸铁锂电池',
            4: '锰酸锂电池',
            5: '钴酸锂电池',
            6: '三元材料电池',
            7: '聚合物锂离子电池',
            8: '钛酸锂电池',
            99: '其它电池'
        }

        # 获取电池类型的唯一值
        unique_battery_types = data['batterytype'].unique()

        # 计算每种电池类型的计数
        battery_counts = {bt: len(data[data['batterytype'] == bt]) for bt in unique_battery_types}

        # 将电池类型编号转换为名称，并准备饼图的数据
        battery_names = [battery_types.get(bt, '未知电池类型') for bt in battery_counts]
        battery_sizes = list(battery_counts.values())
        return battery_names,battery_sizes
        # 绘制饼图
        #plt.figure(figsize=(10, 8))  # 设置图形的大小
        #plt.pie(battery_sizes, labels=battery_names, autopct='%1.1f%%', startangle=140)

        #plt.title('电池类型分布')  # 添加标题
        #plt.axis('equal')  # 确保饼图是圆形的
        #plt.show()  # 显示图形
    def click_index_3(self):
        self.stackedWidget.setCurrentIndex(3)

    def click_index_4(self):
        self.stackedWidget.setCurrentIndex(4)

    def addcsv(self):
        directory=QFileDialog.getExistingDirectory(self,"Select directory that contains CSV files")
        #filename, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "./", "CSV Files (*.csv)")
        if directory:
            self.mdf=self.merge_data(directory)
            cdf=self.cleanse_data(self.mdf)
            self.displayCsv(cdf)
    def cleanse_data(self,merged_df):
        # 数据偏移纠正
        merged_df["bmscurrent"] = merged_df["bmscurrent"] * (-1)
        merged_df["maxcurrent"] = merged_df["maxcurrent"] * (-1)
        # 数据去重
        merged_df = merged_df.drop_duplicates()
        # 按时间和站名排序
        #merged_df['samptime'] = pd.to_datetime(merged_df['samptime'], unit='ms')
        merged_df.loc[:, 'samptime'] = pd.to_datetime(merged_df['samptime'], unit='ms')
        #merged_df['fessamptime'] = pd.to_datetime(merged_df['fessamptime'], unit='ms')
        merged_df.loc[:, 'fessamptime']=pd.to_datetime(merged_df['fessamptime'], unit='ms')
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
                df = pd.read_csv(file_path, header=None, names=header_list,low_memory=False)
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
        self.merged_df=pd.DataFrame()
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

