import sys
import os
import warnings

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDate, QUrl, Qt, QRectF, QRect, QSize, QDateTime
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QBrush, QColor, QFont, QPen,QPainter
from PyQt5.QtWidgets import QFileDialog, QTableView, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QVBoxLayout, \
    QDialog, QPushButton, QLineEdit, QFormLayout, QLabel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMRegressor

from v6 import Ui_Dialog
import folium
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice, QLineSeries, QDateTimeAxis, QValueAxis
import holidays
from sklearn.model_selection import GroupShuffleSplit
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, mean_squared_error, \
    r2_score
import joblib

from PyQt5.QtGui import QPixmap
class ImageViewer(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zoomed Image")
        self.setGeometry(100, 100, pixmap.width(), pixmap.height())

        label = QLabel(self)
        label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
class ClickablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap):
        super().__init__(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.show_zoomed_image()

    def show_zoomed_image(self):
        # 创建一个新的窗口来显示放大的图片
        dialog = ImageViewer(self.pixmap())
        dialog.exec_()
class MainWidget(QtWidgets.QWidget, Ui_Dialog):
    map_data = None
    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)
        self.setupUi(self)
        self.stackedWidget.setCurrentIndex(0)
        self.initMapView()

        self.initCsvTableView()


        self.pushButton_8.clicked.connect(self.load_map_data)


        self.pushButton_20.clicked.connect(self.packageData)
        self.pushButton_11.clicked.connect(self.alert_func)

    def alert_func(self):
        data = {
            'station_id': self.lineEdit_9.text(),
            'pile_id': self.lineEdit_11.text(),
            'fault_type': self.lineEdit_10.text(),
            'model_id': self.lineEdit_12.text(),
            'start_time': self.dateEdit_5.date().toString('yyyy-MM-dd'),
            'end_time': self.dateEdit_6.date().toString('yyyy-MM-dd')
        }
        if __debug__:
            print("Alert Function's data's prepared:")
            print('Station_id is' + data['station_id'])
            print('Pile id is' + data['pile_id'])
            print('Fault type is' + data['fault_type'])
            print('Model id is' + data['model_id'])
            print('Start time is' + data['start_time'])
            print('End time is' + data['end_time'])
        assert data['fault_type'] in ['voltage_fault_1', 'voltage_fault_2', 'voltage_fault_3',
                                      'current_fault_1', 'current_fault_2',
                                      'temperature_fault_1', 'temperature_fault_2', 'all', '']
        if data['fault_type'] == '':
            data['fault_type']='all'
        assert data['model_id'] in ['LGBM','XGBoost','RFforest','']
        if data['model_id'] == '':
            data['model_id']='XGBoost'
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择充电站数据文件", "", "Pickle Files (*.pickle)")
        if __debug__:
            print("File path is" + file_path)
        fdata=self.alert_labeling(file_path,data['fault_type'],data['start_time'],data['end_time'])
        self.alert_groupby(fdata,data['fault_type'])
        self.example(data['fault_type'],data['station_id'],data['model_id'])
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
        # Seaborn window
        self.scene = QGraphicsScene()
        self.seabornView = QGraphicsView(self.scene)
        self.seabornLayout = QVBoxLayout(self.widget_3_1)  # Add to widget_3_1(Seaborn Chart)'s layout
        self.seabornLayout.addWidget(self.seabornView)
        # Confusion Matrix
        self.confusionMatrixscene = QGraphicsScene()
        self.confusionMatrixView = QGraphicsView(self.confusionMatrixscene)
        self.confusionMatrixLayout = QVBoxLayout(self.widget_3_3)  # Add to widget_3_3(Seaborn Chart)'s layout
        self.confusionMatrixLayout.addWidget(self.confusionMatrixView)
        # Metrics
        self.Metricsscene = QGraphicsScene()
        self.MetricsView = QGraphicsView(self.Metricsscene)
        self.MetricsLayout = QVBoxLayout(self.widget_3_2)  # Add to widget_3_3(Seaborn Chart)'s layout
        self.MetricsLayout.addWidget(self.MetricsView)
        #HeatMap
        self.HeatMapscene = QGraphicsScene()
        self.HeatMapView = QGraphicsView(self.HeatMapscene)
        self.HeatMapLayout = QVBoxLayout(self.widget_4_1)  # Add to widget_3_3(Seaborn Chart)'s layout
        self.HeatMapLayout.addWidget(self.HeatMapView)
        #Actual vs Predicted
        self.ActualPredictedScene=QGraphicsScene()
        self.ActualPredictedView=QGraphicsView(self.ActualPredictedScene)
        self.ActualPredictedLayout=QVBoxLayout(self.widget_4_2)
        self.ActualPredictedLayout.addWidget(self.ActualPredictedView)
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
    def train_diagnotics_model(self,picklefile,category,startdate,enddate,model,station_name,stake_name):
        if model=='':
            model='XGBoost'
        pdata=self.label_bofore_predict(pickle_file=picklefile,category=category,startdate=startdate,
                                        enddate=enddate,station_name=station_name,stake_name=stake_name)
        cdata=self.violated_value_cleanse(data=pdata)
        odata=self.overfill_data(data=cdata,category=category)
        gdata=self.save_grouped_defect(data=odata,category=category)
        rdata=self.reform_grouped_defect(data=gdata,category=category)
        self.pre_feature_engieering()
        self.split_the_dataset(category=category)
        if model == 'XGBoost':
            self.xgb_classfier_predict(category=category)
        elif model == 'RFforest':
            self.classfier_predict_via_rfforest(category=category)
        if __debug__:
            print('The following is 2nd section:----Diagnotics Model Test-----')
        self.test_diagnotics_model_result(category,model)


    def classfier_predict_via_rfforest(self,category):
        # 读取训练数据集
        X_data = pd.read_csv("./model_diagnostic/X_train.csv")
        y_data = pd.read_csv("./model_diagnostic/y_train.csv")

        # 将y_data中的每一列转换为数值型
        label_encoders = {col: LabelEncoder() for col in y_data.columns}
        for col in y_data.columns:
            y_data[col] = label_encoders[col].fit_transform(y_data[col])

        # 分层抽样进一步切分训练集和验证集
        groups = y_data.index.tolist()  # 这里假设每个样本都有一个唯一的索引作为组
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, val_index in gss.split(X_data, y_data, groups):
            X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
            y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

        # 优化后的RandomForestClassifier和MultiOutputClassifier
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 输出分类报告并格式化为小数点后4位
        for i, col in enumerate(y_data.columns):
            if col != category and col != 'all':
                continue
            print(f"Classification report for {col}:")
            report = classification_report(y_val[col], y_pred[:, i], digits=4)
            print(report)

            # 绘制混淆矩阵
            cm = confusion_matrix(y_val[col], y_pred[:, i])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders[col].classes_)
            disp.plot(cmap=plt.cm.Blues, values_format='.0f')  # 设置不使用科学计数法，且格式化为整数
            plt.title(f'Confusion Matrix for {col}')

            # 遍历每个单元格并设置文本格式
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    plt.text(c, r, format(cm[r, c], '.0f'), ha='center', va='center', color='red')

            plt.show()

            # 绘制精确率、召回率和F1得分的条形图
            report_dict = classification_report(y_val[col], y_pred[:, i], output_dict=True)
            metrics_df = pd.DataFrame(report_dict).transpose()
            metrics_df = metrics_df.iloc[:-1, :-1]  # 去掉最后的avg/total行和support列
            metrics_df.plot(kind='bar', figsize=(10, 6))
            plt.title(f'Precision, Recall, F1-Score for {col}')
            plt.xlabel('Classes')
            plt.ylabel('Scores')
            plt.ylim(0, 1)
            plt.show()

        import joblib

        # 假设 model 是你训练好的 MultiOutputClassifier(XGBClassifier) 模型
        joblib.dump(model, './model_diagnostic/multioutput_RF_model.pkl')
    def xgb_classfier_predict(self,category):
        # 读取训练数据集
        X_data = pd.read_csv("./model_diagnostic/X_train.csv")
        y_data = pd.read_csv("./model_diagnostic/y_train.csv")

        # 将y_data中的每一列转换为数值型
        label_encoders = {col: LabelEncoder() for col in y_data.columns}
        for col in y_data.columns:
            y_data[col] = label_encoders[col].fit_transform(y_data[col])

        # 分层抽样进一步切分训练集和验证集
        groups = y_data.index.tolist()  # 这里假设每个样本都有一个唯一的索引作为组
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, val_index in gss.split(X_data, y_data, groups):
            X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
            y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

        # 优化后的XGBClassifier和MultiOutputClassifier
        model = MultiOutputClassifier(XGBClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 输出分类报告并格式化为小数点后4位
        for i, col in enumerate(y_data.columns):
            if col != category and col != 'all':
                continue
            print(f"Classification report for {col}:")
            report = classification_report(y_val[col], y_pred[:, i], digits=4)
            print(report)

            # 绘制混淆矩阵
            cm = confusion_matrix(y_val[col], y_pred[:, i])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders[col].classes_)
            disp.plot(cmap=plt.cm.Blues, values_format='.0f')  # 设置不使用科学计数法，且格式化为整数
            plt.title(f'Confusion Matrix for {col}')

            # 遍历每个单元格并设置文本格式
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    plt.text(c, r, format(cm[r, c], '.0f'), ha='center', va='center', color='red')

            plt.show()

            # 绘制精确率、召回率和F1得分的条形图
            report_dict = classification_report(y_val[col], y_pred[:, i], output_dict=True)
            metrics_df = pd.DataFrame(report_dict).transpose()
            metrics_df = metrics_df.iloc[:-1, :-1]  # 去掉最后的avg/total行和support列
            metrics_df.plot(kind='bar', figsize=(10, 6))
            plt.title(f'Precision, Recall, F1-Score for {col}')
            plt.xlabel('Classes')
            plt.ylabel('Scores')
            plt.ylim(0, 1)
            plt.show()
        # 假设 model 是你训练好的 MultiOutputClassifier(XGBClassifier) 模型
        joblib.dump(model, './model_diagnostic/multioutput_xgb_model.pkl')
    def split_the_dataset(self,category):
        # 读取数据文件
        data = pd.read_pickle("./model_diagnostic/data_engineered_1.pickle")

        # 指定故障标签列
        global fault_labels
        if category == 'all':
            fault_labels = [
                'voltage_fault_1', 'voltage_fault_2', 'voltage_fault_3',
                'current_fault_1', 'current_fault_2',
                'temperature_fault_1', 'temperature_fault_2'
            ]
        else:
            fault_labels = [
                category
            ]

        # 将数据集分为特征和标签
        X = data.drop(fault_labels, axis=1)
        y = data[fault_labels]

        # 为了使用GroupShuffleSplit，我们需要一个'groups'参数
        # 我们可以通过将y转换为一个列表来创建这个参数，列表中的每个元素对应于样本的类别标签
        groups = y.index.tolist()  # 这里假设每个样本都有一个唯一的索引作为组

        # 使用GroupShuffleSplit进行拆分
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in gss.split(X, y, groups):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if __debug__:
            # 检查划分后的数据集大小
            print("Training set size:", len(X_train))
            print("Test set size:", len(X_test))

            # 打印测试集的样本
            print("X_test head:\n", X_test.head())
            print("y_test head:\n", y_test.head())
        X_train.to_csv("./model_diagnostic/X_train.csv", index=False)
        X_test.to_csv("./model_diagnostic/X_test.csv", index=False)
        y_train.to_csv("./model_diagnostic/y_train.csv", index=False)
        y_test.to_csv("./model_diagnostic/y_test.csv", index=False)
    def pre_feature_engieering(self):
        # 读取数据文件
        data = pd.read_pickle("./model_diagnostic/data_labeled_1.pickle")

        ####################处理时间类特征########################
        # 把年月日时分秒转换成单独特征
        # 假设 data 是一个包含日期时间字符串的 DataFrame，日期时间在 'samptime' 列中
        datetime = pd.to_datetime(data['samptime'])  # 将字符串转换为日期时间对象

        # 使用.dt访问器提取日期时间的组成部分
        data['year'] = datetime.dt.year
        data['month'] = datetime.dt.month
        data['day'] = datetime.dt.day
        data['hour'] = datetime.dt.hour
        data['minute'] = datetime.dt.minute  # 如果需要，也可以提取分钟
        data['second'] = datetime.dt.second  # 如果需要，也可以提取秒

        # 提取一年中的第几周
        data['week_of_year'] = datetime.dt.isocalendar().week

        # 提取周几，注意dt.weekday()返回的整数是0-6，分别代表星期一到星期日
        data['day_of_week'] = datetime.dt.weekday
        data['day_of_week'] += 1

        import holidays
        # 创建一个中国节假日的实例
        china_holidays = holidays.China()

        # 定义一个函数来检查日期是否是中国的节假日
        def is_holiday(date):
            # 使用china_holidays列表检查日期是否为假日
            # 如果是假日，返回1；否则返回0
            return 1 if date in china_holidays else 0

        # 应用函数到日期列，创建一个新的列 'is_holiday'
        data['is_holiday'] = datetime.apply(is_holiday)

        # ####################处理object类型特征########################
        data['batch_num'] = pd.factorize(data['batch'])[0] + 1  # 加1是因为factorize返回的是从0开始的整数

        df = pd.DataFrame()
        df = data[["station_name", "stake_name", "batch", "batch_num"]].drop_duplicates()
        df.to_pickle("./model_diagnostic/data_station_stake_batch.pickle")

        ####################删除无用列########################
        columns_to_drop = [
            'station_name', 'stake_name', 'station_code', 'tag', 'datatype', 'chargingpilenum',
            'chargingpileinterfacenum',
            'conncetflag', 'charginggunflag', 'electroniclockstatus', 'dcoutputcontactorstatus',
            'electricityconsumptionamount', 'servicefee', 'chargingtype', 'useridentification',
            'tariffmodelnumber', 'servicechargemodelnumber', 'msgnum', 'powertype', 'powerparameter',
            'resultsfeedback', 'fessamptime', 'dt', 'samptime', 'batch']
        data.drop(columns=columns_to_drop, inplace=True)

        ####################导出########################
        data.to_pickle("./model_diagnostic/data_engineered_1.pickle")
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

    def display_seaborn(self,station_name):
        # 加载图表图片并显示在 QGraphicsView 中
        self.display_image("model_diagnostic/RF_heatmap/"+station_name+"_fault_rate_heatmap.png",self.scene,self.seabornView)
    def display_image(self,filename,scene,view):
        # 加载图表图片并显示在 QGraphicsView 中
        pixmap = QPixmap(filename)
        pixmap_item = ClickablePixmapItem(pixmap)
        scene.addItem(pixmap_item)
        view.fitInView(pixmap_item, Qt.KeepAspectRatio)
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
    def click_index_0(self):
        self.stackedWidget.setCurrentIndex(0)
        self.reset_other_button()

    def click_index_1(self):
        self.stackedWidget.setCurrentIndex(1)
        self.reset_other_button()
        self.pushButton_2.setStyleSheet("background-color:lightblue;\n"
                                        "color:rgb(255, 255, 255);\n"
                                        "font: 14pt \"等线\";\n"
                                        "\n"
                                        "")

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
        self.reset_other_button()
        self.pushButton_4.setStyleSheet("background-color:lightblue;\n"
                                        "color:rgb(255, 255, 255);\n"
                                        "font: 14pt \"等线\";\n"
                                        "\n"
                                        "")
    def click_index_3(self):
        self.stackedWidget.setCurrentIndex(3)
        self.reset_other_button()
        self.pushButton_5.setStyleSheet("background-color:lightblue;\n"
                                        "color:rgb(255, 255, 255);\n"
                                        "font: 14pt \"等线\";\n"
                                        "\n"
                                        "")
    def click_index_4(self):
        self.stackedWidget.setCurrentIndex(4)
        self.reset_other_button()
        self.pushButton_6.setStyleSheet("background-color:lightblue;\n"
                                        "color:rgb(255, 255, 255);\n"
                                        "font: 14pt \"等线\";\n"
                                        "\n"
                                        "")

    def reset_other_button(self):
        self.pushButton_2.setStyleSheet("background-color:rgb(38, 104, 191);\n"
                                        "color:rgb(255, 255, 255);\n"
                                        "font: 14pt \"等线\";\n"
                                        "\n"
                                        "")
        self.pushButton_4.setStyleSheet("background-color:rgb(38, 104, 191);\n"
                                        "color:rgb(255, 255, 255);\n"
                                        "font: 14pt \"等线\";\n"
                                        "\n"
                                        "")
        self.pushButton_5.setStyleSheet("background-color:rgb(38, 104, 191);\n"
                                        "color:rgb(255, 255, 255);\n"
                                        "font: 14pt \"等线\";\n"
                                        "\n"
                                        "")
        self.pushButton_6.setStyleSheet("background-color:rgb(38, 104, 191);\n"
                                        "color:rgb(255, 255, 255);\n"
                                        "font: 14pt \"等线\";\n"
                                        "\n"
                                        "")



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

    def test_model_effect(self, y_pred_test, X_test, category):
        # 假设故障列的顺序是：voltage_fault_1, voltage_fault_2, voltage_fault_3, current_fault_1, current_fault_2, temperature_fault_1, temperature_fault_2
        global fault_columns
        if category == 'all':
            fault_columns = [
                'voltage_fault_1', 'voltage_fault_2', 'voltage_fault_3',
                'current_fault_1', 'current_fault_2',
                'temperature_fault_1', 'temperature_fault_2'
            ]
        else:
            fault_columns = [category]

        # 确保列数匹配
        assert y_pred_test.shape[1] == len(fault_columns), "预测结果的列数与故障列数不匹配。"

        # 添加故障列到 X_test
        for i, col in enumerate(fault_columns):
            X_test[col] = y_pred_test[:, i]

        # 读取充电站、桩和订单号码的对应关系表
        df = pd.read_pickle("./model_diagnostic/data_station_stake_batch.pickle")
        if __debug__:
            print(df)

        # 合并 df 和 X_test
        merged_df = pd.merge(X_test, df, on='batch_num', how='left')
        if __debug__:
            print(merged_df)

        # 分组统计
        global grouped_sum
        if category == 'all':
            grouped_sum = merged_df.groupby(['station_name', 'stake_name'])[
                ['voltage_fault_1', 'voltage_fault_2', 'voltage_fault_3',
                    'current_fault_1', 'current_fault_2',
                    'temperature_fault_1', 'temperature_fault_2']].sum()
        else:
            grouped_sum = merged_df.groupby(['station_name', 'stake_name'])[
                [category]].sum()

        grouped_sum.to_excel("./model_diagnostic/xgb_results/xgb_results_fault_num.xlsx")

        # 计算每个桩的总样本数
        grouped_size = merged_df.groupby(['station_name', 'stake_name']).size()
        grouped_size.to_excel("./model_diagnostic/xgb_results/xgb_results_size.xlsx")

        # 计算故障率
        fault_rate = grouped_sum.div(grouped_size, axis=0)
        fault_rate.to_excel("./model_diagnostic/xgb_results/xgb_results_fault_rate.xlsx")
        if __debug__:
            print('Fault Rate Table')
            print(fault_rate)
        self.draw_seaborn(fault_rate)
        self.risk_assessment(fault_rate,grouped_sum,fault_columns)
    def risk_assessment(self,fault_rate,grouped_sum,fault_columns):
        # 根据条件转换数字为比例
        def determine_risk_level(fault_count, fault_rate_value):
            if fault_count <= 0:
                return '无风险'
            elif fault_count > 10:
                if fault_rate_value <= 0.3:
                    return '低风险'
                elif 0.3 < fault_rate_value <= 0.6:
                    return '中风险'
                else:
                    return '高风险'
            else:
                return '无风险'

        # 创建风险等级表
        risk_df = pd.DataFrame(index=fault_rate.index, columns=fault_rate.columns)

        for (station, stake), row in grouped_sum.iterrows():
            for fault_type in fault_columns:
                fault_count = row[fault_type]
                fault_rate_value = fault_rate.loc[(station, stake), fault_type]
                risk_level = determine_risk_level(fault_count, fault_rate_value)
                risk_df.loc[(station, stake), fault_type] = risk_level

        # # 保存风险等级表
        risk_df.to_excel("./model_diagnostic/xgb_results/xgb_results_风险等级表.xlsx")
    def draw_seaborn(self,fault_rate):
        warnings.filterwarnings('ignore')

        # 指定字体为支持中文的字体，这里使用SimHei字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

        # 按照不同的充电站绘制热力图
        stations = fault_rate.index.get_level_values('station_name').unique()

        for station in stations:
            station_fault_rate = fault_rate.loc[station]

            # 对 stake_name 进行排序
            station_fault_rate = station_fault_rate.sort_index()

            plt.figure(figsize=(12, 8))
            sns.heatmap(station_fault_rate, annot=True, cmap='coolwarm', fmt=".4f")
            plt.title(f'Fault Rate Heatmap for {station}')
            plt.xlabel('Fault Types')
            plt.ylabel('Stake Name')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # 保存热力图
            plt.savefig(f"./model_diagnostic/RF_heatmap/{station}_fault_rate_heatmap.png")
            #plt.show()
            self.display_seaborn(station)
    def test_diagnotics_model_result(self, category, model):
        # 加载测试数据集
        X_test = pd.read_csv("./model_diagnostic/X_test.csv")
        y_test = pd.read_csv("./model_diagnostic/y_test.csv")

        # 加载保存的模型
        if model == 'XGBoost':
            model = joblib.load('./model_diagnostic/multioutput_xgb_model.pkl')
        elif model == 'RFforest':
            model = joblib.load('./model_diagnostic/multioutput_RF_model.pkl')
        # 预测
        y_pred_test = model.predict(X_test)

        # 输出分类报告并格式化为小数点后4位
        label_encoders = {col: LabelEncoder() for col in y_test.columns}
        for i, col in enumerate(y_test.columns):
            print(f"Classification report for {col}:")
            report = classification_report(y_test[col], y_pred_test[:, i], digits=4)
            print(report)

            # 绘制混淆矩阵
            cm = confusion_matrix(y_test[col], y_pred_test[:, i])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues, values_format='.0f')
            plt.title(f'Confusion Matrix for {col}')
            # If the directory doesn't exist, create it
            if not os.path.exists("./model_diagnostic/confusion_matrices"):
                os.makedirs("./model_diagnostic/confusion_matrices")
            plt.savefig(f"./model_diagnostic/confusion_matrices/{col}_confusion_matrix.png")
            self.display_image(f"./model_diagnostic/confusion_matrices/{col}_confusion_matrix.png"
                               ,view=self.confusionMatrixView,scene=self.confusionMatrixscene)
            # 绘制精确率、召回率和F1得分的条形图
            report_dict = classification_report(y_test[col], y_pred_test[:, i], output_dict=True)
            metrics_df = pd.DataFrame(report_dict).transpose()
            metrics_df = metrics_df.iloc[:-1, :-1]  # 去掉最后的 avg/total 行和 support 列

            # 绘制条形图
            metrics_df.plot(kind='bar', figsize=(10, 6))
            plt.title(f'Precision, Recall, F1-Score for {col}')
            plt.xlabel('Classes')
            plt.ylabel('Scores')
            plt.ylim(0, 1)
            plt.savefig(f"./model_diagnostic/metrics_bars/{col}_metrics_bars.png")
            # If the directory doesn't exist, create it
            if not os.path.exists("./model_diagnostic/metrics_bars"):
                os.makedirs("./model_diagnostic/metrics_bars")
            self.display_image(f"./model_diagnostic/metrics_bars/{col}_metrics_bars.png",
                               view=self.MetricsView,scene=self.Metricsscene)
        self.test_model_effect(y_pred_test, X_test, category)
    def packageData(self):
        # 读取控件内容并封装成字典
        data = {
            'station_id': self.lineEdit_4.text(),
            'pile_id': self.lineEdit.text(),
            'fault_type': self.lineEdit_2.text(),
            'model_id': self.lineEdit_3.text(),
            'start_time': self.dateEdit.date().toString('yyyy-MM-dd'),
            'end_time': self.dateEdit_2.date().toString('yyyy-MM-dd')
        }
        if __debug__:
            print('Station_id is'+data['station_id'])
            print('Pile id is'+data['pile_id'])
            print('Fault type is'+data['fault_type'])
            print('Model id is'+data['model_id'])
            print('Start time is'+data['start_time'])
            print('End time is'+data['end_time'])
        self.queryFaults(data)

    def queryFaults(self, data):
        if __debug__:
            print(data)
        assert data['fault_type'] in ['voltage_fault_1', 'voltage_fault_2', 'voltage_fault_3',
                                      'current_fault_1', 'current_fault_2',
                                      'temperature_fault_1', 'temperature_fault_2','all','']
        if data['fault_type'] == '':
            data['fault_type'] = 'all'
        assert data['model_id'] in ['XGBoost','RFforest','']
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择充电站数据文件", "", "Pickle Files (*.pickle)")
        if __debug__:
            print('The file path of the pickle file is'+file_path)
        self.train_diagnotics_model(picklefile=file_path,category=data['fault_type'],
                                    startdate=data['start_time'],enddate=data['end_time'],model=data['model_id'],
                                    station_name=data['station_id'],stake_name=data['pile_id'])


    def violated_value_cleanse(self,data):
        ################异常值处理##############
        # 将 demandvoltage 为 0 的值替换为下一行的值
        data['demandvoltage'] = data['demandvoltage'].replace(0, method='bfill')
        # 将 maxbmsvoltage 为 0 的值替换为下一行的值
        data['maxbmsvoltage'] = data['maxbmsvoltage'].replace(0, method='bfill')
        # 将 maxmonomervoltage 为 0 的值替换为下一行的值
        data['maxmonomervoltage'] = data['maxmonomervoltage'].replace(0, method='bfill')
        # 将 maxtemperature 为 0 的值替换为下一行的值
        data['maxtemperature'] = data['maxtemperature'].replace(0, method='bfill')

        # 将 ratedcapacity 为 0 的值替换为下一行的值
        data['ratedcapacity'] = data['ratedcapacity'].replace(0, method='bfill')
        # 将 batterytype 为 0 的值替换为下一行的值
        data['batterytype'] = data['batterytype'].replace(0, method='bfill')
        return data
    def overfill_data(self,data,category):
        # 充电桩输出电压过压
        if category == 'voltage_fault_1' or category == 'all':
            condition1 = data['outputvoltage'] > data['demandvoltage']
            data.loc[condition1, 'voltage_fault_1'] = 1
        elif category == 'voltage_fault_2' or category == 'all':
            # 动力电池充电电压过压
            condition2 = data['bmsvoltage'] > data['maxbmsvoltage']
            data.loc[condition2, 'voltage_fault_2'] = 1
        elif category == 'voltage_fault_3' or category == 'all':
            # 动力电池单体电池充电电压过压
            condition3 = data['maximumbatteryvoltage'] > data['maxmonomervoltage']
            data.loc[condition3, 'voltage_fault_3'] = 1
        elif category == 'current_fault_1' or category == 'all':
            # 充电桩输出电流过流
            condition4 = (data['outputcurrent'] > data['maxcurrent']) | (data['outputcurrent'] > data['demandcurrent'])
            data.loc[condition4, 'current_fault_1'] = 1
        elif category == 'current_fault_2' or category == 'all':
            # 动力电池充电电流过流
            condition5 = data['bmscurrent'] > data['maxcurrent']
            data.loc[condition5, 'current_fault_2'] = 1
        elif category == 'temperature_fault_1' or category == 'all':
            # 充电桩枪温度过温
            condition6 = (data['guntemperature1'] > data['maxtemperature']) | (
                        data['guntemperature2'] > data['maxtemperature']) | (
                                 data['guntemperature3'] > data['maxtemperature']) | (
                                 data['guntemperature4'] > data['maxtemperature'])
            data.loc[condition6, 'temperature_fault_1'] = 1
        elif category == 'temperature_fault_2' or category == 'all':
            # 动力电池单体电池包过温
            condition7 = data['maximumbatterytemperature'] > data['maxtemperature']
            data.loc[condition7, 'temperature_fault_2'] = 1

        if __debug__:
            print(data)
        pd.set_option('future.no_silent_downcasting', True)
        # 将None值替换为0
        if category == 'all':
            for category_ in ['voltage_fault_1', 'voltage_fault_2', 'current_fault_1',
                             'temperature_fault_1', 'voltage_fault_3', 'current_fault_2', 'temperature_fault_2']:
                data[category_] = data[category_].replace({None: 0})
        else:
            data[category] = data[category].replace({None: 0})

        data.to_pickle("./model_diagnostic/data_labeled_1.pickle")
        return data
    def reform_grouped_defect(self,data,category):
        warnings.filterwarnings('ignore')
        # 指定字体为支持中文的字体，这里使用matplotlib内置的STHeiti字体
        plt.rcParams['font.sans-serif'] = ['STHeiti']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

        # 3. 分组统计
        # 这次我们只按充电站名和充电枪名分组
        global grouped
        if category == 'all':
            grouped = data.groupby(['station_name', 'stake_name'])[
                ['voltage_fault_1', 'voltage_fault_2', 'current_fault_1',
                 'temperature_fault_1', 'voltage_fault_3', 'current_fault_2', 'temperature_fault_2']].sum()
        else:
            grouped = data.groupby(['station_name', 'stake_name'])[[category]].sum()

        # 4. 数据重塑
        # 将统计结果转换为宽格式
        wide_data = grouped.unstack()

        grouped.to_excel("./model_diagnostic/grouped_lables2.xlsx")

        pd.set_option('display.max_rows', 200)  # 默认是50
    def save_grouped_defect(self,data,category):
        ###统计各类故障的数量
        global grouped
        # 3. 分组统计
        if category == 'all':
            grouped = data.groupby(['station_name', 'stake_name', 'batch'])[
                ['voltage_fault_1', 'voltage_fault_2', 'current_fault_1',
                 'temperature_fault_1', 'voltage_fault_3', 'current_fault_2', 'temperature_fault_2']].sum()
        else:
            grouped = data.groupby(['station_name', 'stake_name', 'batch'])[
                [category]].sum()

        # 4. 数据重塑
        # 将统计结果转换为宽格式
        wide_data = grouped.unstack(level=-1)

        grouped.to_excel("./model_diagnostic/grouped_lables.xlsx")
        return grouped
    def label_bofore_predict(self,pickle_file,category,startdate,enddate,station_name,stake_name):
        # 设置打印选项，显示最多指定的行数和列数
        pd.set_option('display.max_rows', 100)  # 默认是50
        pd.set_option('display.max_columns', 50)  # 默认是根据窗口大小变化

        data = pd.read_pickle(pickle_file)
        # 添加七列值，并将七列中的所有值填充为空值
        special_column_names = [category]
        s_col_num=1
        if category == 'all':
            special_column_names = ['voltage_fault_1', 'voltage_fault_2', 'current_fault_1',
    'temperature_fault_1', 'voltage_fault_3', 'current_fault_2', 'temperature_fault_2']
            s_col_num=7
        data[special_column_names] = pd.DataFrame([[None] * s_col_num], index=data.index)
        if startdate == '' or enddate == '':
            pass
        else:
            startdate = pd.Timestamp(startdate)
            enddate = pd.Timestamp(enddate)
            # Compare between timestamp!
            data = data[(data['samptime'] >= startdate) & (data['samptime'] <= enddate)]
        # Select the spefic station and stake
        #
        if stake_name == 'all' or stake_name == '':
            data = data[(data['station_name'] == station_name)]
        else:
            data = data[(data['station_name'] == station_name) & (data['stake_name'] == stake_name)]
        if __debug__:
            print(data)
        return data

    #region alert functions
    def alert_labeling(self,pickle_filename,category,startdate,enddate):
        pd.set_option('display.max_rows', 100)  # 默认是50
        pd.set_option('display.max_columns', 50)  # 默认是根据窗口大小变化

        data = pd.read_pickle(pickle_filename)
        #region 添加七列值，并将七列中的所有值填充为空值
        global special_column_names
        global snum
        if category == 'all':
            special_column_names = ['voltage_fault_1', 'voltage_fault_2', 'current_fault_1', 'temperature_fault_1',
                                'voltage_fault_3', 'current_fault_2', 'temperature_fault_2']
            snum = 7
        else:
            special_column_names = [category]
            snum = 1
        data[special_column_names] = pd.DataFrame([[None] * snum], index=data.index)
        if startdate == '' or enddate == '':
            pass
        else:
            startdate = pd.Timestamp(startdate)
            enddate = pd.Timestamp(enddate)
            # Compare between timestamp!
            data = data[(data['samptime'] >= startdate) & (data['samptime'] <= enddate)]
        #endregion
        #region 异常值处理
        # 将 demandvoltage 为 0 的值替换为下一行的值
        data['demandvoltage'] = data['demandvoltage'].replace(0, method='bfill')
        # 将 maxbmsvoltage 为 0 的值替换为下一行的值
        data['maxbmsvoltage'] = data['maxbmsvoltage'].replace(0, method='bfill')
        # 将 maxmonomervoltage 为 0 的值替换为下一行的值
        data['maxmonomervoltage'] = data['maxmonomervoltage'].replace(0, method='bfill')
        # 将 maxtemperature 为 0 的值替换为下一行的值
        data['maxtemperature'] = data['maxtemperature'].replace(0, method='bfill')

        # 将 ratedcapacity 为 0 的值替换为下一行的值
        data['ratedcapacity'] = data['ratedcapacity'].replace(0, method='bfill')
        # 将 batterytype 为 0 的值替换为下一行的值
        data['batterytype'] = data['batterytype'].replace(0, method='bfill')
        #endregion
        #region mark the violations
        # 充电桩输出电压过压
        if category == 'voltage_fault_1' or category == 'all':
            condition1 = data['outputvoltage'] > data['demandvoltage']
            data.loc[condition1, 'voltage_fault_1'] = 1
        elif category == 'voltage_fault_2' or category == 'all':
            # 动力电池充电电压过压
            condition2 = data['bmsvoltage'] > data['maxbmsvoltage']
            data.loc[condition2, 'voltage_fault_2'] = 1
        elif category == 'voltage_fault_3' or category == 'all':
            # 动力电池单体电池充电电压过压
            condition3 = data['maximumbatteryvoltage'] > data['maxmonomervoltage']
            data.loc[condition3, 'voltage_fault_3'] = 1
        elif category == 'current_fault_1' or category == 'all':
            # 充电桩输出电流过流
            condition4 = (data['outputcurrent'] > data['maxcurrent']) | (data['outputcurrent'] > data['demandcurrent'])
            data.loc[condition4, 'current_fault_1'] = 1
        elif category == 'current_fault_2' or category == 'all':
            # 动力电池充电电流过流
            condition5 = data['bmscurrent'] > data['maxcurrent']
            data.loc[condition5, 'current_fault_2'] = 1
        elif category == 'temperature_fault_1' or category == 'all':
            # 充电桩枪温度过温
            condition6 = (data['guntemperature1'] > data['maxtemperature']) | (
                    data['guntemperature2'] > data['maxtemperature']) | (
                                 data['guntemperature3'] > data['maxtemperature']) | (
                                 data['guntemperature4'] > data['maxtemperature'])
            data.loc[condition6, 'temperature_fault_1'] = 1
        elif category == 'temperature_fault_2' or category == 'all':
            # 动力电池单体电池包过温
            condition7 = data['maximumbatterytemperature'] > data['maxtemperature']
            data.loc[condition7, 'temperature_fault_2'] = 1
        #endregion
        #region 将None值替换为0
        if category=='all':
            for fault in ['voltage_fault_1', 'voltage_fault_2', 'current_fault_1',
                          'temperature_fault_1', 'voltage_fault_3', 'current_fault_2', 'temperature_fault_2']:
                data[fault] = data[fault].replace({None: 0})
        else:
            for fault in [category]:
                data[fault] = data[fault].replace({None: 0})
        if not os.path.exists("./model_warning"):
            os.makedirs("./model_warning")
        data.to_pickle("./model_warning/data_labeled_1.pickle")
        #endregion
        return data
    def alert_groupby(self,data,category):
        #region 3. 分组统计
        global grouped
        if category == 'all':
            grouped = data.groupby(['station_name', 'stake_name', 'batch'])[
                ['voltage_fault_1', 'voltage_fault_2', 'current_fault_1',
                 'temperature_fault_1', 'voltage_fault_3', 'current_fault_2', 'temperature_fault_2']].sum()
        else:
            grouped = data.groupby(['station_name', 'stake_name', 'batch'])[
                [category]].sum()
        #endregion
        #region 4. 数据重塑
        # 将统计结果转换为宽格式
        wide_data = grouped.unstack(level=-1)

        grouped.to_excel("./model_warning/grouped_lables.xlsx")
        warnings.filterwarnings('ignore')
        # 指定字体为支持中文的字体，这里使用matplotlib内置的STHeiti字体
        plt.rcParams['font.sans-serif'] = ['STHeiti']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
        #endregion
        # 3. 分组统计
        # 这次我们只按充电站名和充电枪名分组
        if category == 'all':
            grouped = data.groupby(['station_name', 'stake_name'])[
                ['voltage_fault_1', 'voltage_fault_2', 'current_fault_1',
                 'temperature_fault_1', 'voltage_fault_3', 'current_fault_2', 'temperature_fault_2']].sum()
        else:
            grouped = data.groupby(['station_name', 'stake_name'])[[category]].sum()

        # 4. 数据重塑
        # 将统计结果转换为宽格式
        wide_data = grouped.unstack()

        grouped.to_excel("./model_warning/grouped_lables2.xlsx")

        pd.set_option('display.max_rows', 200)  # 默认是50
        return grouped

    def feature_engineering(self,data_file, pre_h,category):
        # Read data
        data = pd.read_pickle(data_file)

        # Convert 'samptime' column to datetime and extract date_hour
        data['samptime'] = pd.to_datetime(data['samptime'])
        data['date_hour'] = data['samptime'].dt.floor('H')

        # Define fault columns and feature columns
        if category == 'all':
            fault_columns = ['voltage_fault_1', 'voltage_fault_2', 'voltage_fault_3',
                         'current_fault_1', 'current_fault_2',
                         'temperature_fault_1', 'temperature_fault_2']
        else:
            fault_columns = [category]
        feature_columns = ['outputvoltage', 'outputcurrent', 'demandvoltage', 'demandcurrent',
                           'chargemode', 'soc', 'batterytype', 'minimumbatterytemperature', 'maximumbatterytemperature',
                           'cumulativechargetime', 'estimatedfullchargetime', 'maximumbatteryvoltage',
                           'minimumbatteryvoltage',
                           'totalactivepower', 'chargingtype', 'status', 'bmsvoltage', 'bmscurrent',
                           'guntemperature1', 'guntemperature2', 'guntemperature3', 'guntemperature4',
                           'maxtemperature', 'maxbmsvoltage', 'maxmonomervoltage', 'maxcurrent',
                           'ratedtotalvoltage', 'currentvoltage', 'ratedcapacity', 'nominalenergy', 'stateofcharge']

        # Group by station_name, stake_name, chargingpilenum, and date_hour
        grouped_faults = data.groupby(['station_name', 'stake_name', 'chargingpilenum', 'date_hour'])[
            fault_columns].mean()
        grouped_features = data.groupby(['station_name', 'stake_name', 'chargingpilenum', 'date_hour'])[
            feature_columns].agg(['mean', 'max', 'min', 'var'])
        grouped_features.columns = ['_'.join(col).strip() for col in grouped_features.columns.values]

        # Reset index to get final DataFrame
        df_faults = grouped_faults.reset_index()
        df_features = grouped_features.reset_index()

        # Merge faults and features
        df = pd.merge(df_faults, df_features, on=['station_name', 'stake_name', 'chargingpilenum', 'date_hour'])

        # Shift fault columns by pre_h
        for col in fault_columns:
            shifted_col_name = f'{col}_shifted_{pre_h}h'
            df[shifted_col_name] = df.groupby(['station_name', 'stake_name', 'chargingpilenum'])[col].shift(-pre_h)

        # Drop original fault columns
        df.drop(columns=fault_columns, inplace=True)

        # Extract time features
        df['month'] = df['date_hour'].dt.month
        df['day'] = df['date_hour'].dt.day
        df['hour'] = df['date_hour'].dt.hour
        df['weekday'] = df['date_hour'].dt.weekday + 1  # Monday is 0, so add 1 to match your format
        df['week_of_year'] = df['date_hour'].dt.isocalendar().week

        # Add holiday information
        cn_holidays = holidays.China()
        df['is_holiday'] = df['date_hour'].apply(lambda x: 1 if x in cn_holidays else 0)

        # Factorize object type features
        df['station_num'] = pd.factorize(df['station_name'])[0] + 1
        df['stake_num'] = pd.factorize(df['stake_name'])[0] + 1
        df['chargingpile_num'] = pd.factorize(df['chargingpilenum'])[0] + 1

        df2 = pd.DataFrame()
        df2 = df[["station_name", "stake_name", "chargingpilenum", "station_num", "stake_num",
                  "chargingpile_num"]].drop_duplicates()
        df2.to_pickle("./model_warning/data_station_stake_pile.pickle")

        # Drop rows where shifted columns are NaN
        df.dropna(subset=[f'{col}_shifted_{pre_h}h' for col in fault_columns], inplace=True)

        # Drop unnecessary columns
        columns_to_drop = ['station_name', 'stake_name', 'chargingpilenum', 'date_hour']
        df.drop(columns=columns_to_drop, inplace=True)

        ################在这里改名###############
        # 把df中[f'{col}_shifted_{pre_h}h' for col in fault_columns]的名称改为[col in fault_columns]
        def rename_shifted_columns(df, fault_columns, pre_h):
            for col in fault_columns:
                shifted_col_name = f'{col}_shifted_{pre_h}h'
                if shifted_col_name in df.columns:
                    df.rename(columns={shifted_col_name: col}, inplace=True)
            return df

        rename_shifted_columns(df, fault_columns, pre_h)

        # Save engineered data to pickle file
        output_file = f"./model_warning/data_engineered/data_engineered_{pre_h}h.pickle"
        if not os.path.exists("./model_warning/data_engineered"):
            os.makedirs("./model_warning/data_engineered")
        df.to_pickle(output_file)

        return output_file
    def example(self,category,station_name,model):
        data_file = "./model_warning/data_labeled_1.pickle"
        lead_times = [6, 12, 24, 36, 48, 72]

        for lead_time in lead_times:
            output_file = self.feature_engineering(data_file, lead_time,category)
            print(f"Data engineered for lead time {lead_time} hours saved to {output_file}")
        # 指定数据文件夹路径
        data_folder = "./model_warning/data_engineered"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        # 调用函数进行处理和保存
        self.split_and_save_data(data_folder,category)
        lead_times = [12, 24, 36, 48, 72]

        for lead_time in lead_times:
            X_train_path = f"./model_warning/data_split/X_train_{lead_time}h.csv"
            y_train_path = f"./model_warning/data_split/y_train_{lead_time}h.csv"
            global model_type
            if model=='XGBoost':
                output_model_path = f'./model_warning/multioutput_xgb_model_{lead_time}h.pkl'
                self.train_and_evaluate_model(X_train_path, y_train_path, output_model_path,category)
                model_type = "xgb"
            elif model=='RFforest':
                output_model_path = f'./model_warning/multioutput_rf_model_{lead_time}h.pkl'
                self.train_and_evaluate_model_rf(X_train_path, y_train_path, output_model_path,category)
                model_type = "rf"
            elif model=='LGBM':
                output_model_path = f'./model_warning/multioutput_lgbm_model_{lead_time}h.pkl'
                self.train_and_evaluate_model_lgbm(X_train_path, y_train_path, output_model_path,category)
                model_type = "lgbm"
        lead_times = [12, 24, 36, 48, 72]

        for lead_time in lead_times:
            self.evaluate_model(model_type, lead_time,category,very_station_name=station_name)

    def evaluate_model(self,model_type, lead_time,category,very_station_name):
        # 构建输出文件夹路径
        results_folder = f"./model_warning/{model_type}_results"
        heatmap_folder = f"./model_warning/{model_type}_heatmap"

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(heatmap_folder):
            os.makedirs(heatmap_folder)

        # 加载测试数据集
        X_test_path = f"./model_warning/data_split/X_test_{lead_time}h.csv"
        y_test_path = f"./model_warning/data_split/y_test_{lead_time}h.csv"
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)

        X_test = X_test.fillna(X_test.mean())
        y_test = y_test.fillna(y_test.mean())

        # 加载保存的模型
        model_name = f'multioutput_{model_type}_model_{lead_time}h.pkl'
        model_path = f'./model_warning/{model_name}'
        model = joblib.load(model_path)

        # 预测
        y_pred_test = model.predict(X_test)

        # 假设故障列的顺序是：voltage_fault_1, voltage_fault_2, voltage_fault_3, current_fault_1, current_fault_2, temperature_fault_1, temperature_fault_2
        if category == 'all':
            fault_columns = ['voltage_fault_1', 'voltage_fault_2', 'voltage_fault_3', 'current_fault_1', 'current_fault_2',
                             'temperature_fault_1', 'temperature_fault_2']
        else:
            fault_columns = [category]

        # 确保列数匹配
        assert y_pred_test.shape[1] == len(fault_columns), "预测结果的列数与故障列数不匹配。"

        # 添加故障列到 X_test
        for i, col in enumerate(fault_columns):
            X_test[col] = y_pred_test[:, i]

        # 读取充电站、桩和订单号码的对应关系表
        df = pd.read_pickle("./model_warning/data_station_stake_pile.pickle")

        # 合并 df 和 X_test
        merged_df = pd.merge(X_test, df, on='chargingpile_num', how='left')

        # 分组统计
        if category == 'all':
            grouped_sum = merged_df.groupby(['station_name', 'stake_name'])[
                ['voltage_fault_1', 'voltage_fault_2', 'voltage_fault_3',
                'current_fault_1', 'current_fault_2',
                'temperature_fault_1', 'temperature_fault_2']].sum()
        else:
            grouped_sum = merged_df.groupby(['station_name', 'stake_name'])[[category]].sum()
        grouped_sum.to_excel(f"{results_folder}/{model_type}_results_fault_num_{lead_time}h.xlsx")

        # 计算每个桩的总样本数
        grouped_size = merged_df.groupby(['station_name', 'stake_name']).size()
        grouped_size.to_excel(f"{results_folder}/{model_type}_results_size_{lead_time}h.xlsx")

        # 计算故障率
        fault_rate = grouped_sum.div(grouped_size, axis=0)
        fault_rate.to_excel(f"{results_folder}/{model_type}_results_fault_rate_{lead_time}h.xlsx")

        # 绘制热力图
        warnings.filterwarnings('ignore')

        # 指定字体为支持中文的字体，这里使用SimHei字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

        # 按照不同的充电站绘制热力图
        stations = fault_rate.index.get_level_values('station_name').unique()

        for station in stations:
            if station!=very_station_name:
                continue
            station_fault_rate = fault_rate.loc[station]

            # 对 stake_name 进行排序
            station_fault_rate = station_fault_rate.sort_index()

            plt.figure(figsize=(12, 8))
            sns.heatmap(station_fault_rate, annot=True, cmap='coolwarm', fmt=".4f")
            plt.title(f'Fault Rate Heatmap for {station}')
            plt.xlabel('Fault Types')
            plt.ylabel('Stake Name')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # 保存热力图
            heatmap_filename = f"{heatmap_folder}/{model_type}_{lead_time}h_{station}_fault_rate_heatmap.png"
            plt.savefig(heatmap_filename)
            self.display_image(heatmap_filename,scene=self.HeatMapscene,view=self.HeatMapView)
        # 根据条件转换数字为比例
        def determine_risk_level(fault_count, fault_rate_value):
            if fault_count <= 0:
                return '无风险'
            elif fault_count > 10:
                if fault_rate_value <= 0.3:
                    return '低风险'
                elif 0.3 < fault_rate_value <= 0.6:
                    return '中风险'
                else:
                    return '高风险'
            else:
                return '无风险'

        # 创建风险等级表
        risk_df = pd.DataFrame(index=fault_rate.index, columns=fault_rate.columns)

        for (station, stake), row in grouped_sum.iterrows():
            for fault_type in fault_columns:
                fault_count = row[fault_type]
                fault_rate_value = fault_rate.loc[(station, stake), fault_type]
                risk_level = determine_risk_level(fault_count, fault_rate_value)
                risk_df.loc[(station, stake), fault_type] = risk_level

        # 保存风险等级表
        risk_df.to_excel(f"{results_folder}/{model_type}_results_风险等级表_{lead_time}h.xlsx")

    def train_and_evaluate_model_lgbm(self,X_train_path, y_train_path, output_model_path,category):
        # 读取训练数据集
        X_data = pd.read_csv(X_train_path)
        y_data = pd.read_csv(y_train_path)

        # 分层抽样进一步切分训练集和验证集
        groups = y_data.index.tolist()  # 这里假设每个样本都有一个唯一的索引作为组
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, val_index in gss.split(X_data, y_data, groups):
            X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
            y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

        # 优化后的LGBMRegressor和MultiOutputRegressor
        model = MultiOutputRegressor(
            LGBMRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, objective='regression'))

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 确保预测值在 0-1 范围内
        y_pred = np.clip(y_pred, 0, 1)

        # 输出回归评估指标
        for i, col in enumerate(y_data.columns):
            if col != category:
                continue
            print(f"Regression metrics for {col}:")
            mse = mean_squared_error(y_val[col], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val[col], y_pred[:, i])
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")

            # 绘制实际值与预测值的对比图
            plt.figure(figsize=(10, 6))
            plt.scatter(y_val[col], y_pred[:, i], alpha=0.3)
            plt.plot([y_val[col].min(), y_val[col].max()], [y_val[col].min(), y_val[col].max()], 'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted for {col}')
            if not os.path.exists("./model_warning/avp_series"):
                os.makedirs("./model_warning/avp_series")
            avp_filename=f"./model_warning/avp_series/avp_{col}.png"
            plt.savefig(avp_filename)
            self.display_image(avp_filename,scene=self.ActualPredictedScene,view
                               =self.ActualPredictedView)
        # 保存模型
        joblib.dump(model, output_model_path)
        print(f"Model saved to {output_model_path}")
    def train_and_evaluate_model_rf(self,X_train_path, y_train_path, output_model_path,category):
        # 读取训练数据集
        X_data = pd.read_csv(X_train_path)
        y_data = pd.read_csv(y_train_path)

        # 填充 NaN 值
        X_data = X_data.fillna(X_data.mean())
        y_data = y_data.fillna(y_data.mean())

        # 分层抽样进一步切分训练集和验证集
        groups = y_data.index.tolist()  # 这里假设每个样本都有一个唯一的索引作为组
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, val_index in gss.split(X_data, y_data, groups):
            X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
            y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

        # 优化后的RandomForestRegressor和MultiOutputRegressor
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 确保预测值在 0-1 范围内
        y_pred = np.clip(y_pred, 0, 1)

        # 输出回归评估指标
        for i, col in enumerate(y_data.columns):
            if col != category:
                continue
            print(f"Regression metrics for {col}:")
            mse = mean_squared_error(y_val[col], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val[col], y_pred[:, i])
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")

            # 绘制实际值与预测值的对比图
            plt.figure(figsize=(10, 6))
            plt.scatter(y_val[col], y_pred[:, i], alpha=0.3)
            plt.plot([y_val[col].min(), y_val[col].max()], [y_val[col].min(), y_val[col].max()], 'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted for {col}')
            plt.show()

        # 保存模型
        joblib.dump(model, output_model_path)
        print(f"Model saved to {output_model_path}")
    def train_and_evaluate_model(self,X_train_path, y_train_path, output_model_path,category):
        # 读取训练数据集
        X_data = pd.read_csv(X_train_path)
        y_data = pd.read_csv(y_train_path)

        # 分层抽样进一步切分训练集和验证集
        groups = y_data.index.tolist()  # 这里假设每个样本都有一个唯一的索引作为组
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, val_index in gss.split(X_data, y_data, groups):
            X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
            y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

        # 优化后的XGBRegressor和MultiOutputRegressor
        model = MultiOutputRegressor(
            XGBRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, objective='reg:squarederror'))

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 确保预测值在 0-1 范围内
        y_pred = np.clip(y_pred, 0, 1)

        # 输出回归评估指标
        for i, col in enumerate(y_data.columns):
            if col != category:
                continue
            print(f"Regression metrics for {col}:")
            mse = mean_squared_error(y_val[col], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val[col], y_pred[:, i])
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")

            # 绘制实际值与预测值的对比图
            plt.figure(figsize=(10, 6))
            plt.scatter(y_val[col], y_pred[:, i], alpha=0.3)
            plt.plot([y_val[col].min(), y_val[col].max()], [y_val[col].min(), y_val[col].max()], 'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted for {col}')
            plt.show()

        # 保存模型
        joblib.dump(model, output_model_path)
        print(f"Model saved to {output_model_path}")
    def split_and_save_data(self,folder_path,category):
        # 获取指定文件夹下所有以"data_engineered"开头且以".pickle"结尾的文件
        files = [f for f in os.listdir(folder_path) if f.startswith("data_engineered") and f.endswith(".pickle")]

        for file in files:
            # 提取文件名中的后缀，例如"6h"
            suffix = file.split("_")[-1].split(".")[0]

            # 构建输入文件的完整路径
            file_path = os.path.join(folder_path, file)

            # 读取数据文件
            data = pd.read_pickle(file_path)

            # 指定故障标签列
            if category=='all':
                fault_labels = [
                    'voltage_fault_1', 'voltage_fault_2', 'current_fault_1',
                    'temperature_fault_1', 'voltage_fault_3', 'current_fault_2', 'temperature_fault_2'
                ]
            else:
                fault_labels = [category]

            # 将数据集分为特征和标签
            X = data.drop(fault_labels, axis=1)
            y = data[fault_labels]

            # 为了使用GroupShuffleSplit，创建一个'groups'参数
            groups = y.index.tolist()  # 假设每个样本都有一个唯一的索引作为组

            # 使用GroupShuffleSplit进行数据集拆分
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in gss.split(X, y, groups):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 构建输出文件夹路径
            output_folder = "./model_warning/data_split"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 构建输出文件名
            prefix = os.path.splitext(file)[0]  # 移除文件扩展名
            output_train_file = os.path.join(output_folder, f"X_train_{suffix}.csv")
            output_test_file = os.path.join(output_folder, f"X_test_{suffix}.csv")
            output_y_train_file = os.path.join(output_folder, f"y_train_{suffix}.csv")
            output_y_test_file = os.path.join(output_folder, f"y_test_{suffix}.csv")

            # 保存训练集、测试集和对应的y数据
            X_train.to_csv(output_train_file, index=False)
            X_test.to_csv(output_test_file, index=False)
            y_train.to_csv(output_y_train_file, index=False)
            y_test.to_csv(output_y_test_file, index=False)

            print(f"Processed and saved {file}")

    def display_map(self):
        self.save_map_to_file()  # This will save the map and update the view

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

    def save_map_to_file(self):
        offline_map_path = './tiles/{z}/{x}/{y}.png'
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
        map_file = './offline_map.html'
        m.save(map_file)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(map_file)))

    def initMapView(self):
        self.map_view = QWebEngineView()
        self.map_layout = QVBoxLayout(self.widget)
        self.map_layout.addWidget(self.map_view)
        self.display_map()  # Show initial map

    def load_map_data(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择充电站数据文件", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.map_data = pd.read_csv(file_path)
                self.display_map()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", f"无法加载文件: {e}")


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

