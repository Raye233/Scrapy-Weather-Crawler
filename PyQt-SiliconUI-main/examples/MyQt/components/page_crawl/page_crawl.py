import os
import random
import sys
import time
from contextlib import contextmanager
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer, QPointF
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QProgressBar, QVBoxLayout, QPushButton, QScrollArea, QFileDialog
from scrapy.crawler import CrawlerRunner
from siui.components.chart import SiTrendChart
from twisted.internet import defer, reactor
from siui.components import SiTitledWidgetGroup
from siui.components.button import (
    SiPushButtonRefactor,
    SiRadioButtonR,
)
from PyQt5.QtCore import QProcess
from siui.components.container import SiTriSectionPanelCard
from siui.components.editbox import SiLineEdit
from siui.components.page import SiPage
from weather_predict.weather_project.weather_project.spiders import day7_weather_spider
from scrapy.utils.project import get_project_settings
from ..option_card import OptionCardPlaneForWidgetDemos
from weather_predict.infomation_view_test import create_weather_charts, create_blank_charts
from siui.components.progress_bar import SiProgressBar
from .components.side_msg_box import send_simple_message_start, send_simple_message_end

@contextmanager
def createPanelCard(parent: QWidget, title: str) -> SiTriSectionPanelCard:
    card = SiTriSectionPanelCard(parent)
    card.setTitle(title)
    try:
        yield card
    finally:
        card.adjustSize()
        parent.addWidget(card)


class CrawlWidgets(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.charts = None
        self._setup_ui()
        self._setup_workers()

        # 新增进度条相关变量
        self.total_days = 7  # 假设总页数（根据实际情况调整）
        self.current_day = 0
        # 弹窗相关变量
        self.message_auto_close_duration = None
        self.message_auto_close = None
        self.message_type = 0

        # 新增图表区域
        self.chart_container = None

    def _setup_ui(self):
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("数据爬取模块")

        # 创建控件组
        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)  # 禁用调整宽度

        # 按钮
        with self.titled_widgets_group as group:
            self.editbox = OptionCardPlaneForWidgetDemos(self)
            self.editbox.setTitle("请输入你要爬取的城市:")

            self.linear_edit_box = SiLineEdit(self)
            self.linear_edit_box.resize(560, 36)
            self.linear_edit_box.setTitle("City Name")
            self.linear_edit_box.setText("北京")

            self.period_7_days = SiRadioButtonR(self)
            self.period_7_days.setText("近七天")
            self.period_7_days.adjustSize()
            self.period_7_days.setChecked(True)

            self.period_5_years = SiRadioButtonR(self)
            self.period_5_years.setText("近五年")
            self.period_5_years.adjustSize()

            self.check_button = SiPushButtonRefactor(self)
            self.check_button.setText("爬取数据")
            self.check_button.clicked.connect(self.on_crawl)
            self.check_button.clicked.connect(self.start_progress)
            # self.check_button.clicked.connect(self.step_progress)
            self.check_button.clicked.connect(lambda: send_simple_message_start(
                self.message_type, self.message_auto_close, self.message_auto_close_duration))

            self.editbox.body().setSpacing(11)
            self.editbox.body().addWidget(self.linear_edit_box)
            self.editbox.body().addWidget(self.period_7_days)
            self.editbox.body().addWidget(self.period_5_years)
            self.editbox.body().addWidget(self.check_button)
            self.editbox.body().adjustSize()
            self.editbox.body().addPlaceholder(12)

            self.progress_bar = SiProgressBar(self)
            self.progress_bar.resize(700, 64)
            self.progress_bar.setValue(0)
            self.editbox.body().addWidget(self.progress_bar)

            self.editbox.adjustSize()
            group.addWidget(self.editbox)

        with self.titled_widgets_group as group:
            group.addTitle("气象数据分析")

            self.view_button = SiPushButtonRefactor(self)
            self.view_button.setText("选择要查看的数据")
            self.view_button.clicked.connect(self.select_file)

            # 创建图表容器
            self.chart_container = OptionCardPlaneForWidgetDemos(self)
            self.chart_container.setTitle("多维度气象趋势")

            # 加载Matplotlib图表
            self.charts = create_blank_charts(self)
            self.charts.setMinimumSize(1000, 2400)  # 根据实际图表尺寸调整

            # 添加滚动支持
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setMinimumWidth(1000)  # 设置最小宽度
            self.scroll.setWidget(self.charts)

            self.chart_container.body().addWidget(self.view_button)
            self.chart_container.body().addWidget(self.scroll)
            self.chart_container.adjustSize()

            group.addWidget(self.chart_container)

        # 添加页脚的空白以增加美观性
        self.titled_widgets_group.addPlaceholder(64)
        # 设置控件组为页面对象
        self.setAttachment(self.titled_widgets_group)

    def plot_csv_data(self, file_path):
        try:
            # 加载Matplotlib图表
            self.charts = create_weather_charts(self, file_path)
            self.charts.setMinimumSize(1000, 2400)  # 根据实际图表尺寸调整

            # 添加滚动支持
            # self.scroll = QScrollArea()
            # self.scroll.setWidgetResizable(True)
            # self.scroll.setMinimumWidth(1000)  # 设置最小宽度
            self.scroll.setWidget(self.charts)
            self.scroll.adjustSize()

        except Exception as e:
            print(f"Error plotting data: {e}")

    def select_file(self):
        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "",
                                                   "All Files (*);;Python Files (*.py);;CSV Files (*.csv)")
        # 如果用户选择了文件，将路径显示在文本框中
        if file_path:
            self.plot_csv_data(file_path)

    def _setup_workers(self):
        """初始化工作线程"""
        # QProcess方案
        self.process = QProcess()
        self.process.finished.connect(self._on_process_finished)
        # # QThread方案
        # self.thread = QThread()
        # self.worker = CrawlWorker("")
        # self.worker.moveToThread(self.thread)
        # self.thread.started.connect(self.worker.run)
        # self.worker.finished.connect(self.thread.quit)
        # self.worker.error.connect(self._on_thread_error)

    def start_progress(self):
        # 开始进度条的动画
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)  # 每 100 毫秒更新一次进度

    def update_progress(self):
        current_value = self.progress_bar.value()
        if current_value >= 1:
            self.timer.stop()
            time.sleep(1)
            current_value = 0
            self.progress_bar.setValue(current_value)
        else:
            # self.progress_bar.setFormat("正在进行...")

            current_value += 0.005
            self.progress_bar.setValue(current_value)

    def _on_process_finished(self, exit_code):
        """进程结束处理"""
        self.check_button.setEnabled(True)
        if exit_code == 0:
            self.progress_bar.setValue(100)
            send_simple_message_end(self.message_type, self.message_auto_close, self.message_auto_close_duration)
            print("爬取任务完成！")
            # 爬取完成后显示图表
        else:
            self.progress_bar.setValue(0)
            print(f"<font color='red'>进程异常退出，代码：{exit_code}</font>")

    def on_crawl(self):
        # self.progress_bar.setFormat("正在爬取...")
        """启动爬取操作"""
        city = self.linear_edit_box.text().strip()
        if not city:
            return

        # 禁用按钮防止重复点击
        self.check_button.setEnabled(False)

        # 方案1：使用QProcess调用命令行
        self._start_scrapy_process(city)

        # 方案2：使用QThread整合Scrapy
        # self.worker.city_name = city
        # self.thread.start()

    def _start_scrapy_process(self, city):
        """通过子进程启动Scrapy"""
        scrapy_exe = r"C:\ProgramData\Miniconda3\envs\weather_predict\Scripts\scrapy.exe"
        project_dir = r"F:\Rayedata2\weather_crawl\weather_predict\weather_project"

        cmd = [
            scrapy_exe,
            "crawl", "day7_weather",
            "-a", f"city_name={city}",
        ]

        # 设置工作目录和环境变量
        self.process.setWorkingDirectory(project_dir)

        print("工作目录:", os.path.exists(project_dir))  # 应输出True
        print("scrapy.exe存在:", os.path.exists(scrapy_exe))  # 应输出True

        self.process.start(cmd[0], cmd[1:])


class CrawlWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, city_name, parent=None):
        super().__init__(parent)
        self.city_name = city_name

    @defer.inlineCallbacks
    def run(self):
        try:
            settings = get_project_settings()
            settings.set("LOG_ENABLED", False)  # 禁用Scrapy自带日志
            # 配置日志
            # 创建爬虫运行器
            runner = CrawlerRunner(settings)
            yield runner.crawl(
                day7_weather_spider.day7_WeatherSpider,
                city_name=self.city_name
            )
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if reactor.running:
                reactor.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CrawlWidgets()
    window.show()
    sys.exit(app.exec_())
