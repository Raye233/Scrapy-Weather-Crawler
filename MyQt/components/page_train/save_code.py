import os
import subprocess
import sys
from contextlib import contextmanager
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QProgressBar, QFileDialog, QTextEdit
from siui.components import SiTitledWidgetGroup, SiLongPressButton, SiPushButton, SiDenseHContainer
from siui.components.button import (
    SiPushButtonRefactor,
    SiRadioButtonR,
)
from PyQt5.QtCore import QProcess
from siui.components.container import SiTriSectionPanelCard
from siui.components.editbox import SiLineEdit
from siui.components.page import SiPage
from siui.core import SiGlobal
from .components.side_msg_box import send_simple_message_train_start, send_simple_message_success, send_simple_message_fail
from ..option_card import OptionCardPlaneForWidgetDemos
from tools.charset_detect import detect_encoding


@contextmanager
def createPanelCard(parent: QWidget, title: str) -> SiTriSectionPanelCard:
    card = SiTriSectionPanelCard(parent)
    card.setTitle(title)
    try:
        yield card
    finally:
        card.adjustSize()
        parent.addWidget(card)

class TrainingThread(QThread):
    output_signal = pyqtSignal(str)  # 用于发送输出信息到主线程

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            encoding = detect_encoding(self.file_path)
            # 打开一个管道来捕获输出
            process = subprocess.Popen(['python',
                                        'test_tensorflow_train.py',
                                        '-e',
                                        encoding,
                                        '-f',
                                        self.file_path],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       text=True)

            # 读取输出并发送到主线程
            for line in iter(process.stdout.readline, ''):
                self.output_signal.emit(line)
            process.stdout.close()
            self.output_signal.emit("模型训练已完成！")

        except Exception as e:
            self.output_signal.emit(f"模型训练失败：{e}")

class TrainWidgets(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_ui()

        self.message_auto_close_duration = None
        self.message_auto_close = None
        self.message_type = 0

    def _setup_ui(self):
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("模型训练模块")

        # 创建控件组
        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)  # 禁用调整宽度

        # 按钮
        with self.titled_widgets_group as group:
            self.editbox = OptionCardPlaneForWidgetDemos(self)
            self.editbox.setTitle("每次训练后会覆盖模型，请保存好以往模型")

            self.select_button = SiPushButton(self)
            self.select_button.resize(128, 32)
            self.select_button.attachment().setText("选择训练集")
            self.select_button.clicked.connect(self.select_file)

            self.linear_edit_box = SiLineEdit(self)
            self.linear_edit_box.resize(560, 36)
            self.linear_edit_box.setTitle("当前数据集为")
            self.linear_edit_box.setText("")

            self.train_button = SiLongPressButton(self)
            self.train_button.resize(128, 32)
            self.train_button.attachment().setText("长按以训练")
            self.train_button.clicked.connect(self.start_model_training)
            # self.train_button.clicked.connect(lambda: send_simple_message_train_start(
            #     self.message_type, self.message_auto_close, self.message_auto_close_duration))

            self.select_tensorflow = SiRadioButtonR(self)
            self.select_tensorflow.setText("使用Tensorflow")
            self.select_tensorflow.adjustSize()
            self.select_tensorflow.setChecked(True)

            self.select_pytorch = SiRadioButtonR(self)
            self.select_pytorch.setText("使用Pytorch")
            self.select_pytorch.adjustSize()

            self.editbox.body().setSpacing(11)
            self.editbox.body().addWidget(self.select_button)
            self.editbox.body().addWidget(self.linear_edit_box)
            self.editbox.body().addWidget(self.select_tensorflow)
            self.editbox.body().addWidget(self.select_pytorch)
            self.editbox.body().addWidget(self.train_button)

            self.editbox.body().adjustSize()
            self.editbox.body().addPlaceholder(12)

            self.editbox.adjustSize()

            group.addWidget(self.editbox)

        with self.titled_widgets_group as group:

            self.output_container = OptionCardPlaneForWidgetDemos(self)
            self.output_container.setTitle("Output Stream")

            container = SiDenseHContainer(self)

            self.output_text = QTextEdit(self)
            self.output_text.setReadOnly(True)
            self.output_text.setStyleSheet("""
                   QTextEdit {
                       background-color: black;
                       color: white;
                       border-radius: 10px; /* 设置圆角半径 */
                       border: 1px solid gray; /* 添加边框，使圆角效果更明显 */
                       padding: 5px; /* 添加内边距，使内容与边框之间有空隙 */
                       font-size: 14pt;
                   }
               """)
            self.output_text.adjustSize()
            # 设置输出文本框的最小尺寸
            self.output_text.setMinimumSize(900, 300)
            # 设置输出文本框的固定尺寸（可选）
            # self.output_text.setFixedSize(400, 300)

            container.addWidget(self.output_text)

            self.output_container.body().addWidget(container)
            self.output_container.body().addPlaceholder(12)
            self.output_container.adjustSize()

            group.addWidget(self.output_container)

        # 添加页脚的空白以增加美观性
        self.titled_widgets_group.addPlaceholder(64)
        # 设置控件组为页面对象
        self.setAttachment(self.titled_widgets_group)

    def select_file(self):
        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "",
                                                   "All Files (*);;Python Files (*.py);;CSV Files (*.csv)")
        # 如果用户选择了文件，将路径显示在文本框中
        if file_path:
            self.linear_edit_box.setText(file_path)
            self.file_path = file_path
            print(self.file_path)

    def start_model_training(self):
        file_path = self.linear_edit_box.text()
        project_dir = r"F:\Rayedata2\weather_crawl\weather_predict"
        os.chdir(project_dir)
        # 检查是否选择了文件
        if file_path == "":
            QMessageBox.warning(self, "警告", "请先选择文件!")
            return

        send_simple_message_train_start(
            self.message_type, self.message_auto_close, self.message_auto_close_duration)

        self.train_button.setEnabled(False)
        self.select_button.setEnabled(False)

        self.training_thread = TrainingThread(file_path)
        self.training_thread.output_signal.connect(self.update_output)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()

    def update_output(self, text):
        self.output_text.append(text)

    def training_finished(self):
        # 恢复按钮状态
        send_simple_message_success(self.message_type, self.message_auto_close, self.message_auto_close_duration)
        self.train_button.setEnabled(True)
        self.select_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainWidgets()
    window.show()
    sys.exit(app.exec_())