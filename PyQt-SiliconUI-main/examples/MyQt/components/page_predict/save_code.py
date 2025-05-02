import os
import subprocess
import sys
from contextlib import contextmanager
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QProgressBar, QFileDialog, QTextEdit, QHBoxLayout
from siui.components import SiTitledWidgetGroup, SiLongPressButton, SiPushButton, SiDenseHContainer, \
    SiCircularProgressBar, SiLineEditWithItemName
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
            # 打开一个管道来捕获输出
            process = subprocess.Popen(['python', 'test_tensorflow_train.py', self.file_path],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       text=True)

            # 读取输出并发送到主线程
            for line in iter(process.stdout.readline, ''):
                self.output_signal.emit(line)
            process.stdout.close()
            self.output_signal.emit("天气预测已完成！")

        except Exception as e:
            self.output_signal.emit(f"天气预测失败：{e}")

class PredictWidgets(SiPage):
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
        self.setTitle("天气预测模块")

        # 创建控件组
        self.titled_widgets_group = SiTitledWidgetGroup(self)
        self.titled_widgets_group.setSpacing(32)
        self.titled_widgets_group.setAdjustWidgetsSize(True)  # 禁用调整宽度

        # 按钮
        with self.titled_widgets_group as group:
            self.editbox = OptionCardPlaneForWidgetDemos(self)
            self.editbox.setTitle("选择模型文件夹路径并输入要预测的日期，格式为xxxx-xx-xx样式")

            self.predict_button = SiPushButtonRefactor(self)
            self.predict_button.resize(128, 32)
            self.predict_button.setText("开始预测")
            # self.predict_button.clicked.connect(self.start_model_training)
            # self.predict_button.clicked.connect(lambda: send_simple_message_train_start(
            #     self.message_type, self.message_auto_close, self.message_auto_close_duration))

            self.demo_progress_bar_circular_indeterminate = SiCircularProgressBar(self)
            self.demo_progress_bar_circular_indeterminate.resize(32, 32)
            self.demo_progress_bar_circular_indeterminate.setIndeterminate(False)

            container_progress_bar_circular_ctrl_buttons = SiDenseHContainer(self)
            container_progress_bar_circular_ctrl_buttons.setFixedHeight(32)
            container_progress_bar_circular_ctrl_buttons.addWidget(self.predict_button)
            container_progress_bar_circular_ctrl_buttons.addWidget(self.demo_progress_bar_circular_indeterminate)

            self.select_button = SiPushButton(self)
            self.select_button.resize(128, 32)
            self.select_button.attachment().setText("打开文件夹")
            self.select_button.clicked.connect(self.select_file)

            self.linear_edit_box = SiLineEdit(self)
            self.linear_edit_box.resize(560, 36)
            self.linear_edit_box.setTitle("当前模型路径为")
            self.linear_edit_box.setText("")

            self.linear_edit_box2 = SiLineEdit(self)
            self.linear_edit_box2.resize(560, 36)
            self.linear_edit_box2.setTitle("请输入要预测的日期")
            self.linear_edit_box2.setText("")

            self.select_tensorflow = SiRadioButtonR(self)
            self.select_tensorflow.setText("对应Tensorflow")
            self.select_tensorflow.adjustSize()
            self.select_tensorflow.setChecked(True)

            self.select_pytorch = SiRadioButtonR(self)
            self.select_pytorch.setText("对应Pytorch")
            self.select_pytorch.adjustSize()

            self.editbox.body().setSpacing(11)
            self.editbox.body().addWidget(self.select_button)
            self.editbox.body().addWidget(self.linear_edit_box)
            self.editbox.body().addWidget(self.linear_edit_box2)
            self.editbox.body().addWidget(self.select_tensorflow)
            self.editbox.body().addWidget(self.select_pytorch)
            self.editbox.body().addWidget(container_progress_bar_circular_ctrl_buttons)
            # self.editbox.body().addWidget(self.predict_button)
            # self.editbox.body().addWidget(self.demo_progress_bar_circular_indeterminate)

            self.editbox.body().adjustSize()
            self.editbox.body().addPlaceholder(12)

            self.editbox.adjustSize()

            group.addWidget(self.editbox)

        with self.titled_widgets_group as group:
            self.named_line_edit = OptionCardPlaneForWidgetDemos(self)
            self.named_line_edit.setSourceCodeURL(
                "https://github.com/ChinaIceF/PyQt-SiliconUI/blob/main/siui/components"
                "/widgets/slider/slider.py")
            self.named_line_edit.setTitle("预测结果")

            self.demo_named_line_edit_1 = SiLineEditWithItemName(self)
            self.demo_named_line_edit_1.setName("最低温度")
            self.demo_named_line_edit_1.lineEdit().setText("")
            self.demo_named_line_edit_1.resize(300, 32)

            self.demo_named_line_edit_2 = SiLineEditWithItemName(self)
            self.demo_named_line_edit_2.setName("最高温度")
            self.demo_named_line_edit_2.lineEdit().setText("")
            self.demo_named_line_edit_2.resize(300, 32)

            layout1 = SiDenseHContainer(self)
            layout1.setFixedHeight(32)
            layout1.addWidget(self.demo_named_line_edit_1)
            layout1.addWidget(self.demo_named_line_edit_2)

            self.demo_named_line_edit_3 = SiLineEditWithItemName(self)
            self.demo_named_line_edit_3.setName("湿度")
            self.demo_named_line_edit_3.lineEdit().setText("")
            self.demo_named_line_edit_3.resize(300, 32)

            self.demo_named_line_edit_4 = SiLineEditWithItemName(self)
            self.demo_named_line_edit_4.setName("风向")
            self.demo_named_line_edit_4.lineEdit().setText("")
            self.demo_named_line_edit_4.resize(300, 32)

            layout2 = SiDenseHContainer(self)
            layout2.setFixedHeight(32)
            layout2.addWidget(self.demo_named_line_edit_3)
            layout2.addWidget(self.demo_named_line_edit_4)

            self.demo_named_line_edit_5 = SiLineEditWithItemName(self)
            self.demo_named_line_edit_5.setName("风力")
            self.demo_named_line_edit_5.lineEdit().setText("")
            self.demo_named_line_edit_5.resize(300, 32)

            self.demo_named_line_edit_6 = SiLineEditWithItemName(self)
            self.demo_named_line_edit_6.setName("紫外线")
            self.demo_named_line_edit_6.lineEdit().setText("")
            self.demo_named_line_edit_6.resize(300, 32)

            layout3 = SiDenseHContainer(self)
            layout3.setFixedHeight(32)
            layout3.addWidget(self.demo_named_line_edit_5)
            layout3.addWidget(self.demo_named_line_edit_6)

            self.demo_named_line_edit_7 = SiLineEditWithItemName(self)
            self.demo_named_line_edit_7.setName("空气质量")
            self.demo_named_line_edit_7.lineEdit().setText("")
            self.demo_named_line_edit_7.resize(300, 32)

            # self.named_line_edit.body().addWidget(self.demo_named_line_edit_1)
            # self.named_line_edit.body().addWidget(self.demo_named_line_edit_2)
            self.named_line_edit.body().addWidget(layout1)
            self.named_line_edit.body().addWidget(layout2)
            self.named_line_edit.body().addWidget(layout3)
            self.named_line_edit.body().addWidget(self.demo_named_line_edit_7)
            self.named_line_edit.body().addPlaceholder(12)
            self.named_line_edit.adjustSize()

            group.addWidget(self.named_line_edit)

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

    def start_model_training(self):
        self.predict_button.setEnabled(False)
        self.select_button.setEnabled(False)
        file_path = self.linear_edit_box.text()
        project_dir = r"F:\Rayedata2\weather_crawl\weather_predict"
        os.chdir(project_dir)
        # 检查是否选择了文件
        if file_path == "":
            QMessageBox.warning(self, "警告", "请先选择文件!")
            return

        self.training_thread = TrainingThread(file_path)
        self.training_thread.output_signal.connect(self.update_output)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()

    def update_output(self, text):
        self.output_text.append(text)

    def training_finished(self):
        # 恢复按钮状态
        send_simple_message_success(self.message_type, self.message_auto_close, self.message_auto_close_duration)
        self.predict_button.setEnabled(True)
        self.select_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictWidgets()
    window.show()
    sys.exit(app.exec_())