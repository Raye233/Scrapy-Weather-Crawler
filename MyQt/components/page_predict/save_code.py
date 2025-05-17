import os
import subprocess
import sys
import time
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
from .components.side_msg_box import send_simple_message_predict_start
from ..option_card import OptionCardPlaneForWidgetDemos
from tensorflow_predict_api import WeatherPredictor

@contextmanager
def createPanelCard(parent: QWidget, title: str) -> SiTriSectionPanelCard:
    card = SiTriSectionPanelCard(parent)
    card.setTitle(title)
    try:
        yield card
    finally:
        card.adjustSize()
        parent.addWidget(card)


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

            self.demo_progress_bar_circular_indeterminate = SiCircularProgressBar(self)
            self.demo_progress_bar_circular_indeterminate.resize(32, 32)
            self.demo_progress_bar_circular_indeterminate.setIndeterminate(False)

            self.predict_button.clicked.connect(self.start_prediction)
            self.predict_button.clicked.connect(lambda: send_simple_message_predict_start(
                self.message_type, self.message_auto_close, self.message_auto_close_duration))

            container_progress_bar_circular_ctrl_buttons = SiDenseHContainer(self)
            container_progress_bar_circular_ctrl_buttons.setFixedHeight(32)
            container_progress_bar_circular_ctrl_buttons.addWidget(self.predict_button)
            container_progress_bar_circular_ctrl_buttons.addWidget(self.demo_progress_bar_circular_indeterminate)

            self.select_button = SiPushButton(self)
            self.select_button.resize(128, 32)
            self.select_button.attachment().setText("打开文件夹")
            self.select_button.clicked.connect(self.select_folder)

            self.linear_edit_box = SiLineEdit(self)
            self.linear_edit_box.resize(560, 36)
            self.linear_edit_box.setTitle("当前模型路径为")
            self.linear_edit_box.setText("")

            self.linear_edit_box2 = SiLineEdit(self)
            self.linear_edit_box2.resize(560, 36)
            self.linear_edit_box2.setTitle("请输入要预测的日期")
            self.linear_edit_box2.setText("")

            self.select_tensorflow = SiRadioButtonR(self)
            self.select_tensorflow.setText("对应Tensorflow（DNN）")
            self.select_tensorflow.adjustSize()
            self.select_tensorflow.setChecked(True)

            self.select_pytorch = SiRadioButtonR(self)
            self.select_pytorch.setText("对应Pytorch（LSTM），目录下必须有近五天的天气数据")
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

            # 侧边栏及其按钮
            self.ctrl_show_global_drawer_left = SiPushButton(self)
            self.ctrl_show_global_drawer_left.resize(128, 32)
            self.ctrl_show_global_drawer_left.attachment().setText("查看详细概率分布")
            self.ctrl_show_global_drawer_left.clicked.connect(
                lambda: SiGlobal.siui.windows["MAIN_WINDOW"].layerLeftGlobalDrawer().showLayer()
            )

            self.named_line_edit.body().addWidget(layout1)
            self.named_line_edit.body().addWidget(layout2)
            self.named_line_edit.body().addWidget(layout3)
            self.named_line_edit.body().addWidget(self.demo_named_line_edit_7)
            self.named_line_edit.body().addWidget(self.ctrl_show_global_drawer_left)
            self.named_line_edit.body().addPlaceholder(12)
            self.named_line_edit.adjustSize()

            group.addWidget(self.named_line_edit)

        # 添加页脚的空白以增加美观性
        self.titled_widgets_group.addPlaceholder(64)
        # 设置控件组为页面对象
        self.setAttachment(self.titled_widgets_group)

    def select_folder(self):
        # 弹出文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "")

        # 如果用户选择了文件夹，将路径显示在文本框中
        if folder_path:
            self.linear_edit_box.setText(folder_path)

    def start_prediction(self):
        self.demo_progress_bar_circular_indeterminate.setIndeterminate(True)
        self.predict_button.setEnabled(False)
        self.select_button.setEnabled(False)
        model_dir = self.linear_edit_box.text()
        predict_date = self.linear_edit_box2.text()
        project_dir = r"F:\Rayedata2\weather_crawl"
        os.chdir(project_dir)
        # 检查是否选择了文件
        if not model_dir:
            QMessageBox.warning(self, "警告", "请先选择模型文件!")
            return

        if not predict_date:
            QMessageBox.warning(self, "警告", "请输入预测日期!")
            return

        try:
            # 调用预测函数并获取结果
            weather_predict = WeatherPredictor(model_dir)
            results = weather_predict.predict(predict_date)

            self.show_prediction_result(results)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"天气预测失败：{str(e)}")
        finally:
            self.prediction_finished()

    def show_prediction_result(self, results):
        self.demo_named_line_edit_1.lineEdit().setText(str(f"{results['最低温度']:.1f}"))
        self.demo_named_line_edit_2.lineEdit().setText(str(f"{results['最高温度']:.1f}"))
        self.demo_named_line_edit_3.lineEdit().setText(str(f"{results['湿度']:.1f}"))
        self.demo_named_line_edit_4.lineEdit().setText(str(results["风向"]))
        self.demo_named_line_edit_5.lineEdit().setText(str(f"{results['风力']:.1f}"))
        self.demo_named_line_edit_6.lineEdit().setText(str(results["紫外线"]))
        self.demo_named_line_edit_7.lineEdit().setText(str(f"{results['空气质量']:.1f}"))

    def prediction_finished(self):
        # 恢复按钮状态
        self.predict_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.demo_progress_bar_circular_indeterminate.setIndeterminate(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictWidgets()
    window.show()
    sys.exit(app.exec_())
