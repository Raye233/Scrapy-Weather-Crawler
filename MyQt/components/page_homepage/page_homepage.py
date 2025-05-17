
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from siui.components import SiPixLabel
from siui.components.option_card import SiOptionCardLinear, SiOptionCardPlane
from siui.components.page import SiPage
from siui.components.titled_widget_group import SiTitledWidgetGroup
from siui.components.widgets import (
    SiDenseHContainer,
    SiLabel,
)
from siui.core import GlobalFont, Si, SiColor, SiGlobal
from siui.gui import SiFont

from .components.themed_option_card import ThemedOptionCardPlane, ThemedOptionCardPlaneWithNoLink


class ExampleHomepage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 滚动区域
        self.scroll_container = SiTitledWidgetGroup(self)

        # 整个顶部
        self.head_area = SiLabel(self)
        self.head_area.setFixedHeight(1000)

        # 创建背景底图和渐变
        self.background_image = SiPixLabel(self.head_area)
        self.background_image.setFixedSize(1366, 300)
        self.background_image.setBorderRadius(6)
        self.background_image.load("./img/homepage_sky.png")

        self.background_fading_transition = SiLabel(self.head_area)
        self.background_fading_transition.setGeometry(0, 100, 0, 200)
        self.background_fading_transition.setStyleSheet(
            """
            background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0, stop:0 {}, stop:1 {})
            """.format(SiGlobal.siui.colors["INTERFACE_BG_B"],
                       SiColor.trans(SiGlobal.siui.colors["INTERFACE_BG_B"], 0))
        )

        # 创建大标题和副标题
        self.title = SiLabel(self.head_area)
        self.title.setGeometry(64, 0, 700, 128)
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.title.setText("基于Scrapy的天气预报爬取分析系统")
        self.title.setStyleSheet("color: {}".format(SiGlobal.siui.colors["TEXT_A"]))
        self.title.setFont(SiFont.tokenized(GlobalFont.XL_MEDIUM))

        # 创建一个水平容器
        self.container_for_cards = SiDenseHContainer(self.head_area)
        self.container_for_cards.move(0, -50)
        self.container_for_cards.setFixedHeight(600)
        self.container_for_cards.setAlignment(Qt.AlignCenter)
        self.container_for_cards.setSpacing(32)

        self.container_for_cardss = SiDenseHContainer(self.head_area)
        self.container_for_cardss.move(0, 350)
        self.container_for_cardss.setFixedHeight(500)
        self.container_for_cardss.setAlignment(Qt.AlignCenter)
        self.container_for_cardss.setSpacing(32)

        # 添加卡片
        self.option_card_example = ThemedOptionCardPlane(self)
        self.option_card_example.setTitle("项目简介")
        self.option_card_example.setFixedSize(400, 250)
        self.option_card_example.setThemeColor("#7573aa")
        self.option_card_example.setDescription("开发一个基于Scrapy框架的高效天气数据爬取系统，能够从多个数据源爬取当前、未来一周和历史同期的天气数据。该系统旨在为后续的天气数据分析和预测提供高质量的数据支持，确保数据的时效性和完整性。")  # noqa: E501
        self.option_card_example.setURL("https://github.com/Raye233/Weather_tool")

        self.option_card_crawl = ThemedOptionCardPlaneWithNoLink(self)
        self.option_card_crawl.setTitle("数据爬取与分析模块")
        self.option_card_crawl.setFixedSize(300, 300)
        self.option_card_crawl.setThemeColor("#855198")
        self.option_card_crawl.setDescription(
            "本系统采用Scrapy框架爬取数据，并对爬取到的数据统计分析与时间序列分析，使用matplotlib和seaborn使结果可视化。")  # noqa: E501

        self.option_card_train = ThemedOptionCardPlaneWithNoLink(self)
        self.option_card_train.setTitle("模型训练模块")
        self.option_card_train.setFixedSize(300, 300)
        self.option_card_train.setThemeColor("#855198")
        self.option_card_train.setDescription(
            "本系统分别可采用Tensorflow基于Keras的DNN模型和Pytorch中引入了嵌入层和LSTM的模型对数据集进行训练。")  # noqa: E501

        self.option_card_predict = ThemedOptionCardPlaneWithNoLink(self)
        self.option_card_predict.setTitle("天气预测模块")
        self.option_card_predict.setFixedSize(300, 300)
        self.option_card_predict.setThemeColor("#855198")
        self.option_card_predict.setDescription(
            "用户可以输入一个日期，格式为xxxx-xx-xx样式，程序将根据输入日期调用函数进行天气预测，包括最低温度、最高温度、湿度、风向、风力、紫外线强度和空气质量等信息。")  # noqa: E501

        # 添加到水平容器
        self.container_for_cards.addPlaceholder(64 - 32)
        self.container_for_cards.addWidget(self.option_card_example)
        self.container_for_cardss.addPlaceholder(64 - 32)
        self.container_for_cardss.addWidget(self.option_card_crawl)
        self.container_for_cardss.addWidget(self.option_card_train)
        self.container_for_cardss.addWidget(self.option_card_predict)
        # 添加到滚动区域容器
        self.scroll_container.addWidget(self.head_area)

        # SiQuickEffect.applyDropShadowOn(self.container_for_cards, color=(0, 0, 0, 80), blur_radius=48)

        # 下方区域标签
        self.body_area = SiLabel(self)
        self.body_area.setSiliconWidgetFlag(Si.EnableAnimationSignals)
        self.body_area.resized.connect(lambda _: self.scroll_container.adjustSize())

        # 下面的 titledWidgetGroups
        self.titled_widget_group = SiTitledWidgetGroup(self.body_area)
        self.titled_widget_group.setSiliconWidgetFlag(Si.EnableAnimationSignals)
        self.titled_widget_group.resized.connect(lambda size: self.body_area.setFixedHeight(size[1]))
        self.titled_widget_group.move(64, 0)

        # 开始搭建界面
        # 控件的线性选项卡

        # self.titled_widget_group.setSpacing(16)
        # self.titled_widget_group.addPlaceholder(64)

        # 添加到滚动区域容器
        # self.body_area.setFixedHeight(self.titled_widget_group.height())
        # self.scroll_container.addWidget(self.body_area)

        # 添加到页面
        self.setAttachment(self.scroll_container)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = event.size().width()
        self.body_area.setFixedWidth(w)
        self.background_image.setFixedWidth(w)
        self.titled_widget_group.setFixedWidth(min(w - 128, 900))
        self.background_fading_transition.setFixedWidth(w)




