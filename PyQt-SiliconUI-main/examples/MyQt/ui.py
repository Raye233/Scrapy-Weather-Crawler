import icons
from components.page_about import About
from components.page_homepage import ExampleHomepage
from components.page_crawl import CrawlWidgets
from components.page_train import TrainWidgets
from components.page_predict import PredictWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget

import siui
from siui.core import SiColor, SiGlobal
from siui.templates.application.application import SiliconApplication

# 载入图标
siui.core.globals.SiGlobal.siui.loadIcons(
    icons.IconDictionary(color=SiGlobal.siui.colors.fromToken(SiColor.SVG_NORMAL)).icons
)


class MySiliconApp(SiliconApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        screen_geo = QDesktopWidget().screenGeometry()
        self.setMinimumSize(1024, 380)
        self.resize(1366, 916)
        self.move((screen_geo.width() - self.width()) // 2, (screen_geo.height() - self.height()) // 2)
        self.layerMain().setTitle("Silicon UI Gallery")
        self.setWindowTitle("Silicon UI Gallery")
        self.setWindowIcon(QIcon("./img/empty_icon.png"))

        self.layerMain().addPage(ExampleHomepage(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_home_filled"),
                                 hint="主页", side="top")
        self.layerMain().addPage(CrawlWidgets(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_database_filled"),
                                 hint="数据爬取", side="top")
        self.layerMain().addPage(TrainWidgets(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_drive_train_filled"),
                                 hint="模型训练", side="top")
        self.layerMain().addPage(PredictWidgets(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_weather_partly_cloudy_day_filled"),
                                 hint="天气预测", side="top")
        self.layerMain().addPage(About(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_info_filled"),
                                 hint="关于", side="bottom")

        self.layerMain().setPage(0)

        SiGlobal.siui.reloadAllWindowsStyleSheet()
