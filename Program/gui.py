import cv2
import PIL.Image as pil
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import threading
import base64
import random

from PyQt5.QtCore import pyqtSignal, QThread

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)


class SSVII_GUI(object):
    """
    Builds the GUI and handles user interaction by delegating to the appropriate modules
    """

    # region Constants/Control variables
    # These hold the many result display modes available (only outline, only keypoints, ...)
    DISPLAY_FM_BOUNDING_BOX = True
    DISPLAY_FM_CIRCLE_PREDICTION = True
    DISPLAY_FM_KEYPOINTS = True
    DISPLAY_FM_MATCHES = True
    DISPLAY_OD_BOUNDING_BOXES = True
    DISPLAY_OD_CLASSES = True
    DISPLAY_OD_MASKS = True

    # These hold the two possible result display layouts (single result - normal ; all results - advanced)
    RESULTS_MODE_SINGLE = 0
    RESULTS_MODE_GRID_1 = 1
    RESULTS_MODE_GRID_2 = 2
    RESULTS_MODE_PILE = 3
    RESULTS_MODE_SPIRAL = 4

    """SORT_MODE_RELEVANCE = 0
    SORT_MODE_KEYPOINTS = 1
    SORT_MODE_DESCRIPTORS = 2
    SORT_MODE_MATCHES = 3
    SORT_MODE_AVG_CONFIDENCE = 4
    SORT_MODE_DETECTIONS = 5
    SORT_MODE_FILENAME = 8"""

    FEATURE_MATCHING_MODE = 0
    OBJECT_DETECTION_MODE = 1

    ANIMATION_FRAME_RATE = 1
    ANIMATE_RESULTS_MODE = False

    running_thread = None

    # Main window dimensions
    MAIN_WINDOW_WIDTH = 941 #925
    MAIN_WINDOW_HEIGHT = 810 #750

    RESULT_MODES_AVAILABLE = ("Single", "Grid 1", "Grid 2", "Pile", "Spiral")
    SORT_MODES_AVAILABLE = {"Feature Matching": ["Relevance", "Kps/Des", "Matches", "Filename"], "Object Detection": ["Avg. Confidence", "Detections", "Filename"]}
    # endregion

    # region Set up the GUI and its widgets
    def __init__(self, ref_manager, res_manager, proc_manager):
        # Manager modules with whom the gui will interact and delegate functions to
        self.ref_manager = ref_manager  # Reference manager
        self.res_manager = res_manager  # Results (analysis) manager
        self.proc_manager = proc_manager  # Image processing manager

        # Starting state of the gui
        self.current_result_index = 0  # Current result to be displayed in normal mode
        self.current_results_mode = SSVII_GUI.RESULTS_MODE_SINGLE  # Current layout mode
        self.current_method_mode = SSVII_GUI.FEATURE_MATCHING_MODE
        self.current_graphics_view_scale_step = 0
        #self.current_sort_mode = SSVII_GUI.SORT_MODE_RELEVANCE
        self.main_window = None  # Main window reference

    def setupUi(self, MainWindow):
        """
        Sets up the user interface elements
        :param MainWindow: window that will be set up
        """

        # region Main window
        self.main_window = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(SSVII_GUI.MAIN_WINDOW_WIDTH, SSVII_GUI.MAIN_WINDOW_HEIGHT)
        MainWindow.setStyleSheet("background-color: #151515")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Title and description
        self.header_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.header_groupbox.setGeometry(QtCore.QRect(0, 0, 791, 81))
        self.header_groupbox.setMinimumSize(QtCore.QSize(791, 81))
        self.header_groupbox.setMaximumSize(QtCore.QSize(791, 81))
        self.header_groupbox.setStyleSheet("border: none")
        self.header_groupbox.setTitle("")
        self.header_groupbox.setFlat(False)
        self.header_groupbox.setObjectName("header_groupbox")
        self.label_title_ssvii = QtWidgets.QLabel(self.header_groupbox)
        self.label_title_ssvii.setGeometry(QtCore.QRect(10, 10, 81, 31))
        self.label_title_ssvii.setMinimumSize(QtCore.QSize(81, 31))
        self.label_title_ssvii.setMaximumSize(QtCore.QSize(81, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_title_ssvii.setFont(font)
        self.label_title_ssvii.setStyleSheet("color: #cccccc")
        self.label_title_ssvii.setObjectName("label_title_ssvii")
        self.label_description = QtWidgets.QLabel(self.header_groupbox)
        self.label_description.setGeometry(QtCore.QRect(10, 40, 331, 16))
        self.label_description.setMinimumSize(QtCore.QSize(331, 16))
        self.label_description.setMaximumSize(QtCore.QSize(331, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_description.setFont(font)
        self.label_description.setStyleSheet("color: #cccccc")
        self.label_description.setObjectName("label_description")

        # Footer
        self.groupbox_footer = QtWidgets.QGroupBox(self.centralwidget)
        self.groupbox_footer.setGeometry(QtCore.QRect(0, 786, 791, 21))
        self.groupbox_footer.setStyleSheet("border: none")
        self.groupbox_footer.setTitle("")
        self.groupbox_footer.setObjectName("groupbox_footer")
        self.label_footer = QtWidgets.QLabel(self.groupbox_footer)
        self.label_footer.setGeometry(QtCore.QRect(191, 0, 545, 21))
        self.label_footer.setMinimumSize(QtCore.QSize(545, 21))
        self.label_footer.setMaximumSize(QtCore.QSize(545, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_footer.setFont(font)
        self.label_footer.setStyleSheet("color: #353535")
        self.label_footer.setObjectName("label_footer")
        # endregion

        # region Files and Methods
        self.groupbox_file_selection = QtWidgets.QGroupBox(self.centralwidget)
        self.groupbox_file_selection.setGeometry(QtCore.QRect(10, 80, 300, 211))
        self.groupbox_file_selection.setStyleSheet("QGroupBox {color: #cccccc; background-color: #1e1e1e; border-radius: 10px} QGroupBox:title {padding-left: 7px; padding-top: 5px}")
        self.groupbox_file_selection.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=15, xOffset=0, yOffset=0))
        self.groupbox_file_selection.setObjectName("groupbox_file_selection")
        self.label_reference_img_path = QtWidgets.QLabel(self.groupbox_file_selection)
        self.label_reference_img_path.setGeometry(QtCore.QRect(20, 45, 200, 21))
        self.label_reference_img_path.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QLabel {color: #cccccc; background-color: #353535; border-radius: 10px}")
        self.label_reference_img_path.setText("")
        # self.label_reference_img_path.setAlignment(
        #    QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        font = QtGui.QFont()
        font.setPointSize(7)
        self.label_reference_img_path.setFont(font)
        self.label_reference_img_path.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_reference_img_path.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_reference_img_path.setContentsMargins(5, 0, 5, 0)
        self.label_reference_img_path.setObjectName("label_reference_img_path")
        self.label_reference_img_path.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_browse_reference_img = QtWidgets.QPushButton(self.groupbox_file_selection)
        self.button_browse_reference_img.setGeometry(QtCore.QRect(227, 45, 58, 21))
        self.button_browse_reference_img.setAutoDefault(False)
        self.button_browse_reference_img.setDefault(False)
        self.button_browse_reference_img.setFlat(False)
        self.button_browse_reference_img.setObjectName("button_browse_reference_img")
        self.button_browse_reference_img.setToolTip("Select a reference image to use with\nthe Feature Matching methods.")
        self.button_browse_reference_img.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_browse_reference_img.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_browse_reference_img.clicked.connect(self.on_open_ref_image)
        self.label_title_reference_img_path = QtWidgets.QLabel(self.groupbox_file_selection)
        self.label_title_reference_img_path.setGeometry(QtCore.QRect(20, 25, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_title_reference_img_path.setFont(font)
        self.label_title_reference_img_path.setObjectName("label_title_reference_img_path")
        self.label_title_reference_img_path.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_title_analysis_imgs_path = QtWidgets.QLabel(self.groupbox_file_selection)
        self.label_title_analysis_imgs_path.setGeometry(QtCore.QRect(20, 75, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_title_analysis_imgs_path.setFont(font)
        self.label_title_analysis_imgs_path.setObjectName("label_title_analysis_imgs_path")
        self.label_title_analysis_imgs_path.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.button_browse_analysis_imgs = QtWidgets.QPushButton(self.groupbox_file_selection)
        self.button_browse_analysis_imgs.setGeometry(QtCore.QRect(227, 95, 58, 21))
        self.button_browse_analysis_imgs.setAutoDefault(False)
        self.button_browse_analysis_imgs.setDefault(False)
        self.button_browse_analysis_imgs.setFlat(False)
        self.button_browse_analysis_imgs.setObjectName("button_browse_analysis_imgs")
        self.button_browse_analysis_imgs.setToolTip("Select a folder containing images/videos for analysis.\nTo process video, the \"Disk\" option should be enabled\ndue to the large number of frames to process.")
        self.button_browse_analysis_imgs.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_browse_analysis_imgs.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_browse_analysis_imgs.clicked.connect(self.on_open_analysis_images)
        self.label_analysis_imgs_path = QtWidgets.QLabel(self.groupbox_file_selection)
        self.label_analysis_imgs_path.setGeometry(QtCore.QRect(20, 95, 200, 21))
        self.label_analysis_imgs_path.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_analysis_imgs_path.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QLabel {color: #cccccc; background-color: #353535; border-radius: 10px}")
        self.label_analysis_imgs_path.setText("")
        # self.label_analysis_imgs_path.setAlignment(
        #    QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        font = QtGui.QFont()
        font.setPointSize(7)
        self.label_analysis_imgs_path.setFont(font)
        self.label_analysis_imgs_path.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_analysis_imgs_path.setObjectName("label_analysis_imgs_path")
        self.label_analysis_imgs_path.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.label_analysis_imgs_path.setContentsMargins(5, 0, 5, 0)
        self.label_title_method = QtWidgets.QLabel(self.groupbox_file_selection)
        self.label_title_method.setGeometry(QtCore.QRect(20, 125, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_title_method.setFont(font)
        self.label_title_method.setObjectName("label_title_method")
        self.label_title_method.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.combobox_method = QtWidgets.QComboBox(self.groupbox_file_selection)
        self.combobox_method.setGeometry(QtCore.QRect(20, 147, 265, 25))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.combobox_method.setFont(font)
        self.combobox_method.setStyleSheet("QComboBox {color: #cccccc; background-color: #353535; selection-background-color: #353535; border-radius: 10px} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.combobox_method.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.combobox_method.setObjectName("combobox_method")
        self.combobox_method.setToolTip("Choose the method to use.")
        model = self.combobox_method.model()
        methods_available = self.proc_manager.get_methods_available()
        for method in methods_available:
            item = QtGui.QStandardItem(method.get_description())
            item.setForeground(QtGui.QColor(204, 204, 204))
            item.setBackground(QtGui.QColor(53, 53, 53))
            model.appendRow(item)
        self.combobox_method.activated[str].connect(self.on_combobox_method_change)
        self.label_title_gpu = QtWidgets.QLabel(self.groupbox_file_selection)
        self.label_title_gpu.setGeometry(QtCore.QRect(20, 185, 60, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_title_gpu.setFont(font)
        self.label_title_gpu.setObjectName("label_title_gpu")
        self.label_title_gpu.setToolTip("Use a GPU to speed up processing. Only\navailable for some methods and a NVIDIA\nGPU with CUDA support is required.")
        self.label_title_gpu.setStyleSheet("QLabel {color: #cccccc; background-color: transparent} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")

        self.checkbox_gpu = QtWidgets.QCheckBox(self.groupbox_file_selection)
        self.checkbox_gpu.setObjectName("checkbox_gpu")
        self.checkbox_gpu.setGeometry(QtCore.QRect(60, 187, 16, 16))
        self.checkbox_gpu.setChecked(False)
        self.checkbox_gpu.setToolTip("Use a GPU to speed up processing. Only\navailable for some methods and a NVIDIA\nGPU with CUDA support is required.")
        self.checkbox_gpu.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QLabel {color: #cccccc}")
        self.label_gpu_availability = QtWidgets.QLabel(self.groupbox_file_selection)
        self.label_gpu_availability.setGeometry(QtCore.QRect(80, 185, 80, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setItalic(True)
        self.label_gpu_availability.setFont(font)
        self.label_gpu_availability.setObjectName("label_gpu_availability")
        self.label_gpu_availability.setStyleSheet("QLabel {color: #cccccc; background-color: transparent} ")
        self.check_cuda_availability()
        self.label_title_disk = QtWidgets.QLabel(self.groupbox_file_selection)
        self.label_title_disk.setGeometry(QtCore.QRect(170, 185, 60, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_title_disk.setFont(font)
        self.label_title_disk.setObjectName("label_title_disk")
        self.label_title_disk.setToolTip("Use the disk to store processing data. Results in slower overall\nprocessing speed and interface responsiveness, but makes it\npossible to process larger quantities of images or even video,\ndepending on available disk space.")
        self.label_title_disk.setStyleSheet("QLabel {color: #cccccc; background-color: transparent} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QLabel {color: #cccccc}")
        self.checkbox_disk = QtWidgets.QCheckBox(self.groupbox_file_selection)
        self.checkbox_disk.setObjectName("checkbox_disk")
        self.checkbox_disk.setGeometry(QtCore.QRect(210, 187, 16, 16))
        self.checkbox_disk.setChecked(False)
        self.checkbox_disk.setToolTip("Use the disk to store processing data. Results in slower overall\nprocessing speed and interface responsiveness, but makes it\npossible to process larger quantities of images or even video,\ndepending on available disk space.")
        self.checkbox_disk.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.checkbox_disk.stateChanged.connect(self.on_checkbox_disk_changed)
        # endregion

        # region Feature Matching Reference and Region Selection
        self.groupbox_reference = QtWidgets.QGroupBox(self.centralwidget)
        self.groupbox_reference.setGeometry(QtCore.QRect(10, 315, 300, 470))
        self.groupbox_reference.setStyleSheet("QGroupBox {color: #cccccc; background-color: #1e1e1e; border-radius: 10px} QGroupBox:title {padding-left: 7px; padding-top: 5px}")
        self.groupbox_reference.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=25, xOffset=0, yOffset=0))
        self.groupbox_reference.setObjectName("groupbox_reference")

        # Middle
        self.label_reference_img = QtWidgets.QLabel(self.groupbox_reference)
        self.label_reference_img.setGeometry(QtCore.QRect(11, 65, 276, 389))
        self.label_reference_img.setText("")
        #self.label_reference_img.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.label_reference_img.setFrameStyle(QtWidgets.QFrame.NoFrame)
        # self.label_reference_img.setPixmap(QtGui.QPixmap("./resources/image_placeholder.png"))
        self.label_reference_img.setScaledContents(False)
        self.label_reference_img.setAlignment(QtCore.Qt.AlignCenter)
        self.label_reference_img.setObjectName("label_reference_img")
        #self.label_reference_img.setToolTip("Click to enlarge.")
        self.label_reference_img.setStyleSheet("QLabel {background-color: transparent} QToolTip {color: #cccccc; background-color: #1e1e1e; border: 2px solid black}")
        #self.label_reference_img.setToolTip('<div style="width: 600px; height: 600px; background-color: #121212"><p style="color: #cccccc">Path: aaa</p>\n\n<img src="./resources/goucha.png" style="background-color: #000000"></div>')
        #self.label_reference_img.setToolTip('<h2 style="color: #cccccc; max-height: 10px">Path: aaaaa</><br><br><img src="./resources/000000000285.jpg"</>')
        self.label_reference_img.mouseReleaseEvent = self.on_popup_reference_img
        self.label_reference_img.setVisible(False)

        # Bottom
        self.label_title_reference_num_keypoints = QtWidgets.QLabel(self.groupbox_reference)
        self.label_title_reference_num_keypoints.setGeometry(QtCore.QRect(154, 15, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_title_reference_num_keypoints.setFont(font)
        self.label_title_reference_num_keypoints.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title_reference_num_keypoints.setObjectName("label_title_reference_num_keypoints")
        self.label_title_reference_num_keypoints.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_title_reference_num_keypoints.setVisible(False)
        self.label_reference_num_keypoints = QtWidgets.QLabel(self.groupbox_reference)
        self.label_reference_num_keypoints.setGeometry(QtCore.QRect(162, 34, 61, 16))
        self.label_reference_num_keypoints.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_reference_num_keypoints.setFont(font)
        self.label_reference_num_keypoints.setAlignment(QtCore.Qt.AlignCenter)
        self.label_reference_num_keypoints.setObjectName("label_reference_num_keypoints")
        self.label_reference_num_keypoints.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.button_select_region = QtWidgets.QPushButton(self.groupbox_reference)
        self.button_select_region.setGeometry(QtCore.QRect(11, 25, 74, 23))
        self.button_select_region.setObjectName("button_select_region")
        self.button_select_region.setToolTip("Draw a rectangle over a region of interest and press ENTER to continue.")
        self.button_select_region.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_select_region.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_select_region.clicked.connect(self.on_button_select_region)
        self.button_select_region.setVisible(False)
        self.button_process = QtWidgets.QPushButton(self.groupbox_reference)
        self.button_process.setGeometry(QtCore.QRect(90, 25, 61, 23))
        self.button_process.setObjectName("button_process")
        self.button_process.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_process.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_process.clicked.connect(self.on_button_process)
        self.button_process.setVisible(False)
        self.progress_bar = QtWidgets.QProgressBar(self.groupbox_reference)
        self.progress_bar.setGeometry(QtCore.QRect(90, 25, 61, 23))
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.progress_bar.setFont(font)
        #self.progress_bar.setStyleSheet("QProgressBar { color: #000000; background-color: #707070; border: 2px solid grey; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #505050; margin: 2px}")
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #707070; border: 2px solid grey; border-radius: 2px; text-align: center; }QProgressBar::chunk { background-color: #505050; }")
        self.progress_bar.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.progress_bar.setVisible(False)
        self.label_title_reference_num_descriptors = QtWidgets.QLabel(self.groupbox_reference)
        self.label_title_reference_num_descriptors.setGeometry(QtCore.QRect(227, 15, 65, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_title_reference_num_descriptors.setFont(font)
        self.label_title_reference_num_descriptors.setObjectName("label_title_reference_num_descriptors")
        self.label_title_reference_num_descriptors.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_title_reference_num_descriptors.setVisible(False)
        self.label_reference_num_descriptors = QtWidgets.QLabel(self.groupbox_reference)
        self.label_reference_num_descriptors.setGeometry(QtCore.QRect(232, 34, 61, 16))
        self.label_reference_num_descriptors.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_reference_num_descriptors.setFont(font)
        self.label_reference_num_descriptors.setAlignment(QtCore.Qt.AlignCenter)
        self.label_reference_num_descriptors.setObjectName("label_reference_num_descriptors")
        self.label_reference_num_descriptors.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        # endregion

        # region Feature Matching Results
        self.groupbox_fm_results = QtWidgets.QGroupBox(self.centralwidget)
        self.groupbox_fm_results.setGeometry(QtCore.QRect(334, 24, 596, 761))
        self.groupbox_fm_results.setStyleSheet("QGroupBox {color: #cccccc; background-color: #1e1e1e; border-radius: 10px} QGroupBox:title {padding-left: 7px; padding-top: 5px}")
        self.groupbox_fm_results.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=15, xOffset=0, yOffset=0))
        self.groupbox_fm_results.setObjectName("groupbox_results")
        self.groupbox_fm_advanced_results = QtWidgets.QGroupBox(self.groupbox_fm_results)
        self.groupbox_fm_advanced_results.setVisible(False)
        self.groupbox_fm_advanced_results.setGeometry(QtCore.QRect(11, 59, 573, 686))
        self.groupbox_fm_advanced_results.setObjectName("groupbox_advanced_results")
        self.groupbox_fm_advanced_results.setStyleSheet("background-color: #353535")
        self.groupbox_fm_advanced_results.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=15, xOffset=0, yOffset=0))
        self.graphics_view_fm_scene = QtWidgets.QGraphicsScene()
        self.graphics_view_fm = QtWidgets.QGraphicsView(self.graphics_view_fm_scene,
                                                        self.groupbox_fm_advanced_results)
        self.graphics_view_fm.setGeometry(0, 0, 573, 686)
        self.graphics_view_fm.setVisible(True)
        self.graphics_view_fm.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.graphics_view_fm.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphics_view_fm.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.graphics_view_fm.setStyleSheet(
            "QGraphicsView {background-color: transparent} QToolTip{ background-color: #1e1e1e; color: #cccccc; border: black solid 1px }")
        self.scroll_bar_graphics_view_fm = QtWidgets.QScrollBar()
        self.scroll_bar_graphics_view_fm.setStyleSheet("""
                                                                 QScrollBar:vertical { background-color: #353535; width: 10px; margin: 3px 0px 3px 0px; border-radius: 4px; }
                                                                 QScrollBar::handle:vertical { background-color: #121212; min-height: 5px;border-radius: 4px; }
                                                                 QScrollBar::sub-line:vertical { border-image: url(:/); }
                                                                 QScrollBar::add-line:vertical { border-image: url(:/); }
                                                                 QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
                                                                 """)
        self.graphics_view_fm.setVerticalScrollBar(self.scroll_bar_graphics_view_fm)
        """self.graphics_view_fm_grid_2_scene = QtWidgets.QGraphicsScene()
        self.graphics_view_fm_grid_2 = QtWidgets.QGraphicsView(self.graphics_view_fm_grid_2_scene,
                                                               self.groupbox_fm_advanced_results)
        self.graphics_view_fm_grid_2.setGeometry(0, 0, 573, 661)
        self.graphics_view_fm_grid_2.setVisible(True)
        self.graphics_view_fm_grid_2.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.graphics_view_fm_grid_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphics_view_fm_grid_2.setFrameStyle(QtWidgets.QFrame.NoFrame)"""
        #self.graphics_view_fm_grid_2.setStyleSheet(
        #    "QGraphicsView {background-color: transparent} QToolTip{ background-color: #1e1e1e; color: #cccccc; border: black solid 1px }")
        #self.scroll_bar_graphics_view_fm_grid_2 = QtWidgets.QScrollBar()
        #self.scroll_bar_graphics_view_fm_grid_2.setStyleSheet("""
        #                                                 QScrollBar:vertical { background-color: #353535; width: 10px; margin: 3px 0px 3px 0px; border-radius: 4px; }
        #                                                 QScrollBar::handle:vertical { background-color: #121212; min-height: 5px;border-radius: 4px; }
        #                                                 QScrollBar::sub-line:vertical { border-image: url(:/); }
        #                                                 QScrollBar::add-line:vertical { border-image: url(:/); }
        #                                                 QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        #                                                 """)
        #self.graphics_view_fm_grid_2.setVerticalScrollBar(self.scroll_bar_graphics_view_fm_grid_2)

        # Top
        self.button_img_fm_circle_prediction = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_img_fm_circle_prediction.setGeometry(QtCore.QRect(409, 15, 55, 20))
        self.button_img_fm_circle_prediction.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.button_img_fm_circle_prediction.setObjectName("button_img_fm_circle_prediction")
        self.button_img_fm_circle_prediction.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_img_fm_circle_prediction.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_img_fm_circle_prediction.clicked.connect(self.on_button_img_fm_circle_prediction)
        self.button_img_fm_circle_prediction.setVisible(False)
        self.button_fm_img_bounding_box = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_fm_img_bounding_box.setGeometry(QtCore.QRect(469, 15, 55, 20))
        self.button_fm_img_bounding_box.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.button_fm_img_bounding_box.setObjectName("button_img_outline")
        self.button_fm_img_bounding_box.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_fm_img_bounding_box.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_fm_img_bounding_box.clicked.connect(self.on_button_img_fm_bounding_box)
        self.button_fm_img_bounding_box.setVisible(False)
        self.button_img_fm_keypoints = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_img_fm_keypoints.setGeometry(QtCore.QRect(409, 40, 55, 20))
        self.button_img_fm_keypoints.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.button_img_fm_keypoints.setObjectName("button_img_fm_keypoints")
        self.button_img_fm_keypoints.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_img_fm_keypoints.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_img_fm_keypoints.clicked.connect(self.on_button_img_fm_keypoints)
        self.button_img_fm_keypoints.setVisible(False)
        self.button_img_fm_matches = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_img_fm_matches.setGeometry(QtCore.QRect(469, 40, 55, 20))
        self.button_img_fm_matches.setObjectName("button_img_fm_matches")
        self.button_img_fm_matches.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_img_fm_matches.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_img_fm_matches.clicked.connect(self.on_button_img_fm_matches)
        self.button_img_fm_matches.setVisible(False)
        """self.button_change_mode = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_change_mode.setGeometry(QtCore.QRect(529, 40, 55, 20))
        self.button_change_mode.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.button_change_mode.setObjectName("button_change_mode")
        self.button_change_mode.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_change_mode.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_change_mode.clicked.connect(self.on_button_change_mode)
        self.button_change_mode.setVisible(False)"""
        self.combobox_fm_results_mode = QtWidgets.QComboBox(self.groupbox_fm_results)
        self.combobox_fm_results_mode.setGeometry(QtCore.QRect(529, 15, 55, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.combobox_fm_results_mode.setFont(font)
        self.combobox_fm_results_mode.setStyleSheet(
            "QComboBox {color: #cccccc; background-color: #353535; selection-background-color: #353535; border-radius: 10px} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.combobox_fm_results_mode.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.combobox_fm_results_mode.setObjectName("combobox_fm_results_mode")
        self.combobox_fm_results_mode.setToolTip("Choose the results display mode.")
        self.combobox_fm_results_mode.setVisible(False)
        model = self.combobox_fm_results_mode.model()
        for i in range(len(SSVII_GUI.RESULT_MODES_AVAILABLE)):
            item = QtGui.QStandardItem(SSVII_GUI.RESULT_MODES_AVAILABLE[i])
            item.setForeground(QtGui.QColor(204, 204, 204))
            item.setBackground(QtGui.QColor(53, 53, 53))
            model.appendRow(item)
        self.combobox_fm_results_mode.activated[str].connect(self.on_combobox_results_mode_change)
        self.combobox_fm_sort_mode = QtWidgets.QComboBox(self.groupbox_fm_results)
        self.combobox_fm_sort_mode.setGeometry(QtCore.QRect(529, 40, 55, 20))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.combobox_fm_sort_mode.setFont(font)
        self.combobox_fm_sort_mode.setStyleSheet(
            "QComboBox {color: #cccccc; background-color: #353535; selection-background-color: #353535; border-radius: 10px} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.combobox_fm_sort_mode.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.combobox_fm_sort_mode.setObjectName("combobox_fm_results_mode")
        self.combobox_fm_sort_mode.setToolTip("Choose the results display mode.")
        self.combobox_fm_sort_mode.setVisible(False)
        model = self.combobox_fm_sort_mode.model()
        for i in range(len(SSVII_GUI.SORT_MODES_AVAILABLE["Feature Matching"])):
            item = QtGui.QStandardItem(SSVII_GUI.SORT_MODES_AVAILABLE["Feature Matching"][i])
            item.setForeground(QtGui.QColor(204, 204, 204))
            item.setBackground(QtGui.QColor(53, 53, 53))
            model.appendRow(item)
        self.combobox_fm_sort_mode.activated[str].connect(self.on_combobox_sort_mode_changed)
        self.button_prev_result = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_prev_result.setGeometry(QtCore.QRect(332, 15, 22, 20))
        self.button_prev_result.setObjectName("button_prev_result")
        self.button_prev_result.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_prev_result.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_prev_result.setVisible(False)
        self.button_prev_result.setIcon(QtGui.QIcon("./resources/previous_icon.png"))
        self.button_prev_result.clicked.connect(self.on_button_prev_result)
        self.button_next_result = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_next_result.setGeometry(QtCore.QRect(380, 15, 22, 20))
        self.button_next_result.setObjectName("button_next_result")
        self.button_next_result.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_next_result.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_next_result.setVisible(False)
        self.button_next_result.setIcon(QtGui.QIcon("./resources/next_icon.png"))
        self.button_next_result.clicked.connect(self.on_button_next_result)
        self.button_animate = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_animate.setGeometry(QtCore.QRect(356, 15, 22, 20))
        self.button_animate.setObjectName("button_animate")
        self.button_animate.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_animate.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_animate.setVisible(False)
        self.button_animate.setIcon(QtGui.QIcon("./resources/play_icon.png"))
        self.button_animate.clicked.connect(self.on_button_animate)
        self.button_speed_down = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_speed_down.setGeometry(QtCore.QRect(332, 40, 33, 20))
        self.button_speed_down.setObjectName("button_speed_down")
        self.button_speed_down.setToolTip("Decrease playback frame rate by 10 fps (min. 1).")
        self.button_speed_down.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_speed_down.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_speed_down.setVisible(False)
        self.button_speed_down.setDisabled(True)
        self.button_speed_down.setIcon(QtGui.QIcon("./resources/speed_down_icon.png"))
        self.button_speed_down.clicked.connect(self.on_button_speed_down)
        self.button_speed_up = QtWidgets.QPushButton(self.groupbox_fm_results)
        self.button_speed_up.setGeometry(QtCore.QRect(369, 40, 33, 20))
        self.button_speed_up.setObjectName("button_speed_up")
        self.button_speed_up.setToolTip("Increase playback frame rate by 10 fps (max. 60).")
        self.button_speed_up.setStyleSheet("QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px } QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_speed_up.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_speed_up.setVisible(False)
        self.button_speed_up.setDisabled(True)
        self.button_speed_up.setIcon(QtGui.QIcon("./resources/speed_up_icon.png"))
        self.button_speed_up.clicked.connect(self.on_button_speed_up)
        self.label_title_result_relevance = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_title_result_relevance.setGeometry(QtCore.QRect(15, 20, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_title_result_relevance.setFont(font)
        self.label_title_result_relevance.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.label_title_result_relevance.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title_result_relevance.setObjectName("label_title_result_relevance")
        self.label_title_result_relevance.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_title_result_relevance.setVisible(False)
        self.label_result_relevance = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_result_relevance.setGeometry(QtCore.QRect(15, 39, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_result_relevance.setFont(font)
        self.label_result_relevance.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result_relevance.setObjectName("label_result_relevance")
        self.label_result_relevance.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_result_relevance.setVisible(False)
        self.label_result_num_keypoints = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_result_num_keypoints.setGeometry(QtCore.QRect(75, 39, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_result_num_keypoints.setFont(font)
        self.label_result_num_keypoints.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result_num_keypoints.setObjectName("label_result_num_keypoints")
        self.label_result_num_keypoints.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_result_num_keypoints.setVisible(False)
        self.label_title_result_num_keypoints = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_title_result_num_keypoints.setGeometry(QtCore.QRect(75, 20, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_title_result_num_keypoints.setFont(font)
        self.label_title_result_num_keypoints.setObjectName("label_title_result_num_keypoints")
        self.label_title_result_num_keypoints.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_title_result_num_keypoints.setVisible(False)
        self.label_result_num_descriptors = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_result_num_descriptors.setGeometry(QtCore.QRect(145, 39, 65, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_result_num_descriptors.setFont(font)
        self.label_result_num_descriptors.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result_num_descriptors.setObjectName("label_result_num_descriptors")
        self.label_result_num_descriptors.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_result_num_descriptors.setVisible(False)
        self.label_title_result_num_descriptors = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_title_result_num_descriptors.setGeometry(QtCore.QRect(145, 20, 65, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_title_result_num_descriptors.setFont(font)
        self.label_title_result_num_descriptors.setObjectName("label_title_result_num_keypoints")
        self.label_title_result_num_descriptors.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_title_result_num_descriptors.setVisible(False)
        self.label_result_num_matches = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_result_num_matches.setGeometry(QtCore.QRect(219, 39, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_result_num_matches.setFont(font)
        self.label_result_num_matches.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result_num_matches.setObjectName("label_result_num_matches")
        self.label_result_num_matches.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_result_num_matches.setVisible(False)
        self.label_title_result_num_matches = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_title_result_num_matches.setGeometry(QtCore.QRect(219, 20, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_title_result_num_matches.setFont(font)
        self.label_title_result_num_matches.setObjectName("label_title_result_num_matches")
        self.label_title_result_num_matches.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_title_result_num_matches.setVisible(False)
        self.label_result_filename = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_result_filename.setGeometry(QtCore.QRect(279, 39, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_result_filename.setFont(font)
        self.label_result_filename.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_result_filename.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_result_filename.setContentsMargins(0, 0, 3, 0)
        self.label_result_filename.setObjectName("label_result_filename")
        self.label_result_filename.setStyleSheet("QLabel {color: #cccccc; background-color: transparent} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.label_result_filename.setVisible(False)
        self.label_title_result_filename = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_title_result_filename.setGeometry(QtCore.QRect(279, 20, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_title_result_filename.setFont(font)
        self.label_title_result_filename.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title_result_filename.setObjectName("label_title_result_filename")
        self.label_title_result_filename.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_title_result_filename.setVisible(False)

        # Middle
        self.label_result_img = QtWidgets.QLabel(self.groupbox_fm_results)
        self.label_result_img.setGeometry(QtCore.QRect(11, 76, 573, 669))
        self.label_result_img.setText("")
        #self.label_result_img.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.label_result_img.setFrameStyle(QtWidgets.QFrame.NoFrame)
        #self.label_result_img.setPixmap(QtGui.QPixmap("./resources/image_placeholder.png"))
        #self.label_result_img.setPixmap(QtGui.QPixmap("./resources/a.png"))
        self.label_result_img.setScaledContents(False)
        self.label_result_img.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result_img.setObjectName("label_result_image")
        self.label_result_img.setToolTip("Click to enlarge.")
        self.label_result_img.setStyleSheet("QLabel {background-color: transparent} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.label_result_img.mouseReleaseEvent = self.on_popup_result_img
        self.label_result_img.setVisible(False)
        # endregion

        # region Object Detection Options
        self.groupbox_od_options = QtWidgets.QGroupBox(self.centralwidget)
        self.groupbox_od_options.setGeometry(QtCore.QRect(334, 24, 596, 217))
        self.groupbox_od_options.setStyleSheet("QGroupBox {color: #cccccc; background-color: #1e1e1e; border-radius: 10px} QGroupBox:title {padding-left: 7px; padding-top: 5px}")
        self.groupbox_od_options.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=15, xOffset=0, yOffset=0))
        self.groupbox_od_options.setObjectName("groupbox_od_options")
        self.groupbox_od_options.setVisible(False)

        self.label_od_all_stats = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_all_stats.setGeometry(QtCore.QRect(10, 20, 90, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_od_all_stats.setFont(font)
        self.label_od_all_stats.setAlignment(QtCore.Qt.AlignLeft)
        self.label_od_all_stats.setObjectName("label_od_all_stats")
        self.label_od_all_stats.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_all_stats.setVisible(False)
        self.label_od_title_all_avg_conf = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_title_all_avg_conf.setGeometry(QtCore.QRect(10, 50, 80, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_title_all_avg_conf.setFont(font)
        self.label_od_title_all_avg_conf.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_title_all_avg_conf.setObjectName("label_od_title_all_avg_conf")
        self.label_od_title_all_avg_conf.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_all_avg_conf.setVisible(False)
        self.label_od_all_avg_conf = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_all_avg_conf.setGeometry(QtCore.QRect(10, 69, 80, 16))
        self.label_od_all_avg_conf.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_all_avg_conf.setFont(font)
        self.label_od_all_avg_conf.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_all_avg_conf.setObjectName("label_od_all_avg_conf")
        self.label_od_all_avg_conf.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_all_num_classes = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_title_all_num_classes.setGeometry(QtCore.QRect(105, 50, 80, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_title_all_num_classes.setFont(font)
        self.label_od_title_all_num_classes.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_title_all_num_classes.setObjectName("label_od_title_all_num_classes")
        self.label_od_title_all_num_classes.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_all_num_classes.setVisible(False)
        self.label_od_all_num_classes = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_all_num_classes.setGeometry(QtCore.QRect(105, 69, 80, 16))
        self.label_od_all_num_classes.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_all_num_classes.setFont(font)
        self.label_od_all_num_classes.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_all_num_classes.setObjectName("label_od_all_num_classes")
        self.label_od_all_num_classes.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_all_num_detections = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_title_all_num_detections.setGeometry(QtCore.QRect(190, 50, 80, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_title_all_num_detections.setFont(font)
        self.label_od_title_all_num_detections.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_title_all_num_detections.setObjectName("label_od_title_all_num_detections")
        self.label_od_title_all_num_detections.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_all_num_detections.setVisible(False)
        self.label_od_all_num_detections = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_all_num_detections.setGeometry(QtCore.QRect(190, 69, 80, 16))
        self.label_od_all_num_detections.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_all_num_detections.setFont(font)
        self.label_od_all_num_detections.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_all_num_detections.setObjectName("label_od_all_num_detections")
        self.label_od_all_num_detections.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_single_stats = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_single_stats.setGeometry(QtCore.QRect(8, 140, 90, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_od_single_stats.setFont(font)
        self.label_od_single_stats.setAlignment(QtCore.Qt.AlignLeft)
        self.label_od_single_stats.setObjectName("label_od_single_stats")
        self.label_od_single_stats.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_single_stats.setVisible(False)
        self.label_od_title_avg_conf = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_title_avg_conf.setGeometry(QtCore.QRect(10, 170, 80, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_title_avg_conf.setFont(font)
        self.label_od_title_avg_conf.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_title_avg_conf.setObjectName("label_od_title_avg_conf")
        self.label_od_title_avg_conf.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_avg_conf.setVisible(False)
        self.label_od_avg_conf = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_avg_conf.setGeometry(QtCore.QRect(10, 189, 80, 16))
        self.label_od_avg_conf.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_avg_conf.setFont(font)
        self.label_od_avg_conf.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_avg_conf.setObjectName("label_od_avg_conf")
        self.label_od_avg_conf.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_num_classes = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_title_num_classes.setGeometry(QtCore.QRect(90, 170, 80, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_title_num_classes.setFont(font)
        self.label_od_title_num_classes.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_title_num_classes.setObjectName("label_od_title_num_classes")
        self.label_od_title_num_classes.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_num_classes.setVisible(False)
        self.label_od_num_classes = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_num_classes.setGeometry(QtCore.QRect(90, 189, 80, 16))
        self.label_od_num_classes.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_num_classes.setFont(font)
        self.label_od_num_classes.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_num_classes.setObjectName("label_od_num_classes")
        self.label_od_num_classes.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_num_detections = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_title_num_detections.setGeometry(QtCore.QRect(165, 170, 80, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_title_num_detections.setFont(font)
        self.label_od_title_num_detections.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_title_num_detections.setObjectName("label_od_title_num_detections")
        self.label_od_title_num_detections.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_title_num_detections.setVisible(False)
        self.label_od_num_detections = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_num_detections.setGeometry(QtCore.QRect(165, 189, 80, 16))
        self.label_od_num_detections.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_num_detections.setFont(font)
        self.label_od_num_detections.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_num_detections.setObjectName("label_od_num_detections")
        self.label_od_num_detections.setStyleSheet("QLabel {color: #cccccc; background-color: transparent}")
        self.label_od_filename = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_filename.setGeometry(QtCore.QRect(250, 189, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_filename.setFont(font)
        self.label_od_filename.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_od_filename.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_od_filename.setContentsMargins(0, 0, 3, 0)
        self.label_od_filename.setObjectName("label_od_filename")
        self.label_od_filename.setStyleSheet("QLabel {color: #cccccc; background-color: transparent} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.label_od_filename.setVisible(False)
        self.label_od_title_filename = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_title_filename.setGeometry(QtCore.QRect(250, 170, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_od_title_filename.setFont(font)
        self.label_od_title_filename.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_title_filename.setObjectName("label_od_title_filename")
        self.label_od_title_filename.setVisible(False)
        self.label_od_title_filename.setStyleSheet("QLabel {color: #cccccc; background-color: transparent} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.label_od_options_classes = QtWidgets.QLabel(self.groupbox_od_options)
        self.label_od_options_classes.setGeometry(QtCore.QRect(436, 15, 105, 10))
        self.label_od_options_classes.setObjectName("label_od_options_classes")
        self.label_od_options_classes.setToolTip("Classes that the current method is trained to detect.")
        self.label_od_options_classes.setStyleSheet("QLabel {color: #cccccc; background-color: transparent} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_od_options_classes.setFont(font)
        self.list_od_classes = QtWidgets.QListWidget(self.groupbox_od_options)
        self.list_od_classes.setGeometry(QtCore.QRect(436, 35, 150, 165))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.list_od_classes.setFont(font)
        self.list_od_classes.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.list_od_classes.setStyleSheet("QListWidget {color: #cccccc; background-color: #353535}")
        self.list_od_classes.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.scroll_bar_od_classes = QtWidgets.QScrollBar()
        self.scroll_bar_od_classes.setStyleSheet("""
                                                 QScrollBar:vertical { background-color: #353535; width: 10px; margin: 3px 0px 3px 0px; border-radius: 4px; }
                                                 QScrollBar::handle:vertical { background-color: #121212; min-height: 5px;border-radius: 4px; }
                                                 QScrollBar::sub-line:vertical { border-image: url(:/); }
                                                 QScrollBar::add-line:vertical { border-image: url(:/); }
                                                 QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
                                                 """)
        self.list_od_classes.setVerticalScrollBar(self.scroll_bar_od_classes)
        self.list_od_classes.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.list_od_classes.itemChanged.connect(self.on_list_od_classes_changed)
        #self.button_od_img_original = QtWidgets.QPushButton(self.groupbox_od_options)
        #self.button_od_img_original.setGeometry(QtCore.QRect(305, 30, 55, 20))
        #self.button_od_img_original.setFocusPolicy(QtCore.Qt.StrongFocus)
        #self.button_od_img_original.setObjectName("button_od_img_original")
        #self.button_od_img_original.clicked.connect(self.on_button_img_original)
        #self.button_od_img_original.setVisible(False)
        self.button_od_img_boxes = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_img_boxes.setGeometry(QtCore.QRect(305, 30, 55, 20))
        self.button_od_img_boxes.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.button_od_img_boxes.setObjectName("button_od_img_boxes")
        self.button_od_img_boxes.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_img_boxes.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_img_boxes.clicked.connect(self.on_button_od_img_boxes)
        self.button_od_img_boxes.setVisible(False)
        self.button_od_img_class_labels = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_img_class_labels.setGeometry(QtCore.QRect(365, 30, 55, 20))
        self.button_od_img_class_labels.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.button_od_img_class_labels.setObjectName("button_od_img_class_labels")
        self.button_od_img_class_labels.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_img_class_labels.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_img_class_labels.clicked.connect(self.on_button_od_img_classes)
        self.button_od_img_class_labels.setVisible(False)
        self.button_od_img_masks = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_img_masks.setGeometry(QtCore.QRect(305, 55, 55, 20))
        self.button_od_img_masks.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.button_od_img_masks.setObjectName("button_od_img_masks")
        self.button_od_img_masks.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_img_masks.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_img_masks.clicked.connect(self.on_button_od_img_masks)
        self.button_od_img_masks.setVisible(False)
        #self.button_od_img_all = QtWidgets.QPushButton(self.groupbox_od_options)
        #self.button_od_img_all.setGeometry(QtCore.QRect(305, 80, 55, 20))
        #self.button_od_img_all.setFocusPolicy(QtCore.Qt.StrongFocus)
        #self.button_od_img_all.setObjectName("button_od_img_all")
        #self.button_od_img_all.clicked.connect(self.on_button_img_all)
        #self.button_od_img_all.setVisible(False)
        #self.button_od_change_mode = QtWidgets.QPushButton(self.groupbox_od_options)
        """self.button_od_change_mode.setGeometry(QtCore.QRect(365, 55, 55, 20))
        self.button_od_change_mode.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.button_od_change_mode.setObjectName("button_od_change_mode")
        self.button_od_change_mode.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_change_mode.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_change_mode.clicked.connect(self.on_button_change_mode)
        self.button_od_change_mode.setVisible(False)"""
        self.combobox_od_results_mode = QtWidgets.QComboBox(self.groupbox_od_options)
        self.combobox_od_results_mode.setGeometry(QtCore.QRect(365, 55, 55, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.combobox_od_results_mode.setFont(font)
        self.combobox_od_results_mode.setStyleSheet(
            "QComboBox {color: #cccccc; background-color: #353535; selection-background-color: #353535; border-radius: 10px} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.combobox_od_results_mode.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.combobox_od_results_mode.setObjectName("combobox_od_results_mode")
        self.combobox_od_results_mode.setToolTip("Choose the results display mode.")
        self.combobox_od_results_mode.setVisible(False)
        model = self.combobox_od_results_mode.model()
        for i in range(len(SSVII_GUI.RESULT_MODES_AVAILABLE)):
            item = QtGui.QStandardItem(SSVII_GUI.RESULT_MODES_AVAILABLE[i])
            item.setForeground(QtGui.QColor(204, 204, 204))
            item.setBackground(QtGui.QColor(53, 53, 53))
            model.appendRow(item)
        self.combobox_od_results_mode.activated[str].connect(self.on_combobox_results_mode_change)
        self.combobox_od_sort_mode = QtWidgets.QComboBox(self.groupbox_od_options)
        self.combobox_od_sort_mode.setGeometry(QtCore.QRect(305, 80, 115, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.combobox_od_sort_mode.setFont(font)
        self.combobox_od_sort_mode.setStyleSheet(
            "QComboBox {color: #cccccc; background-color: #353535; selection-background-color: #353535; border-radius: 10px} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.combobox_od_sort_mode.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.combobox_od_sort_mode.setObjectName("combobox_fm_results_mode")
        self.combobox_od_sort_mode.setToolTip("Choose the results display mode.")
        self.combobox_od_sort_mode.setVisible(False)
        model = self.combobox_od_sort_mode.model()
        for i in range(len(SSVII_GUI.SORT_MODES_AVAILABLE["Object Detection"])):
            item = QtGui.QStandardItem(SSVII_GUI.SORT_MODES_AVAILABLE["Object Detection"][i])
            item.setForeground(QtGui.QColor(204, 204, 204))
            item.setBackground(QtGui.QColor(53, 53, 53))
            model.appendRow(item)
        self.combobox_od_sort_mode.activated[str].connect(self.on_combobox_sort_mode_changed)
        self.button_od_prev_result = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_prev_result.setGeometry(QtCore.QRect(327, 115, 22, 20))
        self.button_od_prev_result.setObjectName("button_od_prev_result")
        self.button_od_prev_result.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_prev_result.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_prev_result.setVisible(False)
        self.button_od_prev_result.setIcon(QtGui.QIcon("./resources/previous_icon.png"))
        self.button_od_prev_result.clicked.connect(self.on_button_prev_result)
        self.button_od_next_result = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_next_result.setGeometry(QtCore.QRect(375, 115, 22, 20))
        self.button_od_next_result.setObjectName("button_od_next_result")
        self.button_od_next_result.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_next_result.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_next_result.setVisible(False)
        self.button_od_next_result.setIcon(QtGui.QIcon("./resources/next_icon.png"))
        self.button_od_next_result.clicked.connect(self.on_button_next_result)
        self.button_od_animate = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_animate.setGeometry(QtCore.QRect(351, 115, 22, 20))
        self.button_od_animate.setObjectName("button_od_animate")
        self.button_od_animate.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_animate.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_animate.setVisible(False)
        self.button_od_animate.setIcon(QtGui.QIcon("./resources/play_icon.png"))
        self.button_od_animate.clicked.connect(self.on_button_animate)
        self.button_od_speed_down = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_speed_down.setGeometry(QtCore.QRect(327, 140, 33, 20))
        self.button_od_speed_down.setObjectName("button_od_speed_down")
        self.button_od_speed_down.setToolTip("Decrease playback frame rate by 10 fps (min. 1).")
        self.button_od_speed_down.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_speed_down.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_speed_down.setVisible(False)
        self.button_od_speed_down.setDisabled(True)
        self.button_od_speed_down.setIcon(QtGui.QIcon("./resources/speed_down_icon.png"))
        self.button_od_speed_down.clicked.connect(self.on_button_speed_down)
        self.button_od_speed_up = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_speed_up.setGeometry(QtCore.QRect(364, 140, 33, 20))
        self.button_od_speed_up.setObjectName("button_od_speed_up")
        self.button_od_speed_up.setToolTip("Increase playback frame rate by 10 fps (max. 60).")
        self.button_od_speed_up.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 5px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_speed_up.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_speed_up.setVisible(False)
        self.button_od_speed_up.setDisabled(True)
        self.button_od_speed_up.setIcon(QtGui.QIcon("./resources/speed_up_icon.png"))
        self.button_od_speed_up.clicked.connect(self.on_button_speed_up)
        self.button_od_process = QtWidgets.QPushButton(self.groupbox_od_options)
        self.button_od_process.setGeometry(QtCore.QRect(327, 170, 70, 20))
        self.button_od_process.setObjectName("button_od_process")
        self.button_od_process.setStyleSheet("QPushButton {color: #cccccc; background-color: #353535; border-radius: 10px} QPushButton:hover {background-color: #454545} QPushButton:pressed {background-color: #555555}")
        self.button_od_process.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=5, xOffset=0, yOffset=0))
        self.button_od_process.setVisible(False)
        self.button_od_process.clicked.connect(self.on_button_process)
        self.progress_bar_od = QtWidgets.QProgressBar(self.groupbox_od_options)
        self.progress_bar_od.setGeometry(QtCore.QRect(327, 170, 70, 20))
        self.progress_bar_od.setObjectName("progress_bar_od")
        self.progress_bar_od.setMinimum(0)
        self.progress_bar_od.setValue(0)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.progress_bar_od.setFont(font)
        self.progress_bar_od.setStyleSheet("QProgressBar { background-color: #707070; border: 2px solid grey; border-radius: 2px; text-align: center; }QProgressBar::chunk { background-color: #505050; }")
        self.progress_bar_od.setVisible(False)
        # endregion

        # region Oject Detection Results
        self.groupbox_od_results = QtWidgets.QGroupBox(self.centralwidget)
        self.groupbox_od_results.setGeometry(QtCore.QRect(10, 265, 920, 520))
        self.groupbox_od_results.setStyleSheet("QGroupBox {color: #cccccc; background-color: #1e1e1e; border-radius: 10px} QGroupBox:title {padding-left: 7px; padding-top: 5px}")
        self.groupbox_od_results.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=15, xOffset=0, yOffset=0))
        self.groupbox_od_results.setObjectName("groupbox_od_results")
        self.groupbox_od_results.setVisible(False)
        self.label_od_result_img = QtWidgets.QLabel(self.groupbox_od_results)
        self.label_od_result_img.setGeometry(QtCore.QRect(12, 28, 897, 481))
        self.label_od_result_img.setText("")
        #self.label_od_result_img.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.label_od_result_img.setFrameStyle(QtWidgets.QFrame.NoFrame)
        # self.label_od_result_img.setPixmap(QtGui.QPixmap("./resources/image_placeholder.png"))
        self.label_od_result_img.setScaledContents(False)
        self.label_od_result_img.setAlignment(QtCore.Qt.AlignCenter)
        self.label_od_result_img.setObjectName("label_od_result_img")
        self.label_od_result_img.setToolTip("Click to enlarge.")
        self.label_od_result_img.setStyleSheet("QLabel {background-color: transparent} QToolTip { background-color: rgb(40, 40, 40); color: #cccccc; border: black solid 1px }")
        self.label_od_result_img.mouseReleaseEvent = self.on_popup_result_img
        self.label_od_result_img.setVisible(True)
        self.groupbox_od_advanced_results = QtWidgets.QGroupBox(self.groupbox_od_results)
        self.groupbox_od_advanced_results.setVisible(False)
        self.groupbox_od_advanced_results.setGeometry(QtCore.QRect(12, 28, 897, 481))
        self.groupbox_od_advanced_results.setObjectName("groupbox_advanced_results")
        self.groupbox_od_advanced_results.setStyleSheet("background-color: #353535")
        self.groupbox_od_advanced_results.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(color=QtGui.QColor(0, 0, 0), blurRadius=15, xOffset=0, yOffset=0))
        self.graphics_view_od_scene = QtWidgets.QGraphicsScene()
        self.graphics_view_od = QtWidgets.QGraphicsView(self.graphics_view_od_scene,
                                                        self.groupbox_od_advanced_results)
        self.graphics_view_od.setGeometry(0, 0, 897, 481)
        self.graphics_view_od.setVisible(True)
        self.graphics_view_od.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.graphics_view_od.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphics_view_od.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.graphics_view_od.setStyleSheet(
            "QGraphicsView {background-color: transparent} QToolTip{ background-color: #1e1e1e; color: #cccccc; border: black solid 1px }")
        self.scroll_bar_graphics_view_od = QtWidgets.QScrollBar()
        self.scroll_bar_graphics_view_od.setStyleSheet("""
                                                         QScrollBar:vertical { background-color: #353535; width: 10px; margin: 3px 0px 3px 0px; border-radius: 4px; }
                                                         QScrollBar::handle:vertical { background-color: #121212; min-height: 5px;border-radius: 4px; }
                                                         QScrollBar::sub-line:vertical { border-image: url(:/); }
                                                         QScrollBar::add-line:vertical { border-image: url(:/); }
                                                         QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
                                                         """)
        self.graphics_view_od.setVerticalScrollBar(self.scroll_bar_graphics_view_od)
        """self.graphics_view_od_grid_2_scene = QtWidgets.QGraphicsScene()
        self.graphics_view_od_grid_2 = QtWidgets.QGraphicsView(self.graphics_view_od_grid_2_scene, self.groupbox_od_advanced_results)
        self.graphics_view_od_grid_2.setGeometry(0, 0, 897, 481)
        self.graphics_view_od_grid_2.setVisible(True)
        self.graphics_view_od_grid_2.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.graphics_view_od_grid_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphics_view_od_grid_2.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.graphics_view_od_grid_2.setStyleSheet("QGraphicsView {background-color: transparent} QToolTip{ background-color: #1e1e1e; color: #cccccc; border: black solid 1px }")
        self.scroll_bar_graphics_view_od_grid_2 = QtWidgets.QScrollBar()"""
        #self.scroll_bar_graphics_view_od_grid_2.setStyleSheet("""
        #                                         QScrollBar:vertical { background-color: #353535; width: 10px; margin: 3px 0px 3px 0px; border-radius: 4px; }
        #                                         QScrollBar::handle:vertical { background-color: #121212; min-height: 5px;border-radius: 4px; }
        #                                         QScrollBar::sub-line:vertical { border-image: url(:/); }
        #                                         QScrollBar::add-line:vertical { border-image: url(:/); }
        #                                         QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        #                                         """)
        #self.graphics_view_od_grid_2.setVerticalScrollBar(self.scroll_bar_graphics_view_od_grid_2)
        # endregion

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_title_ssvii.setText(_translate("MainWindow", "SSVII"))
        self.label_description.setText(_translate("MainWindow", "Search for Similar Visual Information in Images"))
        self.label_footer.setText(_translate("MainWindow",
                                             "Universidade da Madeira           Masters in Informatics Engineering           Developed by Diogo Cruz"))
        self.groupbox_file_selection.setTitle(_translate("MainWindow", "Selection"))
        self.button_browse_reference_img.setText(_translate("MainWindow", "Browse"))
        self.label_title_reference_img_path.setText(_translate("MainWindow", "Image for Reference"))
        self.label_title_analysis_imgs_path.setText(_translate("MainWindow", "Images for Analysis"))
        self.button_browse_analysis_imgs.setText(_translate("MainWindow", "Browse"))
        self.label_title_method.setText(_translate("MainWindow", "Method"))
        self.groupbox_reference.setTitle(_translate("MainWindow", "Reference"))
        self.label_title_reference_num_keypoints.setText(_translate("MainWindow", "# Keypoints"))
        self.label_reference_num_keypoints.setText(_translate("MainWindow", "2676"))
        self.button_select_region.setText(_translate("MainWindow", "Select region"))
        self.button_process.setText(_translate("MainWindow", "Process"))
        self.progress_bar.setFormat(_translate("MainWindow", ""))
        self.label_title_reference_num_descriptors.setText(_translate("MainWindow", "# Descriptors"))
        self.label_reference_num_descriptors.setText(_translate("MainWindow", "384"))
        self.groupbox_fm_results.setTitle(_translate("MainWindow", "Results"))
        self.groupbox_fm_advanced_results.setTitle(_translate("MainWindow", ""))
        self.button_img_fm_keypoints.setText(_translate("MainWindow", "Keypoints"))
        self.button_img_fm_matches.setText(_translate("MainWindow", "Matches"))
        # self.button_prev_result.setText(_translate("MainWindow", "Previous"))
        # self.button_next_result.setText(_translate("MainWindow", "Next"))
        self.label_title_result_relevance.setText(_translate("MainWindow", "Relevance"))
        self.label_result_relevance.setText(_translate("MainWindow", "0.85"))
        self.label_result_num_keypoints.setText(_translate("MainWindow", "1295"))
        self.label_title_result_num_keypoints.setText(_translate("MainWindow", "# Keypoints"))
        self.button_fm_img_bounding_box.setText(_translate("MainWindow", "Outline"))
        self.label_result_num_matches.setText(_translate("MainWindow", "384"))
        self.label_title_result_num_matches.setText(_translate("MainWindow", "# Matches"))
        self.button_img_fm_circle_prediction.setText(_translate("MainWindow", "Avg Point"))
        # self.label_title_display.setText(_translate("MainWindow", "Display"))
        #self.button_change_mode.setText(_translate("MainWindow", "Advanced"))
        self.label_title_result_num_descriptors.setText(_translate("MainWindow", "# Descriptors"))
        self.label_result_num_descriptors.setText(_translate("MainWindow", "384"))
        self.label_result_filename.setText(_translate("MainWindow", ""))
        self.label_title_result_filename.setText(_translate("MainWindow", "Filename"))

        self.label_od_options_classes.setText(_translate("MainWindow", "Objects to detect:"))
        self.groupbox_od_options.setTitle(_translate("MainWindow", "Options"))
        self.groupbox_od_results.setTitle(_translate("MainWindow", "Results"))

        self.label_title_gpu.setText(_translate("MainWindow", "GPU:"))
        self.label_title_disk.setText(_translate("MainWindow", "Disk:"))
        #self.button_transfer.setText(_translate("MainWindow", "Transfer"))

        #self.button_od_img_original.setText(_translate("MainWindow", "Original"))
        self.button_od_img_boxes.setText(_translate("MainWindow", "Boxes"))
        self.button_od_img_class_labels.setText(_translate("MainWindow", "Labels"))
        self.button_od_img_masks.setText(_translate("MainWindow", "Masks"))
        #self.button_od_img_all.setText(_translate("MainWindow", "All"))
        #self.button_od_change_mode.setText(_translate("MainWindow", "Advanced"))
        self.label_od_title_filename.setText(_translate("MainWindow", "Filename"))
        self.label_od_filename.setText(_translate("MainWindow", "-"))

        self.label_od_all_stats.setText(_translate("MainWindow", "All Results:"))
        self.label_od_title_all_avg_conf.setText(_translate("MainWindow", "Avg. Confidence"))
        self.label_od_all_avg_conf.setText(_translate("MainWindow", "-"))
        self.label_od_title_all_num_classes.setText(_translate("MainWindow", "Avg. Classes"))
        self.label_od_all_num_classes.setText(_translate("MainWindow", "-"))
        self.label_od_title_all_num_detections.setText(_translate("MainWindow", "Avg. Detections"))
        self.label_od_all_num_detections.setText(_translate("MainWindow", "-"))
        self.label_od_single_stats.setText(_translate("MainWindow", "Current Result:"))
        self.label_od_title_avg_conf.setText(_translate("MainWindow", "Avg. Confidence"))
        self.label_od_avg_conf.setText(_translate("MainWindow", "-"))
        self.label_od_title_num_classes.setText(_translate("MainWindow", "# Classes"))
        self.label_od_num_classes.setText(_translate("MainWindow", "-"))
        self.label_od_title_num_detections.setText(_translate("MainWindow", "# Detections"))
        self.label_od_num_detections.setText(_translate("MainWindow", "-"))
        self.button_od_process.setText(_translate("MainWindow", "Process"))

    # endregion

    # region Handle displayed images
    def display_reference(self):
        """
        Updates the reference image and relevant information displayed
        """

        # Get the reference original image in RGB
        reference = self.ref_manager.get_reference()
        ref_img_original = cv2.cvtColor(reference.get_img_original(), cv2.COLOR_BGR2RGB)
        img = self.resize_img(ref_img_original, self.label_reference_img.width(), self.label_reference_img.height())

        # Create QImage from cv2 reference img
        qimg = QtGui.QImage(img.data,
                            img.shape[1],
                            img.shape[0],
                            QtGui.QImage.Format_RGB888)

        # Create pixmap from QImage and scale it to fit the QLabel while keeping the aspect ratio
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(self.label_reference_img.width(),
                                                      self.label_reference_img.height(),
                                                      QtCore.Qt.KeepAspectRatio,
                                                      QtCore.Qt.SmoothTransformation)

        # Update image QLabel
        self.label_reference_img.setPixmap(pixmap)

        # Update ToolTip
        img_width = int(SSVII_GUI.MAIN_WINDOW_WIDTH * 0.6)
        img_height = int(SSVII_GUI.MAIN_WINDOW_HEIGHT * 0.6)
        ratio = ref_img_original.shape[1] / ref_img_original.shape[0]

        if int(img_width / ratio) < img_height:
            img_height = int(img_width / ratio)

        resized = self.resize_img(ref_img_original, img_width, img_height)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        import base64
        path = reference.get_path()
        format = path.split('.')[-1]
        _, im_arr = cv2.imencode("." + format, resized)  # im_arr: image in Numpy one-dim array format.
        b64 = str(base64.b64encode(im_arr.tobytes()))[2:]

        self.label_reference_img.setToolTip('<h3 style="color: #cccccc">Path: {0}</><br><br><img src="data:image/{1};base64,{2}"></>'.format(reference.get_path(), format, b64))

        # Update relevant information
        kps = reference.get_keypoints()
        des = reference.get_descriptors()

        if kps is None or des is None:
            return

        self.label_reference_num_keypoints.setText(str(len(kps)))
        self.label_reference_num_descriptors.setText(str(len(des)))

    def update_single_result(self):
        """
        Displays the current result, updating the image and information labels
        """

        # Get the current result
        result = self.res_manager.get_result_at_index(self.current_result_index)
        label = None

        # Update image QLabel
        # Get the result image that corresponds to the current mode
        img = self.get_result_img(result)
        if img is None:
            return

        # Get the label to update
        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            label = self.label_result_img
        elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
            label = self.label_od_result_img

        # If we have an image then we can display it
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # QLabel works with RGB
        img = self.resize_img(img, label.width(), label.height())

        image = QtGui.QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap(image))

        # Update the other GUI elements
        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            self.label_result_relevance.setText("{0}%".format(round(self.res_manager.get_relevance(result), 2)))
            self.label_result_num_keypoints.setText(str(len(self.res_manager.get_keypoints(result))))
            self.label_result_num_descriptors.setText(str(len(self.res_manager.get_descriptors(result))))
            self.label_result_num_matches.setText(str(len(self.res_manager.get_matches(result))))
            self.label_result_filename.setText(self.res_manager.get_original_path(result))
            self.label_result_filename.setToolTip(self.res_manager.get_original_path(result))
        elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE and self.current_results_mode == SSVII_GUI.RESULTS_MODE_SINGLE:
            self.label_od_avg_conf.setText("{0}%".format(round(self.res_manager.get_avg_confidence(result), 2)))
            self.label_od_num_classes.setText(str(self.res_manager.get_num_classes(result)))
            self.label_od_num_detections.setText(str(self.res_manager.get_num_detections(result)))
            self.label_od_filename.setText(str(self.res_manager.get_original_path(result)))
            self.label_od_filename.setToolTip(self.res_manager.get_original_path(result))

    def animate_results(self):
        if SSVII_GUI.ANIMATE_RESULTS_MODE:
            self.on_button_next_result()
            QtCore.QTimer.singleShot((1 / SSVII_GUI.ANIMATION_FRAME_RATE) * 1000, lambda: self.animate_results())

    def update_multi_results_display_modes(self, rebuild=True):
        """
        Updates the grid layout results to be displayed (in advanced mode)
        """

        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            results = list(filter(lambda result: (self.res_manager.get_relevance(result) > 10), self.res_manager.get_results()))
            scene = self.graphics_view_fm_scene
            graphics_view = self.graphics_view_fm
            if self.current_results_mode == SSVII_GUI.RESULTS_MODE_GRID_1:
                num_columns = 4
            elif self.current_results_mode == SSVII_GUI.RESULTS_MODE_GRID_2:
                num_columns = 2
                max_num_columns = 6
        elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
            if "ENet" in self.proc_manager.get_active_method().get_name():
                results = self.res_manager.get_results()
            else:
                results = list(filter(lambda result: (self.res_manager.get_avg_confidence(result) > 30), self.res_manager.get_results()))
            scene = self.graphics_view_od_scene
            graphics_view = self.graphics_view_od
            if self.current_results_mode == SSVII_GUI.RESULTS_MODE_GRID_1:
                num_columns = 6
            elif self.current_results_mode == SSVII_GUI.RESULTS_MODE_GRID_2:
                num_columns = 2
                max_num_columns = 6

        if len(results) == 0:
            return

        if not rebuild:
            if self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == SSVII_GUI.RESULTS_MODE_SPIRAL:
                results.reverse()
            for i in range(len(scene.items())):
                item = scene.items()[len(scene.items()) - 1 - i]

                if item.data(0) is not None:
                    #item.setData(0, self.res_manager.get_id(result))
                    result = self.res_manager.get_result_by_id(item.data(0))
                else:
                    result = results[i]

                img = self.get_result_img(result)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Update pixmap
                label_w = item.pixmap().width()
                label_h = item.pixmap().height()
                resized_img = self.resize_img(img, label_w, label_h)
                qimg = QtGui.QImage(resized_img, resized_img.shape[1], resized_img.shape[0], resized_img.shape[1] * 3,
                                    QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                item.setPixmap(pixmap)

                # Update tooltip
                img_width = int(SSVII_GUI.MAIN_WINDOW_WIDTH * 0.5)
                img_height = int(SSVII_GUI.MAIN_WINDOW_HEIGHT * 0.5)
                ratio = img.shape[1] / img.shape[0]

                if int(img_width / ratio) < img_height:
                    img_height = int(img_width / ratio)

                resized_img = self.resize_img(img, img_width, img_height)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                path = result.get_original_path()
                file_format = path.split('.')[-1]
                _, im_arr = cv2.imencode("." + file_format, resized_img)  # im_arr: image in Numpy one-dim array format.
                b64 = str(base64.b64encode(im_arr.tobytes()))[2:]
                img_tag = '<img src="data:image/{1};base64,{2}"></>'.format(path, file_format, b64)

                if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Relevance: {0}%</><h3 style="color: #cccccc">Keypoints/Descriptors: {1}</><h3 style="color: #cccccc">Matches: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_relevance(result), 2), len(self.res_manager.get_keypoints(result)),
                        len(self.res_manager.get_matches(result)), self.res_manager.get_original_path(result), img_tag))
                elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Avg. Confidence: {0}%</><h3 style="color: #cccccc">Number of Classes: {1}</><h3 style="color: #cccccc">Number of Detections: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_avg_confidence(result), 2),
                        self.res_manager.get_num_classes(result),
                        self.res_manager.get_num_detections(result), self.res_manager.get_original_path(result),
                        img_tag))
            return

        # Get the current results to display them in the grid
        row = 0
        column = 0
        scene.clear()
        graphics_view.resetTransform()

        def wheel_moved(event):
            if self.current_results_mode != SSVII_GUI.RESULTS_MODE_SPIRAL:
                QtWidgets.QGraphicsView.wheelEvent(graphics_view, event)
                event.accept()
                return

            # Get the old position
            old_pos = graphics_view.mapToScene(event.pos())

            # Scale
            if event.angleDelta().y() > 0 and self.current_graphics_view_scale_step < 6:
                scale_factor = 1.1
                self.current_graphics_view_scale_step += 1
            elif event.angleDelta().y() <= 0 and self.current_graphics_view_scale_step > -46:
                scale_factor = 1/1.1
                self.current_graphics_view_scale_step -= 1
            else:
                return

            graphics_view.scale(scale_factor, scale_factor)

            # Get the new position
            new_pos = graphics_view.mapToScene(event.pos())

            # Move scene to old position
            delta = new_pos - old_pos
            graphics_view.translate(delta.x(), delta.y())
            event.accept()

        graphics_view.wheelEvent = wheel_moved
        graphics_view.verticalScrollBar().setSliderPosition(1)
        self.current_graphics_view_scale_step = 0

        if self.current_results_mode == SSVII_GUI.RESULTS_MODE_GRID_1:
            margin = 5
            label_w = int((graphics_view.width() - (margin * (num_columns + 1))) / num_columns)
            label_h = int(label_w * 0.77)
            for result in results:
                img = self.get_result_img(result)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Update pixmap
                resized_img = self.resize_img(img, label_w, label_h)
                qimg = QtGui.QImage(resized_img, resized_img.shape[1], resized_img.shape[0], resized_img.shape[1] * 3,
                                    QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                item = QtWidgets.QGraphicsPixmapItem()
                item.setPos(QtCore.QPointF(column * label_w + column * margin + margin, row * label_h + row * margin + margin))
                item.setPixmap(pixmap)

                # Update tooltip
                img_width = int(SSVII_GUI.MAIN_WINDOW_WIDTH * 0.5)
                img_height = int(SSVII_GUI.MAIN_WINDOW_HEIGHT * 0.5)
                ratio = img.shape[1] / img.shape[0]

                if int(img_width / ratio) < img_height:
                    img_height = int(img_width / ratio)

                resized_img = self.resize_img(img, img_width, img_height)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                path = result.get_original_path()
                file_format = path.split('.')[-1]
                _, im_arr = cv2.imencode("." + file_format, resized_img)  # im_arr: image in Numpy one-dim array format.
                b64 = str(base64.b64encode(im_arr.tobytes()))[2:]
                img_tag = '<img src="data:image/{1};base64,{2}"></>'.format(path, file_format, b64)

                if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Relevance: {0}%</><h3 style="color: #cccccc">Keypoints/Descriptors: {1}</><h3 style="color: #cccccc">Matches: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_relevance(result), 2), len(self.res_manager.get_keypoints(result)),
                        len(self.res_manager.get_matches(result)), self.res_manager.get_original_path(result), img_tag))
                elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Avg. Confidence: {0}%</><h3 style="color: #cccccc">Number of Classes: {1}</><h3 style="color: #cccccc">Number of Detections: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_avg_confidence(result), 2),
                        self.res_manager.get_num_classes(result),
                        self.res_manager.get_num_detections(result), self.res_manager.get_original_path(result),
                        img_tag))

                # Add item
                scene.addItem(item)

                # Update the row/column we're at
                if column >= num_columns - 1:
                    column = 0
                    row += 1
                else:
                    column += 1
            scene.setSceneRect(scene.itemsBoundingRect())
        elif self.current_results_mode == SSVII_GUI.RESULTS_MODE_GRID_2:
            num_columns = 2
            margin = 5
            label_w = int((graphics_view.width() - (margin * (num_columns + 1))) / num_columns)
            label_h = label_w
            yoffset = 0
            for result in results:
                img = self.get_result_img(result)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Update pixmap
                resized_img = self.resize_img(img, label_w, label_h)
                qimg = QtGui.QImage(resized_img, resized_img.shape[1], resized_img.shape[0], resized_img.shape[1] * 3, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                item = QtWidgets.QGraphicsPixmapItem()
                item.setPos(QtCore.QPointF(column * label_w + column * margin + margin, yoffset + row * margin + margin))
                item.setPixmap(pixmap)

                # Update tooltip
                img_width = int(SSVII_GUI.MAIN_WINDOW_WIDTH * 0.5)
                img_height = int(SSVII_GUI.MAIN_WINDOW_HEIGHT * 0.5)
                ratio = img.shape[1] / img.shape[0]

                if int(img_width / ratio) < img_height:
                    img_height = int(img_width / ratio)

                resized_img = self.resize_img(img, img_width, img_height)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                path = result.get_original_path()
                file_format = path.split('.')[-1]
                _, im_arr = cv2.imencode("." + file_format, resized_img)  # im_arr: image in Numpy one-dim array format.
                b64 = str(base64.b64encode(im_arr.tobytes()))[2:]
                img_tag = '<img src="data:image/{1};base64,{2}"></>'.format(path, file_format, b64)

                if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
                    item.setToolTip(str('<h3 style="color: #cccccc">Relevance: {0}%</><h3 style="color: #cccccc">Keypoints/Descriptors: {1}</><h3 style="color: #cccccc">Matches: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(round(self.res_manager.get_relevance(result), 2), len(self.res_manager.get_keypoints(result)), len(self.res_manager.get_matches(result)), self.res_manager.get_original_path(result), img_tag))
                elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Avg. Confidence: {0}%</><h3 style="color: #cccccc">Number of Classes: {1}</><h3 style="color: #cccccc">Number of Detections: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_avg_confidence(result), 2),
                        self.res_manager.get_num_classes(result),
                        self.res_manager.get_num_detections(result), self.res_manager.get_original_path(result),
                        img_tag))

                # Add item
                scene.addItem(item)

                # Update the row/column we're at
                if column >= num_columns - 1:
                    column = 0
                    row += 1
                    yoffset += label_h
                    if num_columns < max_num_columns and row != 0 and row % 2 == 0:  # Increase number of labels per row every two rows
                        num_columns += 1
                        label_w = int((graphics_view.width() - (margin * (num_columns + 1))) / num_columns)
                        label_h = int(label_w * 0.77)
                else:
                    column += 1
            scene.setSceneRect(scene.itemsBoundingRect())
        elif self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE:
            for result in reversed(results):
                img = self.get_result_img(result)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Update pixmap
                label_w = int(graphics_view.width() / 4)
                label_h = int(label_w * 0.77)
                resized_img = self.resize_img(img, label_w, label_h, -1)
                qimg = QtGui.QImage(resized_img, resized_img.shape[1], resized_img.shape[0], resized_img.shape[1] * 3, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                item = QtWidgets.QGraphicsPixmapItem()
                pos_x = random.randint(0, graphics_view.width() - 10 - label_w)
                pos_y = random.randint(0, graphics_view.height() - 10 - label_h)
                item.setPos(QtCore.QPointF(pos_x, pos_y))
                item.setPixmap(pixmap)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
                item.setData(0, result.get_id())
                item.setData(1, 1.0)
                def mouse_pressed(event):
                    max_z = 0
                    for i in scene.items():
                        max_z = max(max_z, i.zValue())
                    item_pressed = scene.itemAt(event.scenePos(), graphics_view.transform())
                    item_pressed.setZValue(max_z + 1)
                    scene.update()
                def wheel_moved(event):
                    item_under = scene.itemAt(event.scenePos(), graphics_view.transform())
                    if item_under is None:
                        return
                    scale = item_under.data(1)
                    delta = event.delta()
                    delta = round(np.interp(delta, [-360, 360], [-1, 1]), 2)
                    if (delta < 0 and scale + delta <= 0.3) or (delta > 0 and scale + delta) >= 10:
                        return
                    scale += delta
                    item_under.setData(1, scale)
                    result = self.res_manager.get_result_by_id(item_under.data(0))
                    img = self.get_result_img(result)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Update pixmap
                    width = int(graphics_view.width() / 4 * scale)
                    height = int(width * 0.77)
                    resized_img = self.resize_img(img, width, height, -1)
                    qimg = QtGui.QImage(resized_img, resized_img.shape[1], resized_img.shape[0],
                                        resized_img.shape[1] * 3, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    offsetx = pixmap.width() - item_under.pixmap().width()
                    offsety = pixmap.height() - item_under.pixmap().height()
                    x = item_under.x() - offsetx / 2
                    y = item_under.y() - offsety / 2
                    item_under.setPos(x, y)
                    item_under.setPixmap(pixmap)
                    scene.update()
                item.mousePressEvent = mouse_pressed
                item.wheelEvent = wheel_moved

                # Update tooltip
                img_width = int(SSVII_GUI.MAIN_WINDOW_WIDTH * 0.5)
                img_height = int(SSVII_GUI.MAIN_WINDOW_HEIGHT * 0.5)
                ratio = img.shape[1] / img.shape[0]

                if int(img_width / ratio) < img_height:
                    img_height = int(img_width / ratio)

                resized_img = self.resize_img(img, img_width, img_height)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                path = result.get_original_path()
                file_format = path.split('.')[-1]
                _, im_arr = cv2.imencode("." + file_format, resized_img)  # im_arr: image in Numpy one-dim array format.
                b64 = str(base64.b64encode(im_arr.tobytes()))[2:]
                img_tag = '<img src="data:image/{1};base64,{2}"></>'.format(path, file_format, b64)

                if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Relevance: {0}%</><h3 style="color: #cccccc">Keypoints/Descriptors: {1}</><h3 style="color: #cccccc">Matches: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_relevance(result), 2), len(self.res_manager.get_keypoints(result)),
                        len(self.res_manager.get_matches(result)), self.res_manager.get_original_path(result), img_tag))
                elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Avg. Confidence: {0}%</><h3 style="color: #cccccc">Number of Classes: {1}</><h3 style="color: #cccccc">Number of Detections: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_avg_confidence(result), 2),
                        self.res_manager.get_num_classes(result),
                        self.res_manager.get_num_detections(result), self.res_manager.get_original_path(result),
                        img_tag))

                # Add item
                scene.addItem(item)
            scene.setSceneRect(scene.itemsBoundingRect())
        elif self.current_results_mode == SSVII_GUI.RESULTS_MODE_SPIRAL:
            scene.setSceneRect(0, 0, 80000, 80000)
            graphics_view.centerOn(40000, 40000)

            # Spiral variables
            # https://en.wikipedia.org/wiki/Archimedean_spiral
            c = 1
            v = 35
            w = 70
            t = 0
            num_results = 85  # Number of results to display
            width_increase = 64
            height_increase = 48
            label_w = int(graphics_view.width() / 4)  # Starting width
            label_h = int(graphics_view.width() / 4)  # Starting height

            for result in reversed(results[:num_results]):
                img = self.get_result_img(result)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Update pixmap
                resized_img = self.resize_img(img, label_w, label_h)
                qimg = QtGui.QImage(resized_img, resized_img.shape[1], resized_img.shape[0], resized_img.shape[1] * 3,
                                    QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg)
                item = QtWidgets.QGraphicsPixmapItem()
                scene_rect = scene.sceneRect()
                pos_x = (v * t + c) * np.cos(w * t) - label_w / 2 + scene_rect.width() / 2
                pos_y = (v * t + c) * np.sin(w * t) - label_h / 2 + scene_rect.height() / 2
                item.setPos(QtCore.QPointF(pos_x, pos_y))
                item.setPixmap(pixmap)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
                #item.setData(0, result.get_id())
                #item.setData(1, 1.0)

                # Update tooltip
                img_width = int(SSVII_GUI.MAIN_WINDOW_WIDTH * 0.5)
                img_height = int(SSVII_GUI.MAIN_WINDOW_HEIGHT * 0.5)
                ratio = img.shape[1] / img.shape[0]

                if int(img_width / ratio) < img_height:
                    img_height = int(img_width / ratio)

                resized_img = self.resize_img(img, img_width, img_height)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                path = result.get_original_path()
                file_format = path.split('.')[-1]
                _, im_arr = cv2.imencode("." + file_format, resized_img)  # im_arr: image in Numpy one-dim array format.
                b64 = str(base64.b64encode(im_arr.tobytes()))[2:]
                img_tag = '<img src="data:image/{1};base64,{2}"></>'.format(path, file_format, b64)

                if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Relevance: {0}%</><h3 style="color: #cccccc">Keypoints/Descriptors: {1}</><h3 style="color: #cccccc">Matches: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_relevance(result), 2), len(self.res_manager.get_keypoints(result)),
                        len(self.res_manager.get_matches(result)), self.res_manager.get_original_path(result), img_tag))
                elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
                    item.setToolTip(str(
                        '<h3 style="color: #cccccc">Avg. Confidence: {0}%</><h3 style="color: #cccccc">Number of Classes: {1}</><h3 style="color: #cccccc">Number of Detections: {2}</><h3 style="color: #cccccc">Path: {3}</><br><br>{4}').format(
                        round(self.res_manager.get_avg_confidence(result), 2),
                        self.res_manager.get_num_classes(result),
                        self.res_manager.get_num_detections(result), self.res_manager.get_original_path(result),
                        img_tag))

                # Add item
                scene.addItem(item)

                # Update values for next result
                label_w += width_increase
                label_h += height_increase
                t += 3.14 * 2

            # Center the qgraphicsview in the center of the the spiral
            center_item = scene.items()[-1]
            graphics_view.centerOn(QtCore.QPoint(center_item.pos().x() + center_item.pixmap().width() / 2, center_item.pos().y() + center_item.pixmap().height() / 2))

    def update_progress_bar(self, current):
        """
        Updates the progress bar. Called by the processing manager via callback
        :param current: Number of images processed
        :param total: Total number of images to be processed
        """

        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            progress_bar = self.progress_bar
        elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
            progress_bar = self.progress_bar_od
        progress_bar.setValue(current)
        progress_bar.setMaximum(len(self.res_manager.get_analysis()) * 3)
        progress_bar.setFormat("{}%".format(int(round(((current / 3) / len(self.res_manager.get_analysis())) * 100))))

    def update_visible_elements(self, update_info):
        """
        Updates the visible elements in the results groupbox depending on the current layout mode (normal or advanced).
        Called when the user changes the layout mode or tries to process
        """

        # If we just processed some images then we update some gui elements to display the new information
        if update_info:
            # Update the combobox item label so that the user always knows what method was last used to process
            for i in range(self.combobox_method.count()):
                self.combobox_method.setItemText(i, self.combobox_method.itemText(i).replace(' *', ''))
            cur_index = self.combobox_method.currentIndex()
            self.combobox_method.setItemText(cur_index, str(self.combobox_method.itemText(cur_index) + " *"))

            # Update the result displayed
            self.current_result_index = 0  # Reset the current result index
            self.update_single_result()  # Update the normal layout's GUI elements with the new current result

            # Hide the transfer button as the new data is saved in the correct place
            #self.button_transfer.setVisible(False)

        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            if update_info:
                # Reset everything related to other methods in case they were used to process before
                self.label_od_result_img.setVisible(False)
                self.groupbox_od_advanced_results.setVisible(False)
                #self.h_box_layout_od.setEnabled(False)
                self.label_od_all_stats.setVisible(False)
                self.label_od_single_stats.setVisible(False)
                self.label_od_title_all_avg_conf.setVisible(False)
                self.label_od_title_all_num_classes.setVisible(False)
                self.label_od_title_all_num_detections.setVisible(False)
                self.label_od_title_avg_conf.setVisible(False)
                self.label_od_title_num_classes.setVisible(False)
                self.label_od_title_num_detections.setVisible(False)
                self.label_od_title_filename.setVisible(False)
                self.label_od_all_avg_conf.setVisible(False)
                self.label_od_all_num_classes.setVisible(False)
                self.label_od_all_num_detections.setVisible(False)
                self.label_od_avg_conf.setVisible(False)
                self.label_od_num_classes.setVisible(False)
                self.label_od_num_detections.setVisible(False)
                self.label_od_filename.setVisible(False)
                #self.button_od_img_original.setVisible(False)
                self.button_od_img_boxes.setVisible(False)
                self.button_od_img_class_labels.setVisible(False)
                self.button_od_img_masks.setVisible(False)
                #self.button_od_img_all.setVisible(False)
                #self.button_od_change_mode.setVisible(False)
                self.combobox_od_results_mode.setVisible(False)
                self.combobox_od_sort_mode.setVisible(False)
                self.button_od_prev_result.setVisible(False)
                self.button_od_animate.setVisible(False)
                self.button_od_next_result.setVisible(False)
                self.button_od_speed_down.setVisible(False)
                self.button_od_speed_up.setVisible(False)

                # Update information
                self.label_title_reference_num_keypoints.setVisible(True)
                self.label_title_reference_num_descriptors.setVisible(True)
                self.label_reference_num_keypoints.setVisible(True)
                self.label_reference_num_descriptors.setVisible(True)
                reference = self.ref_manager.get_reference()
                self.label_reference_num_keypoints.setText(str(len(reference.get_keypoints())))
                self.label_reference_num_descriptors.setText(str(len(reference.get_descriptors())))

            # In normal mode, disable/hide the advanced mode GUI elements and enable/unhide the normal mode GUI elements
            if self.current_results_mode == self.RESULTS_MODE_SINGLE:
                #self.button_change_mode.setText("Grid 1")
                self.groupbox_fm_advanced_results.setVisible(False)
                #self.h_box_layout.setEnabled(False)
                self.button_img_fm_circle_prediction.setGeometry(QtCore.QRect(409, 15, 55, 20))
                self.button_fm_img_bounding_box.setGeometry(QtCore.QRect(469, 15, 55, 20))
                self.button_img_fm_keypoints.setGeometry(QtCore.QRect(409, 40, 55, 20))
                self.button_img_fm_matches.setGeometry(QtCore.QRect(469, 40, 55, 20))
                #self.button_change_mode.setGeometry(QtCore.QRect(529, 40, 55, 20))
                self.combobox_fm_results_mode.setGeometry(QtCore.QRect(529, 15, 55, 20))
                self.combobox_fm_sort_mode.setGeometry(QtCore.QRect(529, 40, 55, 20))
                self.label_result_img.setVisible(True)
                self.label_title_result_relevance.setVisible(True)
                self.label_title_result_num_keypoints.setVisible(True)
                self.label_title_result_num_descriptors.setVisible(True)
                self.label_title_result_num_matches.setVisible(True)
                self.label_title_result_filename.setVisible(True)
                self.label_result_relevance.setVisible(True)
                self.label_result_num_keypoints.setVisible(True)
                self.label_result_num_descriptors.setVisible(True)
                self.label_result_num_matches.setVisible(True)
                self.label_result_filename.setVisible(True)
                self.button_img_fm_circle_prediction.setVisible(True)
                self.button_fm_img_bounding_box.setVisible(True)
                self.button_img_fm_keypoints.setVisible(True)
                self.button_img_fm_matches.setVisible(True)
                #self.button_change_mode.setVisible(True)
                self.combobox_fm_results_mode.setVisible(True)
                self.combobox_fm_sort_mode.setVisible(True)
                if self.ANIMATE_RESULTS_MODE:
                    self.on_button_animate()
                self.button_prev_result.setVisible(True)
                self.button_animate.setVisible(True)
                self.button_next_result.setVisible(True)
                self.button_speed_down.setVisible(True)
                self.button_speed_up.setVisible(True)
            # In advanced mode, disable/hide the normal mode GUI elements and enable/unhide the advanced mode GUI elements
            elif self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == self.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
                self.graphics_view_fm.setVisible(True)
                #if self.current_results_mode == self.RESULTS_MODE_GRID_1:
                    #self.button_change_mode.setText("Grid 2")
                    #self.graphics_view_fm_grid_2.setVisible(False)
                #else:
                    #self.button_change_mode.setText("Single")
                    #self.graphics_view_fm_grid_1.setVisible(False)
                    #self.graphics_view_fm_grid_2.setVisible(True)
                self.update_multi_results_display_modes()
                self.groupbox_fm_advanced_results.setVisible(True)
                #self.h_box_layout.setEnabled(True)
                self.button_img_fm_circle_prediction.setGeometry(QtCore.QRect(20, 25, 55, 20))
                self.button_fm_img_bounding_box.setGeometry(QtCore.QRect(80, 25, 55, 20))
                self.button_img_fm_keypoints.setGeometry(QtCore.QRect(140, 25, 55, 20))
                self.button_img_fm_matches.setGeometry(QtCore.QRect(200, 25, 55, 20))
                #self.button_change_mode.setGeometry(QtCore.QRect(260, 25, 55, 20))
                self.combobox_fm_results_mode.setGeometry(QtCore.QRect(260, 25, 55, 20))
                self.combobox_fm_sort_mode.setGeometry(QtCore.QRect(320, 25, 55, 20))
                self.label_result_img.setVisible(False)
                self.label_title_result_relevance.setVisible(False)
                self.label_title_result_num_keypoints.setVisible(False)
                self.label_title_result_num_descriptors.setVisible(False)
                self.label_title_result_num_matches.setVisible(False)
                self.label_title_result_filename.setVisible(False)
                self.label_result_relevance.setVisible(False)
                self.label_result_num_keypoints.setVisible(False)
                self.label_result_num_descriptors.setVisible(False)
                self.label_result_num_matches.setVisible(False)
                self.label_result_filename.setVisible(False)
                self.button_img_fm_circle_prediction.setVisible(True)
                self.button_fm_img_bounding_box.setVisible(True)
                self.button_img_fm_keypoints.setVisible(True)
                self.button_img_fm_matches.setVisible(True)
                #self.button_change_mode.setVisible(True)
                self.combobox_fm_results_mode.setVisible(True)
                self.combobox_fm_sort_mode.setVisible(True)
                if self.ANIMATE_RESULTS_MODE:
                    self.on_button_animate()
                self.button_prev_result.setVisible(False)
                self.button_animate.setVisible(False)
                self.button_next_result.setVisible(False)
                self.button_speed_down.setVisible(False)
                self.button_speed_up.setVisible(False)
        elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
            self.label_od_all_stats.setVisible(True)
            self.label_od_single_stats.setVisible(True)
            self.label_od_title_all_avg_conf.setVisible(True)
            self.label_od_title_all_num_classes.setVisible(True)
            self.label_od_title_all_num_detections.setVisible(True)
            self.label_od_title_avg_conf.setVisible(True)
            self.label_od_title_num_classes.setVisible(True)
            self.label_od_title_num_detections.setVisible(True)
            self.label_od_title_filename.setVisible(True)
            self.label_od_all_avg_conf.setVisible(True)
            self.label_od_all_num_classes.setVisible(True)
            self.label_od_all_num_detections.setVisible(True)
            self.label_od_avg_conf.setVisible(True)
            self.label_od_num_classes.setVisible(True)
            self.label_od_num_detections.setVisible(True)
            self.label_od_filename.setVisible(True)
            #self.button_od_img_original.setVisible(True)
            self.button_od_img_boxes.setVisible(True)
            self.button_od_img_class_labels.setVisible(True)
            self.button_od_img_masks.setVisible(True)
            #self.button_od_img_all.setVisible(True)
            #self.button_od_change_mode.setVisible(True)
            self.combobox_od_results_mode.setVisible(True)
            self.combobox_od_sort_mode.setVisible(True)
            self.button_od_prev_result.setVisible(True)
            self.button_od_animate.setVisible(True)
            self.button_od_next_result.setVisible(True)
            self.button_od_speed_down.setVisible(True)
            self.button_od_speed_up.setVisible(True)
            if update_info:
                # Reset everything related to other methods in case they were used to process before
                self.label_title_reference_num_keypoints.setVisible(False)
                self.label_title_reference_num_descriptors.setVisible(False)
                self.label_reference_num_keypoints.setText("")
                self.label_reference_num_descriptors.setText("")
                self.label_result_img.setVisible(False)
                self.groupbox_fm_advanced_results.setVisible(False)
                #self.h_box_layout.setEnabled(False)
                self.label_title_result_relevance.setVisible(False)
                self.label_title_result_num_keypoints.setVisible(False)
                self.label_title_result_num_descriptors.setVisible(False)
                self.label_title_result_num_matches.setVisible(False)
                self.label_title_result_filename.setVisible(False)
                self.label_result_relevance.setVisible(False)
                self.label_result_num_keypoints.setVisible(False)
                self.label_result_num_descriptors.setVisible(False)
                self.label_result_num_matches.setVisible(False)
                self.label_result_filename.setVisible(False)
                self.button_img_fm_circle_prediction.setVisible(False)
                self.button_fm_img_bounding_box.setVisible(False)
                self.button_img_fm_keypoints.setVisible(False)
                self.button_img_fm_matches.setVisible(False)
                #self.button_change_mode.setVisible(False)
                self.combobox_fm_results_mode.setVisible(False)
                self.combobox_fm_sort_mode.setVisible(False)
                self.button_prev_result.setVisible(False)
                self.button_animate.setVisible(False)
                self.button_next_result.setVisible(False)
                self.button_speed_down.setVisible(False)
                self.button_speed_up.setVisible(False)

                # Update information
                results = self.res_manager.get_results()
                avg_conf_all = 0
                avg_num_classes = 0
                avg_num_detections = 0
                for result in results:
                    avg_conf_all += self.res_manager.get_avg_confidence(result)
                    avg_num_classes += self.res_manager.get_num_classes(result)
                    avg_num_detections += self.res_manager.get_num_detections(result)
                avg_conf_all /= len(results)
                avg_num_classes /= len(results)
                avg_num_detections /= len(results)
                self.label_od_all_avg_conf.setText("{0}%".format(round(avg_conf_all, 2)))
                self.label_od_all_num_classes.setText(str(round(avg_num_classes)))
                self.label_od_all_num_detections.setText(str(round(avg_num_detections)))

            if self.current_results_mode == self.RESULTS_MODE_SINGLE:
                #self.button_od_change_mode.setText("Grid 1")
                self.groupbox_od_advanced_results.setVisible(False)
                #self.h_box_layout_od.setEnabled(False)

                self.label_od_result_img.setVisible(True)
                result = self.res_manager.get_result_at_index(self.current_result_index)
                self.label_od_avg_conf.setText("{0}%".format(round(self.res_manager.get_avg_confidence(result), 2)))
                self.label_od_num_classes.setText(str(self.res_manager.get_num_classes(result)))
                self.label_od_num_detections.setText(str(self.res_manager.get_num_detections(result)))
                self.label_od_filename.setText(str(self.res_manager.get_original_path(result)))
                if self.ANIMATE_RESULTS_MODE:
                    self.on_button_animate()
                else:
                    self.button_od_prev_result.setDisabled(False)
                    self.button_od_animate.setDisabled(False)
                    self.button_od_next_result.setDisabled(False)
                    self.button_od_speed_down.setDisabled(True)
                    self.button_od_speed_up.setDisabled(True)
            elif self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == self.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
                self.graphics_view_od.setVisible(True)
                #if self.current_results_mode == self.RESULTS_MODE_GRID_1:
                    #self.button_od_change_mode.setText("Grid 2")
                    #self.graphics_view_od_grid_2.setVisible(False)
                #else:
                    #self.button_od_change_mode.setText("Single")
                    #self.graphics_view_od_grid_1.setVisible(False)
                    #self.graphics_view_od_grid_2.setVisible(True)
                self.update_multi_results_display_modes()
                self.groupbox_od_advanced_results.setVisible(True)
                #self.h_box_layout_od.setEnabled(True)
                self.label_od_result_img.setVisible(False)
                self.label_od_avg_conf.setText("-")
                self.label_od_num_classes.setText("-")
                self.label_od_num_detections.setText("-")
                self.label_od_filename.setText("-")
                if self.ANIMATE_RESULTS_MODE:
                    self.on_button_animate()
                self.button_od_prev_result.setDisabled(True)
                self.button_od_next_result.setDisabled(True)
                self.button_od_animate.setDisabled(True)
                self.button_od_speed_down.setDisabled(True)
                self.button_od_speed_up.setDisabled(True)


    def get_result_img(self, result=None):
        """
        Gets the image that corresponds to the current display mode from a result and returns it
        :param result: result to return the image from
        :return img: image from a result that corresponds to the current display mode
        """

        # If a result wasn't passed as a parameter, use the current result
        if result is None:
            result = self.res_manager.get_result_at_index(self.current_result_index)

        img = self.res_manager.get_img_original(result).copy()

        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            if SSVII_GUI.DISPLAY_FM_MATCHES:
                img = self.apply_overlay(img, self.res_manager.get_img_fm_matches(result))
            if SSVII_GUI.DISPLAY_FM_KEYPOINTS:
                img = self.apply_overlay(img, self.res_manager.get_img_fm_keypoints(result))
            if SSVII_GUI.DISPLAY_FM_CIRCLE_PREDICTION:
                img = self.apply_overlay(img, self.res_manager.get_img_fm_circle_prediction(result))
            if SSVII_GUI.DISPLAY_FM_BOUNDING_BOX:
                img = self.apply_overlay(img, self.res_manager.get_img_fm_bounding_box(result))
        elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
            a = SSVII_GUI.DISPLAY_OD_MASKS
            if SSVII_GUI.DISPLAY_OD_MASKS:
                img = self.apply_overlay(img, self.res_manager.get_img_od_masks(result))
            if SSVII_GUI.DISPLAY_OD_BOUNDING_BOXES:
                img = self.apply_overlay(img, self.res_manager.get_img_od_bounding_boxes(result))
            if SSVII_GUI.DISPLAY_OD_CLASSES:
                img = self.apply_overlay(img, self.res_manager.get_img_od_class_labels(result))

        return img

    def apply_overlay(self, original, overlay):
        if overlay is None:
            return original
        original = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
        if original.shape[0] > overlay.shape[0] or original.shape[1] > overlay.shape[1]:
            overlay = cv2.copyMakeBorder(src=overlay, top=0, bottom=original.shape[0] - overlay.shape[0], left=original.shape[1] - overlay.shape[1], right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
        elif original.shape[0] < overlay.shape[0] or original.shape[1] < overlay.shape[1]:
            original = cv2.copyMakeBorder(src=original, top=0, bottom=overlay.shape[0] - original.shape[0], left=overlay.shape[1] - original.shape[1], right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
        mask = overlay[:, :, 3] > 0

        original[mask] = overlay[mask]
        return original

    def resize_img(self, img, width, height, fill=30):
        """
        Resizes any given image so that it fits in any given frame size,
        keeping its aspect ratio and adding black bars if needed
        :param img: image to be resized
        :param width: width of the frame
        :param height: height of the frame
        :return: resized image with the original aspect ratio and possibly black bars
        """
        # Calculate the ratio
        ratio = img.shape[1] / img.shape[0]

        # If the target width is smaller than the target height then that width is the max width possible and we use it
        # as the new resized width
        if width < height:
            resized_width = width
            resized_height = int(resized_width / ratio)  # Calculate what height corresponds to the resized width
            # If the resized height is larger than the target height then we must use the target height as the new
            # reference (instead of the width)
            if resized_height > height:
                resized_height = height
                resized_width = int(resized_height * ratio)  # Calculate what width corresponds to the resized height
        else:
            resized_height = height
            resized_width = int(resized_height * ratio)
            if resized_width > width:
                resized_width = width
                resized_height = int(resized_width / ratio)

        # Resize the image to the new dimensions
        dim = (resized_width, resized_height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        if fill == -1:
            return resized_img

        # Create an empty image with the specified size
        new_img = np.zeros((height, width, img.shape[2]), dtype="uint8")
        new_img.fill(fill)

        # Copy resized image into the new one (label-sized). We do this so that the image label is always filled,
        # independently of the aspect ratio
        sx = int((width - resized_width) / 2)
        sy = int((height - resized_height) / 2)
        new_img[sy:sy + resized_img.shape[0], sx:sx + resized_img.shape[1]] = resized_img
        return new_img

    # endregion

    # region Handle button click events
    def on_open_ref_image(self):
        """
        Handles the process of browsing for a file for reference, delegates its processing and then displays it.
        Called when the user tries to browse for a reference image by pressing the open_ref_image button.
        """

        # Open a file dialog for the user to browse for files
        filename = QtWidgets.QFileDialog.getOpenFileName(self.main_window, 'Open File')

        # Delegate the reference set up process to the reference manager
        success = self.ref_manager.set_reference(filename[0])

        # If the selected file isn't of an accepted format, display an error message and abort
        if not success:
            msg = QtWidgets.QMessageBox()
            msg.setStyleSheet("QLabel{min-height: 30px;}")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowIcon(QtGui.QIcon("./resources/icon.png"))
            msg.setText("Couldn't open the image")
            msg.setInformativeText("The image should be in one of the following formats: png, jpg")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec()
            return

        # Try to create a pixmap using the opened file.
        # If it fails, the file isn't an image/usable and we display an error message and abort
        pixmap = QtGui.QPixmap(filename[0])
        if pixmap.isNull():
            msg = QtWidgets.QMessageBox()
            msg.setStyleSheet("QLabel{min-height: 30px;}")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowIcon(QtGui.QIcon("./resources/icon.png"))
            msg.setText("Couldn't open the image")
            msg.setInformativeText("The file must be a valid image")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec()
            return

        # If a reference has been correctly set up then we enable/unhide GUI elements in the reference groupbox
        if self.ref_manager.get_reference() is not None:
            self.groupbox_reference.setVisible(True)
            self.label_reference_img_path.setText(filename[0])
            self.label_reference_img_path.setToolTip(filename[0])  # Update the tooltip
            self.label_reference_num_keypoints.setText("")
            self.label_reference_num_descriptors.setText("")
            self.button_select_region.setVisible(True)
            self.button_process.setVisible(True)
            self.label_reference_img.setVisible(True)

        # Display the reference
        self.display_reference()

    def on_open_analysis_images(self):
        """
        Handles the process of browsing for files for analysis and delegates its processing.
        Called when the user tries to browse for analysis images by pressing the open_analysis_images button.
        """

        if SSVII_GUI.running_thread is not None:
            SSVII_GUI.running_thread.stop = True
            SSVII_GUI.running_thread.join()

        # Open a dialog so that the user can select a directory containing the images for analysis
        path = QtWidgets.QFileDialog.getExistingDirectory(self.main_window, "a", "C://",
                                                          QtWidgets.QFileDialog.ShowDirsOnly)

        class Worker(QtCore.QObject):
            sig_done = pyqtSignal(bool, str)

            def run(self, target, path, use_disk):
                success = target(path, use_disk)
                self.sig_done.emit(success, path)

        def finished(success, path):
            if success:
                if len(path) < 40:
                    self.label_analysis_imgs_path.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                else:
                    self.label_analysis_imgs_path.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.label_analysis_imgs_path.setText(path)  # Update the path
                self.label_analysis_imgs_path.setToolTip(path)  # Update the tooltip
                if self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
                    self.button_od_process.setVisible(True)
            SSVII_GUI.running_thread = None

        worker = Worker()
        worker.sig_done.connect(finished)
        thread = threading.Thread(target=worker.run, args=(self.res_manager.open_analysis, path, self.checkbox_disk.isChecked()))
        thread.daemon = True
        thread.stop = False
        SSVII_GUI.running_thread = thread
        thread.start()

    def on_combobox_method_change(self, method):
        """
        Called when the user changes the method in the combobox
        :param method: Selected item's text
        """

        self.proc_manager.set_active_method(method.replace(" *", ""))
        active_method = self.proc_manager.get_active_method()
        self.check_cuda_availability()
        SSVII_GUI.DISPLAY_FM_BOUNDING_BOX = False
        SSVII_GUI.DISPLAY_FM_CIRCLE_PREDICTION = False
        SSVII_GUI.DISPLAY_FM_KEYPOINTS = False
        SSVII_GUI.DISPLAY_FM_MATCHES = False
        SSVII_GUI.DISPLAY_OD_BOUNDING_BOXES = False
        SSVII_GUI.DISPLAY_OD_CLASSES = False
        SSVII_GUI.DISPLAY_OD_MASKS = False
        if "Feature Matching" in active_method.get_type():
            self.current_method_mode = SSVII_GUI.FEATURE_MATCHING_MODE
            self.button_browse_reference_img.setVisible(True)
            self.label_title_reference_img_path.setVisible(True)
            self.label_reference_img_path.setVisible(True)
            self.label_title_analysis_imgs_path.setGeometry(QtCore.QRect(20, 75, 121, 16))
            self.label_analysis_imgs_path.setGeometry(QtCore.QRect(20, 95, 200, 21))
            self.button_browse_analysis_imgs.setGeometry(QtCore.QRect(227, 95, 58, 21))
            self.label_title_method.setGeometry(QtCore.QRect(20, 125, 121, 16))
            self.combobox_method.setGeometry(QtCore.QRect(20, 147, 265, 25))
            self.groupbox_file_selection.setGeometry(QtCore.QRect(10, 80, 300, 211))
            self.label_title_gpu.setGeometry(QtCore.QRect(20, 185, 60, 16))
            self.checkbox_gpu.setGeometry(QtCore.QRect(60, 187, 16, 16))
            self.label_gpu_availability.setGeometry(QtCore.QRect(80, 185, 80, 16))
            self.label_title_disk.setGeometry(QtCore.QRect(170, 185, 60, 16))
            self.checkbox_disk.setGeometry(QtCore.QRect(210, 187, 16, 16))
            self.groupbox_od_options.setVisible(False)
            self.groupbox_od_results.setVisible(False)

            self.groupbox_reference.setVisible(True)
            self.groupbox_fm_results.setVisible(True)
        elif "Object Detection" in active_method.get_type():
            self.current_method_mode = SSVII_GUI.OBJECT_DETECTION_MODE
            self.groupbox_reference.setVisible(False)
            self.groupbox_fm_results.setVisible(False)
            self.groupbox_od_options.setVisible(True)
            self.groupbox_od_results.setVisible(True)

            self.label_title_analysis_imgs_path.setGeometry(QtCore.QRect(20, 25, 121, 16))
            self.label_analysis_imgs_path.setGeometry(QtCore.QRect(20, 45, 200, 21))
            self.button_browse_analysis_imgs.setGeometry(QtCore.QRect(227, 45, 58, 21))
            self.label_title_method.setGeometry(QtCore.QRect(20, 75, 121, 16))
            self.combobox_method.setGeometry(QtCore.QRect(20, 97, 265, 25))
            self.groupbox_file_selection.setGeometry(QtCore.QRect(10, 80, 300, 161))
            self.label_title_gpu.setGeometry(QtCore.QRect(20, 135, 121, 16))
            self.checkbox_gpu.setGeometry(QtCore.QRect(60, 137, 16, 16))
            self.label_gpu_availability.setGeometry(QtCore.QRect(80, 135, 80, 16))
            self.label_title_disk.setGeometry(QtCore.QRect(170, 135, 60, 16))
            self.checkbox_disk.setGeometry(QtCore.QRect(210, 137, 16, 16))
            self.label_title_reference_img_path.setVisible(False)
            self.label_reference_img_path.setVisible(False)
            self.button_browse_reference_img.setVisible(False)
            self.proc_manager.clear_classes()
            self.list_od_classes.clear()
            item = QtWidgets.QListWidgetItem("all")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.list_od_classes.addItem(item)
            labels = self.proc_manager.get_class_labels()
            for label in labels:
                item = QtWidgets.QListWidgetItem(label)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                self.list_od_classes.addItem(item)
                self.proc_manager.activate_class(label)
            if len(self.res_manager.get_analysis()) > 0:
                self.button_od_process.setVisible(True)

    def on_checkbox_disk_changed(self):
        self.change_all_clickables_state(False)

        if SSVII_GUI.running_thread is not None:
            SSVII_GUI.running_thread.join()

        class Worker(QtCore.QObject):
            sig_done = pyqtSignal(bool)

            def run(self, target, use_disk):
                self.sig_done.emit(target(use_disk))

        def finished(success):
            self.change_all_clickables_state(True)
            SSVII_GUI.running_thread = None

        worker = Worker()
        worker.sig_done.connect(finished)
        thread = threading.Thread(target=worker.run, args=(self.res_manager.change_storage_mode, self.checkbox_disk.isChecked()))
        thread.daemon = True
        thread.stop = False
        SSVII_GUI.running_thread = thread
        thread.start()

    def on_list_od_classes_changed(self, label):
        if label.text() == "all":
            items = self.list_od_classes.findItems('*', QtCore.Qt.MatchWildcard)
            for item in items:
                item.setCheckState(label.checkState())
            return

        if label.checkState() == QtCore.Qt.Checked:
            self.proc_manager.activate_class(label.text())
        else:
            self.proc_manager.deactivate_class(label.text())

    def on_button_select_region(self):
        """
        Handles the process of selecting a region of interest in the reference image.
        Called when the user tries to select a region of interest by pressing the button_select_region button.
        """

        # If a reference isn't setup then abort
        reference = self.ref_manager.get_reference()
        if reference is None:
            return

        # Open an opencv window with the reference image where the user will be able to select a region
        import cv2, numpy
        window_name = 'Drag to select a region and press enter to confirm'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, SSVII_GUI.MAIN_WINDOW_WIDTH, SSVII_GUI.MAIN_WINDOW_HEIGHT)
        img = self.resize_img(reference.get_img_original(), SSVII_GUI.MAIN_WINDOW_WIDTH,
                              SSVII_GUI.MAIN_WINDOW_HEIGHT)
        # Enable region selection in this window
        r = cv2.selectROI(window_name, img)
        # After confirmation we can close the window
        cv2.destroyWindow(window_name)
        if r[0] == 0 and r[1] == 0 and r[2] == 0 and r[3] == 0:
            return
        # Get the selected region as an image by taking it from the original image
        region = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Let the reference manager set itself up with the new region
        reference.set_reference_region(region)
        # Resize the region image so that it fits the reference image label while keeping the aspect ratio
        # and display it in place of the reference image
        region = self.resize_img(numpy.copy(region), self.label_reference_img.width(),
                                 self.label_reference_img.height())
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        # Create QImage from cv2 img
        qimg = QtGui.QImage(region.data,
                            region.shape[1],
                            region.shape[0],
                            QtGui.QImage.Format_RGB888)
        # Create pixmap from QImage
        pixmap = QtGui.QPixmap.fromImage(qimg)
        # Update image QLabel
        self.label_reference_img.setPixmap(pixmap)
        # Update reference's information labels
        # self.update_reference_info()

    def on_button_process(self):
        """
        Called when the user tries to process the current analysis images by pressing the button_process button.
        Delegates the processing to the processing manager and, if successful, updates the displayed result(s)
        """

        if SSVII_GUI.running_thread is not None:
            SSVII_GUI.running_thread.join()

        if len(self.res_manager.get_analysis()) == 0 or (self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE and self.ref_manager.get_reference() is None):
            return

        # Disable all buttons as soon as we start processing. Use a timer so that we don't have to rely on the gui
        # worker and instead do it immediately
        QtCore.QTimer.singleShot(1, lambda: self.change_all_clickables_state(False))

        # Hide process button and display the progress bar
        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            button_process = self.button_process
            progress_bar = self.progress_bar
        else:
            button_process = self.button_od_process
            progress_bar = self.progress_bar_od

        button_process.setVisible(False)
        progress_bar.setVisible(True)
        progress_bar.setValue(0)
        progress_bar.setFormat("")

        class Worker(QtCore.QObject):

            sig_update_progress = pyqtSignal(int)
            sig_done = pyqtSignal(bool)

            def run(self, target, use_gpu, use_disk):
                self.sig_done.emit(target(use_gpu, use_disk, self.sig_update_progress))

        def finished(success):
            # Unhide process button and display the progress bar
            if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
                button_process = self.button_process
                progress_bar = self.progress_bar
            else:
                button_process = self.button_od_process
                progress_bar = self.progress_bar_od

            button_process.setVisible(True)
            progress_bar.setVisible(False)
            progress_bar.setValue(0)
            progress_bar.setFormat("")

            if not success:
                msg = QtWidgets.QMessageBox()
                msg.setStyleSheet("QLabel{min-height: 30px;}")
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setWindowIcon(QtGui.QIcon("./resources/icon.png"))
                msg.setText("Something went wrong!")
                msg.setWindowTitle("Error")
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec()
            else:
                # Update visible GUI widgets according to the active layout (normal or advanced)
                mode = None
                if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
                    mode = self.combobox_fm_sort_mode.currentText()
                elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
                    mode = self.combobox_od_sort_mode.currentText()
                self.res_manager.sort_results(mode)
                self.update_visible_elements(update_info=True)

            # Enable all buttons after we finish processing
            # QtCore.QTimer.singleShot(1, lambda: self.change_all_clickables_state(True))
            self.change_all_clickables_state(True)
            SSVII_GUI.running_thread = None

        worker = Worker()
        worker.sig_update_progress.connect(self.update_progress_bar)
        worker.sig_done.connect(finished)
        thread = threading.Thread(target=worker.run, args=(self.proc_manager.process, self.checkbox_gpu.isChecked(), self.checkbox_disk.isChecked()))
        thread.daemon = True
        SSVII_GUI.running_thread = thread
        thread.start()

    def on_button_img_fm_bounding_box(self):
        """
        Changes the current display mode to DISPLAY_MODE_OUTLINE and updates the currently displayed result image to
        reflect it. Called when the button button_img_outline is pressed.
        """

        SSVII_GUI.DISPLAY_FM_BOUNDING_BOX = not SSVII_GUI.DISPLAY_FM_BOUNDING_BOX
        self.update_single_result()
        if self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
            self.update_multi_results_display_modes(rebuild=False)

    def on_button_img_fm_circle_prediction(self):
        """
        Changes the current display mode to DISPLAY_MODE_OUTLINE and updates the currently displayed result image to
        reflect it. Called when the button button_img_outline is pressed.
        """

        SSVII_GUI.DISPLAY_FM_CIRCLE_PREDICTION = not SSVII_GUI.DISPLAY_FM_CIRCLE_PREDICTION
        self.update_single_result()
        if self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
            self.update_multi_results_display_modes(rebuild=False)

    def on_button_img_fm_keypoints(self):
        """
        Changes the current display mode to DISPLAY_MODE_KEYPOINTS and updates the currently displayed result image to
        reflect it. Called when the button button_img_keypoints is pressed.
        """

        SSVII_GUI.DISPLAY_FM_KEYPOINTS = not SSVII_GUI.DISPLAY_FM_KEYPOINTS
        self.update_single_result()
        if self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
            self.update_multi_results_display_modes(rebuild=False)

    def on_button_img_fm_matches(self):
        """
        Changes the current display mode to DISPLAY_MODE_MATCHES and updates the currently displayed result image to
        reflect it. Called when the button button_img_matches is pressed.
        """

        SSVII_GUI.DISPLAY_FM_MATCHES = not SSVII_GUI.DISPLAY_FM_MATCHES
        self.update_single_result()
        if self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
            self.update_multi_results_display_modes(rebuild=False)

    def on_button_od_img_boxes(self):
        SSVII_GUI.DISPLAY_OD_BOUNDING_BOXES = not SSVII_GUI.DISPLAY_OD_BOUNDING_BOXES
        self.update_single_result()
        if self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
            self.update_multi_results_display_modes(rebuild=False)

    def on_button_od_img_classes(self):
        SSVII_GUI.DISPLAY_OD_CLASSES = not SSVII_GUI.DISPLAY_OD_CLASSES
        self.update_single_result()
        if self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
            self.update_multi_results_display_modes(rebuild=False)

    def on_button_od_img_masks(self):
        SSVII_GUI.DISPLAY_OD_MASKS = not SSVII_GUI.DISPLAY_OD_MASKS
        self.update_single_result()
        if self.current_results_mode == self.RESULTS_MODE_GRID_1 or self.current_results_mode == self.RESULTS_MODE_GRID_2 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
            self.update_multi_results_display_modes(rebuild=False)

    def on_combobox_results_mode_change(self, mode):
        if "Single" in mode:
            self.current_results_mode = SSVII_GUI.RESULTS_MODE_SINGLE
        elif "Grid 1" in mode:
            self.current_results_mode = SSVII_GUI.RESULTS_MODE_GRID_1
        elif "Grid 2" in mode:
            self.current_results_mode = SSVII_GUI.RESULTS_MODE_GRID_2
        elif "Pile" in mode:
            self.current_results_mode = SSVII_GUI.RESULTS_MODE_PILE
        elif "Spiral" in mode:
            self.current_results_mode = SSVII_GUI.RESULTS_MODE_SPIRAL

        self.combobox_fm_results_mode.setCurrentText(mode)
        self.combobox_od_results_mode.setCurrentText(mode)

        # Update the visible results groupbox GUI elements
        self.update_visible_elements(update_info=False)

    def on_combobox_sort_mode_changed(self, mode):
        if (self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE and mode not in SSVII_GUI.SORT_MODES_AVAILABLE["Feature Matching"]) or (self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE and mode not in SSVII_GUI.SORT_MODES_AVAILABLE["Object Detection"]):
            return

        """if "Relevance" in mode:
            self.current_sort_mode = SSVII_GUI.SORT_MODE_RELEVANCE
        elif "Keypoints" in mode:
            self.current_sort_mode = SSVII_GUI.SORT_MODE_KEYPOINTS
        elif "Descriptors" in mode:
            self.current_sort_mode = SSVII_GUI.SORT_MODE_DESCRIPTORS
        elif "Matches" in mode:
            self.current_sort_mode = SSVII_GUI.SORT_MODE_MATCHES
        elif "Avg. Confidence" in mode:
            self.current_sort_mode = SSVII_GUI.SORT_MODE_AVG_CONFIDENCE
        elif "Detections" in mode:
            self.current_sort_mode = SSVII_GUI.SORT_MODE_DETECTIONS
        elif "Filename" in mode:
            self.current_sort_mode = SSVII_GUI.SORT_MODE_FILENAME"""

        #self.combobox_fm_sort_mode.setCurrentText(mode)
        #self.combobox_od_sort_mode.setCurrentText(mode)

        # Update the visible results groupbox GUI elements
        self.res_manager.sort_results(mode)
        self.current_result_index = 0
        self.update_single_result()
        self.update_visible_elements(update_info=False)

    def on_button_next_result(self):
        """
        Called when the button_next_result button is pressed. Updates the current result index and displays it
        """

        num_results = len(self.res_manager.get_results())
        # If there is less than 2 results then we abort
        if num_results < 2:
            return

        # If the current result is the last then go back to the first, else go to the next
        if num_results - 1 > self.current_result_index:
            self.current_result_index += 1
        else:
            self.current_result_index = 0

        # Display the now current result
        self.update_single_result()

    def on_button_prev_result(self):
        """
        Called when the button_prev_result is pressed. Updates the current result index and displays it
        """

        num_results = len(self.res_manager.get_results())
        # If there is less than 2 results then we abort
        if num_results < 2:
            return

        # If the current result is the first then go back to the last, else go to the previous
        if self.current_result_index > 0:
            self.current_result_index -= 1
        else:
            self.current_result_index = num_results - 1
        self.update_single_result()

    def on_button_animate(self):
        SSVII_GUI.ANIMATE_RESULTS_MODE = not SSVII_GUI.ANIMATE_RESULTS_MODE

        #if SSVII_GUI.ANIMATE_RESULTS_MODE:
        #    self.res_manager.sort_results("Filename")

        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            if SSVII_GUI.ANIMATE_RESULTS_MODE:
                self.button_animate.setIcon(QtGui.QIcon("./resources/pause_icon.png"))
                self.button_prev_result.setDisabled(True)
                self.button_next_result.setDisabled(True)
                self.button_speed_down.setDisabled(False)
                self.button_speed_up.setDisabled(False)
                self.animate_results()
            else:
                #self.res_manager.sort_results("relevance")
                self.button_animate.setIcon(QtGui.QIcon("./resources/play_icon.png"))
                self.button_prev_result.setDisabled(False)
                self.button_next_result.setDisabled(False)
                self.button_speed_down.setDisabled(True)
                self.button_speed_up.setDisabled(True)
        elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
            if SSVII_GUI.ANIMATE_RESULTS_MODE:
                self.button_od_animate.setIcon(QtGui.QIcon("./resources/pause_icon.png"))
                self.button_od_prev_result.setDisabled(True)
                self.button_od_next_result.setDisabled(True)
                self.button_od_speed_down.setDisabled(False)
                self.button_od_speed_up.setDisabled(False)
                self.animate_results()
            else:
                #self.res_manager.sort_results("Avg. Confidence")
                self.button_od_animate.setIcon(QtGui.QIcon("./resources/play_icon.png"))
                self.button_od_prev_result.setDisabled(False)
                self.button_od_next_result.setDisabled(False)
                self.button_od_speed_down.setDisabled(True)
                self.button_od_speed_up.setDisabled(True)

    def on_button_speed_down(self):
        if SSVII_GUI.ANIMATION_FRAME_RATE <= 10:
            SSVII_GUI.ANIMATION_FRAME_RATE = 1
        else:
            SSVII_GUI.ANIMATION_FRAME_RATE -= 10

    def on_button_speed_up(self):
        if SSVII_GUI.ANIMATION_FRAME_RATE >= 50:
            SSVII_GUI.ANIMATION_FRAME_RATE = 60
        else:
            SSVII_GUI.ANIMATION_FRAME_RATE += 10

    def on_popup_reference_img(self, event):
        """
        Opens an opencv window displaying the reference image. Called when the button popup_reference_img is pressed.
        """

        # If a reference isn't set then abort
        reference = self.ref_manager.get_reference()
        if reference is None:
            return

        # Create window and display the reference image
        window_name = 'Reference'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, SSVII_GUI.MAIN_WINDOW_WIDTH, SSVII_GUI.MAIN_WINDOW_HEIGHT)
        img = reference.get_reference_region()
        img = self.resize_img(img, SSVII_GUI.MAIN_WINDOW_WIDTH, SSVII_GUI.MAIN_WINDOW_HEIGHT)
        cv2.imshow(window_name, img)

    def on_popup_result_img(self, event):
        """
        Opens an opencv window displaying the current result image in a larger size.
        Called when the user wants to enlarge the result image by pressing the popup_result_img button.
        """

        # If there are no results then abort
        results = self.res_manager.get_results()
        if len(results) <= 0:
            return

        # Get the current result
        result = self.res_manager.get_result_at_index(self.current_result_index)

        # Create an opencv window to display the result
        if self.current_method_mode == SSVII_GUI.FEATURE_MATCHING_MODE:
            window_name = "Result {0}:   {1}   Relevance {2}%   # Keypoints {3}   # Descriptors {4}   # Matches {5}".format(
                str(result.get_id()),
                self.res_manager.get_original_path(result),
                str(round(result.get_relevance(), 2)),
                str(len(result.get_keypoints())),
                str(len(result.get_descriptors())),
                str(len(result.get_matches())))
        elif self.current_method_mode == SSVII_GUI.OBJECT_DETECTION_MODE:
            window_name = "Result {0}:   {1}   Avg. Confidence {2}%   # Classes {3}   # Detections {4}".format(
                str(result.get_id()),
                self.res_manager.get_original_path(result),
                str(round(result.get_avg_confidence(), 2)),
                str(result.get_num_classes()),
                str(result.get_num_detections()))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, SSVII_GUI.MAIN_WINDOW_WIDTH, SSVII_GUI.MAIN_WINDOW_HEIGHT)

        # If the current display mode includes the matches found (DISPLAY_MODE_MATCHES and DISPLAY_MODE_ALL), and there
        # weren't found relevant matches for this result, display an appropriate placeholder image mentioning it
        # If there were found enough matches for this result or we are in another display mode, display the result image
        # as normal
        img = self.get_result_img()
        if img is None:
            img = cv2.imread("./resources/image_no_matches.png", 1)

        # Resize the image to fit the window and display it
        img = self.resize_img(img, SSVII_GUI.MAIN_WINDOW_WIDTH, SSVII_GUI.MAIN_WINDOW_HEIGHT)
        cv2.imshow(window_name, img)

    def change_all_clickables_state(self, new_state):
        """
        Enables/Disables all clickable widgets.
        :param new_state: Bool that determines whether to enable or disable the widgets. True -> Enable ; False -> Disable
        """

        new_state = not new_state
        self.button_browse_reference_img.setDisabled(new_state)
        self.button_browse_analysis_imgs.setDisabled(new_state)
        self.combobox_method.setDisabled(new_state)
        self.checkbox_gpu.setDisabled(new_state)
        self.checkbox_disk.setDisabled(new_state)
        self.button_select_region.setDisabled(new_state)
        self.button_process.setDisabled(new_state)
        self.button_img_fm_circle_prediction.setDisabled(new_state)
        self.button_fm_img_bounding_box.setDisabled(new_state)
        self.button_img_fm_keypoints.setDisabled(new_state)
        self.button_img_fm_matches.setDisabled(new_state)
        #self.button_change_mode.setDisabled(new_state)
        self.combobox_fm_results_mode.setDisabled(new_state)
        self.combobox_fm_sort_mode.setDisabled(new_state)
        self.button_next_result.setDisabled(new_state)
        self.button_prev_result.setDisabled(new_state)
        self.button_animate.setDisabled(new_state)
        self.label_reference_img.setDisabled(new_state)
        self.label_result_img.setDisabled(new_state)
        #self.scroll_area_widget.setDisabled(new_state)

        self.button_od_process.setDisabled(new_state)
        #self.button_od_img_original.setDisabled(new_state)
        self.button_od_img_boxes.setDisabled(new_state)
        self.button_od_img_class_labels.setDisabled(new_state)
        self.button_od_img_masks.setDisabled(new_state)
        #self.button_od_img_all.setDisabled(new_state)
        #self.button_od_change_mode.setDisabled(new_state)
        self.combobox_od_results_mode.setDisabled(new_state)
        self.combobox_od_sort_mode.setDisabled(new_state)
        if self.current_results_mode == SSVII_GUI.RESULTS_MODE_GRID_1 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_GRID_2 or self.current_results_mode == SSVII_GUI.RESULTS_MODE_PILE or self.current_results_mode == self.RESULTS_MODE_SPIRAL:
            self.button_od_prev_result.setDisabled(True)
            self.button_od_next_result.setDisabled(True)
            self.button_od_animate.setDisabled(True)
            self.button_od_speed_down.setDisabled(True)
            self.button_od_speed_up.setDisabled(True)
        else:
            self.button_od_prev_result.setDisabled(new_state)
            self.button_od_next_result.setDisabled(new_state)
            self.button_od_animate.setDisabled(new_state)
            if not new_state:
                self.button_od_speed_down.setDisabled(True)
                self.button_od_speed_up.setDisabled(True)
            else:
                self.button_od_speed_down.setDisabled(new_state)
                self.button_od_speed_up.setDisabled(new_state)
        self.list_od_classes.setDisabled(new_state)
        self.label_od_result_img.setDisabled(new_state)
        # self.scroll_area_widget_od.setDisabled(new_state)

    def check_cuda_availability(self):
        if cv2.cuda.getCudaEnabledDeviceCount() > 0 and self.proc_manager.get_active_method().get_cuda_support():
            self.label_gpu_availability.setText("(available)")
            self.checkbox_gpu.setDisabled(False)
        else:
            self.label_gpu_availability.setText("(unavailable)")
            self.checkbox_gpu.setChecked(False)
            self.checkbox_gpu.setDisabled(True)
    # endregion
