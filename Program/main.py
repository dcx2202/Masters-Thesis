from PyQt5 import QtWidgets, QtGui
import sys, shutil
import gui
from file_manager import FileManager
from reference_manager import ReferenceManager
from results_manager import ResultsManager
from processing_manager import ProcessingManager
from feature_matcher import FeatureMatcher
from object_detector import ObjectDetector


if __name__ == "__main__":
    # List of image formats accepted
    formats_accepted = ["png", "jpg", "jpeg", "mp4", "mkv"]

    # Initialize managers
    file_manager = FileManager(accepted_formats=formats_accepted)
    ref_manager = ReferenceManager(file_manager=file_manager)
    res_manager = ResultsManager(file_manager=file_manager)
    feature_matcher = FeatureMatcher()
    object_detector = ObjectDetector()
    proc_manager = ProcessingManager(reference_manager=ref_manager,
                                     results_manager=res_manager,
                                     feature_matcher=feature_matcher,
                                     object_detector=object_detector)

    # Create window and setup GUI
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = gui.SSVII_GUI(ref_manager, res_manager, proc_manager)
    ui.setupUi(main_window)
    main_window.setWindowTitle("SSVII - Search for Similar Visual Information in Images")
    main_window.setWindowIcon(QtGui.QIcon("./resources/icon.png"))
    main_window.setFixedSize(main_window.size())
    main_window.show()

    try: shutil.rmtree("./temp", ignore_errors=True)
    except: pass
    app.exec_()
    try: shutil.rmtree("./temp", ignore_errors=True)
    except: pass
    sys.exit()
