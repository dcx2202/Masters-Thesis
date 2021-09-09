
from mask_rcnn import MaskRCNN
from ssd import SSD
from yolov3 import YOLOv3
from enet import ENet
from enet_maskrcnn import ENet_MaskRCNN


class ObjectDetector:
    """
    This class is responsible for image processing using the SIFT, SURF or ORB feature matching algorithms
    """

    def __init__(self):
        self.yolov3 = YOLOv3()
        self.ssd = SSD()
        self.mask_rcnn = MaskRCNN()
        self.enet = ENet()
        self.enet_maskrcnn = ENet_MaskRCNN()
        self.active_classes = []

    def get_class_labels(self, method):
        method_name = method.get_name()
        if method.get_name() == "YOLOv3":
            method = self.yolov3
        elif method.get_name() == "SSD":
            method = self.ssd
        elif method.get_name() == "Mask R-CNN":
            method = self.mask_rcnn
        elif method_name == "ENet":
            method = self.enet
        elif method_name == "ENet + Mask R-CNN":
            method = self.enet_maskrcnn
        return sorted(method.get_class_labels())

    def activate_class(self, label):
        if label not in self.active_classes:
            self.active_classes.append(label)

    def deactivate_class(self, label):
        if label in self.active_classes:
            self.active_classes.remove(label)

    def clear_classes(self):
        self.active_classes = []

    def detect(self, image, method, use_cuda):
        method = method.get_name()
        if method == "YOLOv3":
            return self.yolov3.detect(image, self.active_classes, use_cuda)
        elif method == "SSD":
            return self.ssd.detect(image, self.active_classes, use_cuda)
        elif method == "ENet":
            return self.enet.detect(image, self.active_classes, use_cuda)
        elif method == "Mask R-CNN":
            return self.mask_rcnn.detect(image, self.active_classes, use_cuda)
        elif method == "ENet + Mask R-CNN":
            return self.enet_maskrcnn.detect(image, self.active_classes, use_cuda)
