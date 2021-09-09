
class Detection:
    """
        This class represents an object detection.
        A detection object has a confidence, a class label, a bounding box and possibly a mask
    """

    def __init__(self, confidence, class_label):
        self.__confidence = confidence
        self.__class_label = class_label

    def get_confidence(self):
        return self.__confidence

    def get_class_label(self):
        return self.__class_label
