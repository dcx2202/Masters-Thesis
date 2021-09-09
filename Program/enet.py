import cv2
import numpy as np


class ENet:
    """
    This class is responsible for image processing using the SIFT, SURF or ORB feature matching algorithms
    """

    def __init__(self):
        # COCO class labels that the model was trained to detect
        self.__class_labels = ['unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
                               'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

        # Get a list of colors to represent the classes
        self.__colors = [[0, 0, 0], [81, 0, 81], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                         [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                         [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                         [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                         [0, 0, 230], [119, 11, 32]]
        self.__colors = np.array(self.__colors, dtype="uint8")
        self.__model_path = "./resources/models/enet_model/enet-model.net"
        self.__net = cv2.dnn.readNet(self.__model_path)

    def get_class_labels(self):
        return self.__class_labels

    def detect(self, image, classes, use_cuda):
        # Enable/Disable gpu processing
        if use_cuda:
            self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        img = cv2.resize(image, (1024, 512))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (1024, 512), 0,
                                     swapRB=True, crop=False)
        self.__net.setInput(blob)
        output = self.__net.forward()
        (number_classes, height, width) = output.shape[1:4]

        # our output class ID map will be number_classes x height x width in
        # size, so we take the argmax to find the class label with the
        # largest probability for each and every (x, y)-coordinate in the
        # image
        class_map = np.argmax(output[0], axis=0)
        for id in np.unique(class_map):
            if self.__class_labels[id] not in classes:
                class_map = np.where(class_map == id, 0, class_map)

        # given the class ID map, we can map each of the class IDs to its
        # corresponding color
        mask = self.__colors[class_map]

        # resize the mask such that its dimensions match the original size
        # of the input frame
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        # perform a weighted combination of the input frame with the mask
        # to form an output visualization
        img_od_masks = ((0.3 * img) + (0.7 * mask)).astype("uint8")
        min_height = 300
        if image.shape[0] < min_height:
            img_od_masks = cv2.resize(img_od_masks, (int(min_height / (image.shape[0] / image.shape[1])), min_height))
        else:
            img_od_masks = cv2.resize(img_od_masks, (image.shape[1], image.shape[0]))

        # Initialize the legend visualization
        num_classes = len(np.unique(class_map))
        row_height = 25
        legend = np.full(((num_classes * row_height), 150, 3), 33, dtype="uint8")

        # Get the detected class's names
        class_names = []
        for id in np.unique(class_map):
            if self.__class_labels[id] in classes:
                class_names.append(self.__class_labels[id])

        # Draw the class names + colors on the legend
        i = 0
        for label in sorted(class_names):
            id = self.__class_labels.index(label)
            color = [int(c) for c in self.__colors[id]]
            cv2.putText(legend, self.__class_labels[id], (5, (i * row_height) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 204, 204), 1, cv2.LINE_AA)
            cv2.rectangle(legend, (100, (i * row_height)), (150, (i * row_height) + row_height), tuple(color), -1)
            i += 1

        legend = cv2.resize(legend, (int(legend.shape[1] / legend.shape[0] * img_od_masks.shape[0]), img_od_masks.shape[0]))
        img_od_masks = np.concatenate((legend, img_od_masks), axis=1)
        img_od_masks = np.c_[img_od_masks, np.full((img_od_masks.shape[0], img_od_masks.shape[1], 1), 255)]

        return {"detections": None, "num_classes": num_classes}, {"img_od_bounding_boxes": None,
                                                                         "img_od_class_labels": None,
                                                                         "img_od_masks": img_od_masks}
