import cv2
import numpy as np
from detection import Detection


class SSD:
    """
    This class is responsible for image processing using the SIFT, SURF or ORB feature matching algorithms
    """

    def __init__(self):
        # COCO class labels that the model was trained to detect
        self.__class_labels = ["background", "airplane", "bicycle", "bird", "boat",
                               "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
                               "dog", "horse", "motorcycle", "person", "potted plant", "sheep",
                               "couch", "train", "tv"]
        # Get a list of colors to represent the classes
        np.random.seed(5)  # Set a seed so that we always get the same colors
        self.__colors = np.random.randint(0, 255, size=(len(self.__class_labels), 3), dtype="uint8")
        self.__model_path = "./resources/models/ssd_model/MobileNetSSD_deploy.caffemodel"
        self.__config_path = "./resources/models/ssd_model/MobileNetSSD_deploy.prototxt.txt"
        self.__net = cv2.dnn.readNetFromCaffe(self.__config_path, self.__model_path)
        self.__confidence_threshold = 0.2

    def get_class_labels(self):
        return self.__class_labels

    def detect(self, image, object_classes, use_cuda):
        # Enable/Disable gpu processing
        if use_cuda:
            self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        detections = []
        classes = []
        img_od_bounding_boxes = np.zeros((image.shape[0], image.shape[1], 4), dtype="uint8")
        img_od_class_labels = np.zeros((image.shape[0], image.shape[1], 4), dtype="uint8")
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.__net.setInput(blob)
        boxes = self.__net.forward()

        # Loop over the detections
        for i in np.arange(0, boxes.shape[2]):
            confidence = boxes[0, 0, i, 2]  # Get the confidence

            # If this detection's confidence is higher than our set confidence
            # threshold then we consider it relevant and process it
            if confidence > self.__confidence_threshold:
                class_id = int(boxes[0, 0, i, 1])  # Get the id
                class_name = self.__class_labels[class_id]  # Get the label
                if class_name not in object_classes:
                    continue

                if class_name not in classes:
                    classes.append(class_name)

                # Scale the bounding box coordinates back,
                # relative to the size of the frame
                (H, W) = image.shape[:2]
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box
                color = [int(c) for c in self.__colors[class_id]]
                color.append(255)
                value = image.shape[1] * image.shape[0]
                thickness = int(round(np.interp(value, [40000, 4000000], [1, 10])))
                cv2.rectangle(img_od_bounding_boxes, (startX, startY), (endX, endY), color, thickness)

                # Draw the class label and confidence
                text = "{0}: {1}%".format(class_name, round(confidence * 100, 1))
                desired_text_height = img_od_bounding_boxes.shape[0] * 0.02
                font_scale = 0.1
                while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(int(font_scale * 3), 1))[0][
                    1] < desired_text_height:
                    font_scale += 0.1
                font_scale = round(font_scale, 1)
                thickness = max(int(font_scale * 3), 1)
                #thickness = int(round(np.interp(value, [40000, 4000000], [1, 12])))
                #font_scale = np.interp(value, [40000, 4000000], [0.3, 2.5])

                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_w = text_size[0][0]
                text_h = text_size[0][1]
                text_x = startX - 1
                if startY - text_h - 5 <= 0:
                    text_y = text_h
                else:
                    text_y = int(startY - 5)
                cv2.rectangle(img_od_class_labels, (text_x, text_y - text_h), (text_x + text_w, text_y + int(text_h / 2)), color, -1)
                cv2.putText(img_od_class_labels, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 255), thickness)

                #cv2.putText(img_od_class_labels, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                #            thickness)

                # Create detection object
                detections.append(Detection(confidence, class_name))

        return {"detections": detections, "num_classes": len(classes)}, {"img_od_bounding_boxes": img_od_bounding_boxes,
                                                                         "img_od_class_labels": img_od_class_labels,
                                                                         "img_od_masks": None}
