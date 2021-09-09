import cv2
import numpy as np
from detection import Detection


class YOLOv3:
    """
    This class is responsible for image processing using the SIFT, SURF or ORB feature matching algorithms
    """

    def __init__(self):
        # COCO class labels that the model was trained to detect
        self.__class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                              'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        # Get a list of colors to represent the classes
        np.random.seed(5)  # Set a seed so that we always get the same colors
        self.__colors = np.random.randint(0, 255, size=(len(self.__class_labels), 3), dtype="uint8")
        self.__weights_path = "./resources/models/yolo_model/yolov3.weights"
        self.__config_path = "./resources/models/yolo_model/yolov3.cfg"
        self.__net = cv2.dnn.readNetFromDarknet(self.__config_path, self.__weights_path)
        self.__confidence_threshold = 0.5
        self.__threshold = 0.3

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
        layer_names = self.__net.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.__net.setInput(blob)
        layerOutputs = self.__net.forward(layer_names)

        # Lists of detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Loop over each of the layer outputs
        for output in layerOutputs:
            # Loop over the detections
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)  # Get the id
                class_name = self.__class_labels[class_id]  # Get the label
                confidence = scores[class_id]  # Get the confidence

                if class_name not in object_classes:
                    continue

                # If this detection's confidence is higher than our set
                # confidence threshold then we consider it relevant and process it
                if confidence > self.__confidence_threshold:
                    # Scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    (H, W) = image.shape[:2]
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Get the bounding box top left coordinates
                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (height / 2))

                    # Update lists of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([startX, startY, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to suppress weak,
        # overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.__confidence_threshold, self.__threshold)

        # If at least one detection exists
        if len(indexes) > 0:
            indexes = indexes.flatten()
            # Loop over the indexes we are keeping
            for i in indexes:
                class_name = self.__class_labels[class_ids[i]]

                if class_name not in classes:
                    classes.append(class_name)

                # Get the bounding box coordinates
                (startX, startY) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Draw the bounding box
                color = [int(c) for c in self.__colors[class_ids[i]]]
                color.append(255)
                value = image.shape[1] * image.shape[0]
                thickness = int(round(np.interp(value, [40000, 4000000], [1, 10])))
                cv2.rectangle(img_od_bounding_boxes, (startX, startY), (startX + w, startY + h), color, thickness)

                # Draw the class label and confidence
                text = "{0}: {1}%".format(class_name, round(confidences[i] * 100, 1))
                desired_text_height = img_od_bounding_boxes.shape[0] * 0.02
                font_scale = 0.1
                while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(int(font_scale * 3), 1))[0][1] < desired_text_height:
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

                # Create detection object
                detections.append(Detection(confidences[i], class_name))

        return {"detections": detections, "num_classes": len(classes)}, {"img_od_bounding_boxes": img_od_bounding_boxes,
                                                                         "img_od_class_labels": img_od_class_labels,
                                                                         "img_od_masks": None}
