import cv2
import numpy as np

from detection import Detection


class MaskRCNN:
    """
    This class is responsible for image processing using the SIFT, SURF or ORB feature matching algorithms
    """

    def __init__(self):
        # COCO class labels that the model was trained to detect
        self.__class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                               'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
                               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat',
                               'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
                               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                               'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                               'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
                               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                               'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
                               'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
                               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                               'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
                               'teddy bear', 'hair drier', 'toothbrush'
                               ]
        # Get a list of colors to represent the classes
        np.random.seed(5)  # Set a seed so that we always get the same colors
        self.__colors = np.random.randint(0, 255, size=(len(self.__class_labels), 3), dtype="uint8")
        self.__weights_path = "./resources/models/mask_rcnn_model/frozen_inference_graph.pb"
        self.__config_path = "./resources/models/mask_rcnn_model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        self.__net = cv2.dnn.readNetFromTensorflow(self.__weights_path, self.__config_path)
        self.__confidence_threshold = 0.5
        self.__mask_threshold = 0.7

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
        img_od_masks = np.zeros((image.shape[0], image.shape[1], 4), dtype="uint8")
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        self.__net.setInput(blob)
        (boxes, masks) = self.__net.forward(["detection_out_final", "detection_masks"])

        # Loop over the detections
        for i in range(0, boxes.shape[2]):
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

                # Scale the bounding box coordinates back, relative to the size
                # of the frame, and calculate the dimensions of the bounding box
                (H, W) = image.shape[:2]
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY

                # Get the segmentation mask
                mask = masks[i, class_id]
                # Resize to the dimensions of the bounding box
                mask = cv2.resize(mask, (boxW, boxH),
                                  interpolation=cv2.INTER_CUBIC)
                # Create a binary mask considering our set threshold
                mask = (mask > self.__mask_threshold)

                # Get the region of the image that corresponds to the mask
                region = image[startY:endY, startX:endX][mask]

                # Blend this class's color with the region under the mask
                color = self.__colors[class_id]
                blended = ((0.4 * color) + (0.6 * region)).astype("uint8")
                blended = np.c_[blended, np.full((blended.shape[0], 1), 255)]
                # Put the blended region over the original frame
                img_od_masks[startY:endY, startX:endX, :][mask] = blended

                # Draw the bounding box
                color = [int(c) for c in color]
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
                cv2.putText(img_od_class_labels, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0, 255), thickness, cv2.LINE_AA)

                # Create detection object
                detections.append(Detection(confidence, class_name))

        return {"detections": detections, "num_classes": len(classes)}, {"img_od_bounding_boxes": img_od_bounding_boxes, "img_od_class_labels": img_od_class_labels, "img_od_masks": img_od_masks}
