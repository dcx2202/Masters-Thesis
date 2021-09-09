import cv2
import numpy as np


class ENet_MaskRCNN:
    """
    This class is responsible for image processing using the SIFT, SURF or ORB feature matching algorithms
    """

    def __init__(self):
        # COCO class labels that the model was trained to detect
        self.__enet_class_labels = ['unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                                    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
                                    'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        # Get a list of colors to represent the classes
        self.__enet_colors = np.array([[0, 0, 0], [81, 0, 81], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                                       [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                                       [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                                       [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                                       [0, 0, 230], [119, 11, 32]], dtype="uint8")
        self.__enet_model_path = "./resources/models/enet_model/enet-model.net"
        self.__enet_net = cv2.dnn.readNet(self.__enet_model_path)
        self.__maskrcnn_class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
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
                                        'teddy bear', 'hair drier', 'toothbrush']
        self.__maskrcnn_weights_path = "./resources/models/mask_rcnn_model/frozen_inference_graph.pb"
        self.__maskrcnn_config_path = "./resources/models/mask_rcnn_model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        self.__maskrcnn_net = cv2.dnn.readNetFromTensorflow(self.__maskrcnn_weights_path, self.__maskrcnn_config_path)
        self.__maskrcnn_confidence_threshold = 0.5
        self.__maskrcnn_mask_threshold = 0.7

    def get_class_labels(self):
        return self.__enet_class_labels + list(set(self.__maskrcnn_class_labels) - set(self.__enet_class_labels))

    def maskrcnn_detect(self, image, classes, ss):
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        self.__maskrcnn_net.setInput(blob)
        (boxes, masks) = self.__maskrcnn_net.forward(["detection_out_final", "detection_masks"])

        # Loop over the detections
        for i in range(0, boxes.shape[2]):
            confidence = boxes[0, 0, i, 2]  # Get the confidence

            # If this detection's confidence is higher than our set confidence
            # threshold then we consider it relevant and process it
            if confidence > self.__maskrcnn_confidence_threshold:
                class_id = int(boxes[0, 0, i, 1])  # Get the id
                class_name = self.__maskrcnn_class_labels[class_id]  # Get the label
                if class_name not in classes:
                    continue

                # Scale the bounding box coordinates back, relative to the size
                # of the frame, and calculate the dimensions of the bounding box
                (H, W) = image.shape[:2]
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY

                # Get the segmentation mask
                border = 4
                mask = masks[i, class_id]
                # Resize to the dimensions of the bounding box
                mask_small = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
                mask_small = (mask_small > self.__maskrcnn_mask_threshold)
                mask_big = cv2.resize(mask, (boxW + border * 2, boxH + border * 2), interpolation=cv2.INTER_CUBIC)
                mask_big = (mask_big > self.__maskrcnn_mask_threshold)

                # Create an empty image with the specified size
                img = np.zeros((boxH + border * 2, boxW + border * 2, 4), dtype="uint8")
                img[mask_big != 0] = [255, 255, 255, 255]

                roi = image[startY:endY, startX:endX, :]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
                roi[mask_small == 0] = [0, 0, 0, 0]

                top = int((img.shape[0] - roi.shape[0]) / 2)
                side = int((img.shape[1] - roi.shape[1]) / 2)
                roi = cv2.copyMakeBorder(src=roi, top=top, bottom=top,
                                         left=side, right=side,
                                         borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
                m = roi[:, :, 3] > 0
                img[m] = [0, 0, 0, 0]
                img = cv2.resize(img, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
                m = img[:, :, 3] > 0

                # Get the region of the image that corresponds to the mask
                region = ss[startY:endY, startX:endX][mask_small]

                # Blend this class's color with the region under the mask
                color = np.array([[255, 255, 255, 255]])
                blended = ((0.2 * color) + (0.8 * region)).astype("uint8")
                ss[startY:endY, startX:endX, :][mask_small] = blended
                ss[startY:endY, startX:endX, :][m] = img[m]
                text = "{0}: {1}%".format(class_name, round(confidence * 100, 1))
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                text_w = text_size[0][0]
                text_h = text_size[0][1]
                text_x = int(startX - (text_w - (endX - startX)) / 2)
                text_y = int(startY - 5)
                cv2.rectangle(ss, (text_x, text_y - text_h), (text_x + text_w, text_y), (50, 50, 50), -1)
                cv2.putText(ss, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return ss

    def ss_detect(self, image, classes):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False)
        self.__enet_net.setInput(blob)
        output = self.__enet_net.forward()

        # infer the total number of classes along with the spatial
        # dimensions of the mask image via the shape of the output array
        (numClasses, height, width) = output.shape[1:4]

        # our output class ID map will be num_classes x height x width in
        # size, so we take the argmax to find the class label with the
        # largest probability for each and every (x, y)-coordinate in the
        # image
        class_map = np.argmax(output[0], axis=0)

        for i in classes:
            if i in self.__enet_class_labels:
                continue
            id = self.__enet_class_labels.index(i)
            class_map = np.where(class_map == id, 0, class_map)

        ss_mask = self.__enet_colors[class_map]

        # resize the mask such that its dimensions match the original size
        # of the input frame
        ss_mask = cv2.resize(ss_mask, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        # perform a weighted combination of the input frame with the mask
        # to form an output visualization
        output = ((0.3 * image) + (0.7 * ss_mask)).astype("uint8")

        classes_detected = []
        for id in np.unique(class_map):
            if self.__enet_class_labels[id] in classes and self.__enet_class_labels[id] not in classes_detected:
                classes_detected.append(self.__enet_class_labels[id])
        return output, classes_detected

    def detect(self, img, classes, use_cuda):
        # Enable/Disable gpu processing
        if use_cuda:
            self.__enet_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.__enet_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.__maskrcnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.__maskrcnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.__enet_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.__enet_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.__maskrcnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.__maskrcnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        min_height = 300
        if img.shape[0] < min_height:
            image = cv2.resize(img, (int(min_height / (img.shape[0] / img.shape[1])), min_height))
        else:
            image = img

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False)
        self.__enet_net.setInput(blob)
        output = self.__enet_net.forward()

        # infer the total number of classes along with the spatial
        # dimensions of the mask image via the shape of the output array
        # (numClasses, height, width) = output.shape[1:4]

        # our output class ID map will be num_classes x height x width in
        # size, so we take the argmax to find the class label with the
        # largest probability for each and every (x, y)-coordinate in the
        # image

        class_map = np.argmax(output[0], axis=0)
        enet_classes_detected = []
        for id in np.unique(class_map):
            if self.__enet_class_labels[id] not in classes:
                class_map = np.where(class_map == id, 0, class_map)
            elif self.__enet_class_labels[id] not in enet_classes_detected:
                enet_classes_detected.append(self.__enet_class_labels[id])

        ss_mask = self.__enet_colors[class_map]

        # resize the mask such that its dimensions match the original size
        # of the input frame
        ss_mask = cv2.resize(ss_mask, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        # perform a weighted combination of the input frame with the mask
        # to form an output visualization
        img_od_masks = ((0.3 * image) + (0.7 * ss_mask)).astype("uint8")

        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        self.__maskrcnn_net.setInput(blob)
        (boxes, masks) = self.__maskrcnn_net.forward(["detection_out_final", "detection_masks"])

        # Loop over the detections
        maskrcnn_classes_detected = []
        for i in range(0, boxes.shape[2]):
            confidence = boxes[0, 0, i, 2]  # Get the confidence

            # If this detection's confidence is higher than our set confidence
            # threshold then we consider it relevant and process it
            if confidence > self.__maskrcnn_confidence_threshold:
                class_id = int(boxes[0, 0, i, 1])  # Get the id
                class_name = self.__maskrcnn_class_labels[class_id]  # Get the label
                if class_name not in classes:
                    continue
                elif class_name not in maskrcnn_classes_detected:
                    maskrcnn_classes_detected.append(class_name)

                # Scale the bounding box coordinates back, relative to the size
                # of the frame, and calculate the dimensions of the bounding box
                (H, W) = image.shape[:2]
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY

                # Get the segmentation mask
                border = 4
                mask = masks[i, class_id]
                # Resize to the dimensions of the bounding box
                mask_small = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
                mask_small = (mask_small > self.__maskrcnn_mask_threshold)
                mask_big = cv2.resize(mask, (boxW + border * 2, boxH + border * 2), interpolation=cv2.INTER_CUBIC)
                mask_big = (mask_big > self.__maskrcnn_mask_threshold)

                # Create an empty image with the specified size
                aux = np.zeros((boxH + border * 2, boxW + border * 2, 4), dtype="uint8")
                aux[mask_big != 0] = [255, 255, 255, 255]

                roi = image[startY:endY, startX:endX, :]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
                roi[mask_small == 0] = [0, 0, 0, 0]

                top = int((aux.shape[0] - roi.shape[0]) / 2)
                side = int((aux.shape[1] - roi.shape[1]) / 2)
                roi = cv2.copyMakeBorder(src=roi, top=top, bottom=top,
                                         left=side, right=side,
                                         borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
                m = roi[:, :, 3] > 0
                aux[m] = [0, 0, 0, 0]
                aux = cv2.resize(aux, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
                m = aux[:, :, 3] > 0

                # Get the region of the image that corresponds to the mask
                img_od_masks = cv2.cvtColor(img_od_masks, cv2.COLOR_BGR2BGRA)
                region = img_od_masks[startY:endY, startX:endX][mask_small]

                # Blend this class's color with the region under the mask
                color = np.array([[255, 255, 255, 255]])
                blended = ((0.2 * color) + (0.8 * region)).astype("uint8")
                img_od_masks[startY:endY, startX:endX, :][mask_small] = blended
                img_od_masks[startY:endY, startX:endX, :][m] = aux[m]
                text = "{0}: {1}%".format(class_name, round(confidence * 100, 1))
                desired_text_height = img_od_masks.shape[0] * 0.02
                font_size = 0.1
                while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, max(int(font_size * 3), 1))[0][1] < desired_text_height:
                    font_size += 0.1
                font_size = round(font_size, 1)
                thickness = max(int(font_size * 3), 1)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
                text_w = text_size[0][0]
                text_h = text_size[0][1]
                text_x = int(startX - (text_w - (endX - startX)) / 2)
                text_y = int(startY - 5)
                cv2.rectangle(img_od_masks, (text_x, text_y - text_h), (text_x + text_w, text_y), (50, 50, 50, 255), -1)
                cv2.putText(img_od_masks, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255, 255), thickness, cv2.LINE_AA)

        # initialize the legend visualization
        row_height = 25
        legend = np.full(((len(enet_classes_detected) * row_height), 150, 3), 33, dtype="uint8")
        legend = np.c_[legend, np.full((legend.shape[0], legend.shape[1], 1), 255)].astype("uint8")

        # Draw the class name + color on the legend
        i = 0
        for label in sorted(enet_classes_detected):
            id = self.__enet_class_labels.index(label)
            color = [int(c) for c in self.__enet_colors[id]]
            color.append(255)
            cv2.putText(legend, label, (5, (i * row_height) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 204, 204), 1, cv2.LINE_AA)
            cv2.rectangle(legend, (100, (i * row_height)), (150, (i * row_height) + row_height), tuple(color), -1)
            i += 1

        legend = cv2.resize(legend, (int(img_od_masks.shape[0] / (legend.shape[0] / legend.shape[1])), img_od_masks.shape[0]))
        if img_od_masks.shape[2] == 3:
            img_od_masks = np.c_[img_od_masks, np.full((img_od_masks.shape[0], img_od_masks.shape[1], 1), 255)]
        img_od_masks = np.concatenate((legend, img_od_masks), axis=1)

        return {"detections": None, "num_classes": len(np.unique(enet_classes_detected + maskrcnn_classes_detected))}, {"img_od_bounding_boxes": None,
                                                                                                              "img_od_class_labels": None,
                                                                                                              "img_od_masks": img_od_masks}
