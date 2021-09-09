class Result:
    def __init__(self, id, original_path, img_original):
        self.id = id
        self.original_path = original_path
        self.img_original = img_original

    def get_id(self):
        return self.id

    def get_original_path(self):
        return self.original_path

    def get_img_original(self):
        return self.img_original

    def set_img_original(self, img_original):
        self.img_original = img_original


class FMResult(Result):
    def __init__(self, info, images):
        super().__init__(info["id"], info["original_path"], images["img_original"])
        self.keypoints = info["keypoints"]
        self.descriptors = info["descriptors"]
        self.matches = info["matches"]
        self.relevance = info["relevance"]
        self.img_fm_circle_prediction = images['img_fm_circle_prediction']
        self.img_fm_bounding_box = images["img_fm_bounding_box"]
        self.img_fm_keypoints = images["img_fm_keypoints"]
        self.img_fm_matches = images["img_fm_matches"]

    def get_matches(self):
        return self.matches

    def get_relevance(self):
        return self.relevance

    def set_relevance(self, relevance):
        self.relevance = relevance

    def get_img_fm_bounding_box(self):
        return self.img_fm_bounding_box

    def set_img_fm_bounding_box(self, new_img):
        self.img_fm_bounding_box = new_img

    def get_img_fm_circle_prediction(self):
        return self.img_fm_circle_prediction

    def set_img_fm_circle_prediction(self, new_img):
        self.img_fm_circle_prediction = new_img

    def get_img_fm_keypoints(self):
        return self.img_fm_keypoints

    def set_img_fm_keypoints(self, new_img):
        self.img_fm_keypoints = new_img

    def get_img_fm_matches(self):
        return self.img_fm_matches

    def set_img_fm_matches(self, new_img):
        self.img_fm_matches = new_img

    def get_keypoints(self):
        return self.keypoints

    def get_descriptors(self):
        return self.descriptors


class ODResult(Result):
    def __init__(self, info, images):
        super().__init__(info["id"], info["original_path"], images["img_original"])
        self.detections = info["detections"]
        self.avg_confidence = info["avg_confidence"]
        self.num_classes = info["num_classes"]
        if self.detections:
            self.num_detections = len(self.detections)
        else:
            self.num_detections = 0
        self.img_od_bounding_boxes = images["img_od_bounding_boxes"]
        self.img_od_class_labels = images["img_od_class_labels"]
        self.img_od_masks = images["img_od_masks"]

    def get_detections(self):
        return self.detections

    def get_avg_confidence(self):
        return self.avg_confidence

    def get_num_classes(self):
        return self.num_classes

    def get_num_detections(self):
        return self.num_detections

    def get_img_od_bounding_boxes(self):
        return self.img_od_bounding_boxes

    def set_img_od_bounding_boxes(self, new_img):
        self.img_od_bounding_boxes = new_img

    def get_img_od_class_labels(self):
        return self.img_od_class_labels

    def set_img_od_class_labels(self, new_img):
        self.img_od_class_labels = new_img

    def get_img_od_masks(self):
        return self.img_od_masks

    def set_img_od_masks(self, new_img):
        self.img_od_masks = new_img
