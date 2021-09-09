from method import Method

class ProcessingManager:
    """
    Manages the processing of reference and analysis images
    """

    def __init__(self, reference_manager, results_manager, feature_matcher, object_detector):
        # Managers to delegate to
        self.__ref_manager = reference_manager
        self.res_manager = results_manager
        self.__feature_matcher = feature_matcher
        self.object_detector = object_detector
        self.methods_available = (Method("SIFT", "Feature Matching:  SIFT", "Feature Matching", False),
                                  Method("SURF", "Feature Matching:  SURF", "Feature Matching", True),
                                  Method("ORB", "Feature Matching:  ORB", "Feature Matching", True),
                                  Method("BRISK", "Feature Matching:  BRISK", "Feature Matching", False),
                                  Method("AKAZE", "Feature Matching:  AKAZE", "Feature Matching", False),
                                  Method("YOLOv3", "Object Localization:  YOLOv3", "Object Detection", True),
                                  Method("SSD", "Object Localization:  SSD", "Object Detection", True),
                                  Method("ENet", "Semantic Segmentation:  ENet", "Object Detection", True),
                                  Method("Mask R-CNN", "Instance Segmentation:  Mask R-CNN", "Object Detection", True),
                                  Method("ENet + Mask R-CNN", "Panoptic Segmentation:  ENet + Mask R-CNN", "Object Detection", True))
        self.active_method = self.methods_available[0]
        self.new_results = []
        self.threads = []

    def get_class_labels(self):
        return self.object_detector.get_class_labels(self.active_method)

    def activate_class(self, label):
        self.object_detector.activate_class(label)

    def deactivate_class(self, label):
        self.object_detector.deactivate_class(label)

    def clear_classes(self):
        self.object_detector.clear_classes()

    def get_methods_available(self):
        return self.methods_available

    def get_active_method(self):
        return self.active_method

    def set_active_method(self, description):
        method = next((m for m in self.methods_available if m.get_description() == description), None)
        self.active_method = method

    def process(self, use_cuda, use_disk, sig_progress):
        if "Feature Matching" in self.active_method.get_type():
            return self.process_results_fm(use_cuda, use_disk, sig_progress)
        elif "Object Detection" in self.active_method.get_type():
            return self.process_results_od(use_cuda, use_disk, sig_progress)

    def process_results_fm(self, use_cuda, use_disk, sig_progress):
        reference = self.__ref_manager.get_reference()
        analysis = self.res_manager.get_analysis()

        # If a reference/images for analysis haven't been set then abort
        if reference is None or len(analysis) == 0:
            return False

        # Process the reference image, obtaining the keypoints and descriptors
        try:
            if use_cuda:
                keypoints, keypoints_cuda, descriptors, descriptors_cuda = self.__feature_matcher.process_reference(reference.get_reference_region(), self.active_method, use_cuda)
            else:
                keypoints, descriptors = self.__feature_matcher.process_reference(reference.get_reference_region(), self.active_method, use_cuda)
        except Exception as e:
            return False

        if keypoints is None or descriptors is None:
            return False

        # Update the reference information
        reference.set_keypoints(keypoints)
        reference.set_descriptors(descriptors)

        ref_img = reference.get_reference_region()
        new_results = []
        max_num_matches = 0
        sum_matches = 0
        index = 0
        sig_progress.emit(index)

        for result in analysis:
            try:
                if use_cuda:
                    desc = descriptors_cuda
                else:
                    desc = descriptors
                info, images = self.__feature_matcher.process_result(ref_img, keypoints, desc, self.res_manager.get_img_original(result), self.active_method, use_cuda)
            except:
                return False

            info["id"] = self.res_manager.get_id(result)
            info["original_path"] = self.res_manager.get_original_path(result)
            images["img_original"] = self.res_manager.get_img_original(result, False)
            new_results.append({"info": info, "images": images})
            if len(info["matches"]) > max_num_matches:
                max_num_matches = len(info["matches"])
            sum_matches += len(info["matches"])
            index += 1
            sig_progress.emit(index)

        for item in new_results:
            if len(descriptors) <= 0:#sum_matches <= 0:
                item["info"]["relevance"] = 0
            else:
                item["info"]["relevance"] = len(item["info"]["matches"]) / len(descriptors) * 100#len(item["info"]["matches"]) / max_num_matches * 100#len(item["info"]["matches"]) / len(item["info"]["keypoints"]) * 100#num_matches / sum_matches  # self.calc_fm_relevance()
            index += 1

        return self.res_manager.set_fm_results(new_results, use_disk, sig_progress, index)

    def process_results_od(self, use_cuda, use_disk, sig_progress):

        analysis = self.res_manager.get_analysis()

        # If a reference/images for analysis haven't been set then abort
        if len(analysis) == 0:
            return False

        new_results = []
        index = 0
        sig_progress.emit(index)

        for result in analysis:
            #try:
            info, images = self.object_detector.detect(self.res_manager.get_img_original(result), self.active_method, use_cuda)
            #except:
            #    return False

            if not info["detections"] or len(info["detections"]) == 0:
                info["avg_confidence"] = 0
            else:
                sum_confidence = 0
                for detection in info["detections"]:
                    sum_confidence += detection.get_confidence()
                info["avg_confidence"] = sum_confidence / len(info["detections"]) * 100

            info["id"] = self.res_manager.get_id(result)
            info["original_path"] = self.res_manager.get_original_path(result)
            images["img_original"] = self.res_manager.get_img_original(result, False)
            new_results.append({"info": info, "images": images})
            index += 1
            sig_progress.emit(index)

        #new_results = sorted(new_results, key=lambda r: r["info"]["avg_confidence"], reverse=True)
        return self.res_manager.set_od_results(new_results, use_disk, sig_progress, index)
