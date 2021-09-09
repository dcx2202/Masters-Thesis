import threading

from results import Result, FMResult, ODResult
import os
import shutil
import cv2


class ResultsManager:
    """
    This class is responsible for managing the results and controlling access to them
    """

    def __init__(self, file_manager):
        self.analysis = []
        self.results = []
        self.__file_manager = file_manager

    def get_analysis(self):
        """:return: Returns the results"""
        return self.analysis

    def set_analysis(self, results):
        self.analysis = results

    def open_analysis(self, path, use_disk=False):
        """
        Sets up the images in the given path to be ready for processing later
        :param path: Path that points to the directory where the images for analysis are
        """

        thread = threading.currentThread()

        files = self.__file_manager.open_analysis_files(path)
        if files:
            if use_disk:
                dir = "./temp/analysis"
                try:
                    shutil.rmtree(dir, ignore_errors=True)
                except:
                    pass
                try:
                    os.makedirs(dir, exist_ok=True)
                except OSError:
                    return False

            self.analysis = []

            for i in range(len(files)):
                if getattr(thread, "stop", False):
                    if use_disk:
                        dir = "./temp/analysis"
                        try:
                            shutil.rmtree(dir, ignore_errors=True)
                        except:
                            pass
                    return False
                img_original = files[i][0]#cv2.imread(files[i][1], cv2.IMREAD_COLOR)
                if use_disk:
                    path = dir + "/{0}.jpg".format(i + 1)
                    cv2.imwrite(path, img_original, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    img_original = path
                self.analysis.append(Result(i + 1, files[i][1], img_original))
            return True
        return False

    def change_storage_mode(self, use_disk):
        if len(self.analysis) == 0 and len(self.results) == 0:
            #callback(True)
            return True
        if use_disk:
            # Prepare directories
            base_dir = "./temp"
            try:
                shutil.rmtree(base_dir, ignore_errors=True)
            except:
                pass
            try:
                os.makedirs(base_dir, exist_ok=True)
                os.makedirs(base_dir + "/analysis", exist_ok=True)
                os.makedirs(base_dir + "/processed", exist_ok=True)
            except OSError:
                #callback(True)
                return False

            # Save unprocessed results' images in disk
            for item in self.analysis:
                path = base_dir + "/analysis/{0}.jpg".format(item.get_id())
                cv2.imwrite(path, item.get_img_original(), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                item.set_img_original(path)

            # Save processed results' images in disk
            for result in self.results:
                id = result.get_id()
                result_dir = base_dir + "/processed/{0}".format(id)
                try:
                    os.makedirs(result_dir, exist_ok=True)
                except OSError:
                    #callback(True)
                    return False

                # Original image
                #path = result_dir + "/{0}_0_0_0_0.png".format(id)
                #cv2.imwrite(path, result.get_img_original(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                #result.set_img_original(path)

                if type(result) is FMResult:
                    # Circle
                    path = result_dir + "/{0}_1_0_0_0.png".format(id)
                    cv2.imwrite(path, result.get_img_fm_circle_prediction(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                    result.set_img_fm_circle_prediction(path)

                    # Bounding box
                    path = result_dir + "/{0}_0_1_0_0.png".format(id)
                    cv2.imwrite(path, result.get_img_fm_bounding_box(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                    result.set_img_fm_bounding_box(path)

                    # Keypoints
                    path = result_dir + "/{0}_0_0_1_0.png".format(id)
                    cv2.imwrite(path, result.get_img_fm_keypoints(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                    result.set_img_fm_keypoints(path)

                    # Matches
                    path = result_dir + "/{0}_0_0_0_1.png".format(id)
                    cv2.imwrite(path, result.get_img_fm_matches(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                    result.set_img_fm_matches(path)
                elif type(result) is ODResult:
                    # Bounding boxes
                    if result.get_img_od_bounding_boxes() is not None:
                        path = result_dir + "/{0}_1_0_0.png".format(id)
                        cv2.imwrite(path, result.get_img_od_bounding_boxes(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                        result.set_img_od_bounding_boxes(path)

                    # Class labels
                    if result.get_img_od_class_labels() is not None:
                        path = result_dir + "/{0}_0_1_0.png".format(id)
                        cv2.imwrite(path, result.get_img_od_class_labels(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                        result.set_img_od_class_labels(path)

                    # Masks
                    if result.get_img_od_masks() is not None:
                        path = result_dir + "/{0}_0_0_1.png".format(id)
                        cv2.imwrite(path, result.get_img_od_masks(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                        result.set_img_od_masks(path)
        else:
            base_dir = "./temp"
            # Read images and store in memory
            for item in self.analysis:
                item.set_img_original(cv2.imread(item.get_img_original(), cv2.IMREAD_COLOR))

            for result in self.results:
                if type(result.get_img_original()) is str:
                    result.set_img_original(cv2.imread(result.get_img_original(), cv2.IMREAD_COLOR))
                if type(result) is FMResult:
                    result.set_img_fm_bounding_box(cv2.imread(result.get_img_fm_bounding_box(), cv2.IMREAD_UNCHANGED))
                    result.set_img_fm_circle_prediction(cv2.imread(result.get_img_fm_circle_prediction(), cv2.IMREAD_UNCHANGED))
                    result.set_img_fm_keypoints(cv2.imread(result.get_img_fm_keypoints(), cv2.IMREAD_UNCHANGED))
                    result.set_img_fm_matches(cv2.imread(result.get_img_fm_matches(), cv2.IMREAD_UNCHANGED))
                elif type(result) is ODResult:
                    result.set_img_od_bounding_boxes(cv2.imread(result.get_img_od_bounding_boxes(), cv2.IMREAD_UNCHANGED))
                    result.set_img_od_class_labels(cv2.imread(result.get_img_od_class_labels(), cv2.IMREAD_UNCHANGED))
                    result.set_img_od_masks(cv2.imread(result.get_img_od_masks(), cv2.IMREAD_UNCHANGED))
            try:
                shutil.rmtree(base_dir, ignore_errors=True)
            except:
                pass
        #callback(True)
        return True

    def get_results(self):
        return self.results

    def set_fm_results(self, results, use_disk, sig_progress, index):
        new_results = []

        if not use_disk:
            for result in results:
                new_results.append(FMResult(result["info"], result["images"]))
                index += 1
                sig_progress.emit(index)
            self.results = new_results
            return True

        base_dir = "./temp/processed"
        try:
            shutil.rmtree(base_dir, ignore_errors=True)
        except:
            pass
        try:
            os.makedirs(base_dir, exist_ok=True)
        except OSError:
            return False

        for result in results:
            id = result["info"]["id"]
            result_dir = base_dir + "/{0}".format(id)
            try:
                os.makedirs(result_dir, exist_ok=True)
            except OSError:
                return False

            # Circle
            path_img_circle = result_dir + "/{0}_1_0_0_0.png".format(id)
            cv2.imwrite(path_img_circle, result["images"]["img_fm_circle_prediction"],
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            result["images"]["img_fm_circle_prediction"] = path_img_circle

            # Bounding box
            path_img_bounding_box = result_dir + "/{0}_0_1_0_0.png".format(id)
            cv2.imwrite(path_img_bounding_box, result["images"]["img_fm_bounding_box"],
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            result["images"]["img_fm_bounding_box"] = path_img_bounding_box

            # Keypoints
            path_img_keypoints = result_dir + "/{0}_0_0_1_0.png".format(id)
            cv2.imwrite(path_img_keypoints, result["images"]["img_fm_keypoints"],
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            result["images"]["img_fm_keypoints"] = path_img_keypoints

            # Matches
            path_img_matches = result_dir + "/{0}_0_0_0_1.png".format(id)
            cv2.imwrite(path_img_matches, result["images"]["img_fm_matches"], [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            result["images"]["img_fm_matches"] = path_img_matches

            new_results.append(FMResult(result["info"], result["images"]))
            index += 1
            sig_progress.emit(index)

        self.results = new_results
        return True

    def set_od_results(self, results, use_disk, sig_progress, index):
        new_results = []

        if not use_disk:
            for result in results:
                new_results.append(ODResult(result["info"], result["images"]))
                index += 1
                sig_progress.emit(index)
            self.results = new_results
            return True

        base_dir = "./temp/processed"
        try:
            shutil.rmtree(base_dir, ignore_errors=True)
        except:
            pass
        try:
            os.makedirs(base_dir, exist_ok=True)
        except OSError:
            return False

        for result in results:
            id = result["info"]["id"]
            result_dir = base_dir + "/{0}".format(id)
            try:
                os.makedirs(result_dir, exist_ok=True)
            except OSError:
                return False

            # Bounding boxes
            if result["images"]["img_od_bounding_boxes"] is not None:
                path_img_bounding_boxes = result_dir + "/{0}_1_0_0.png".format(id)
                cv2.imwrite(path_img_bounding_boxes, result["images"]["img_od_bounding_boxes"], [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                result["images"]["img_od_bounding_boxes"] = path_img_bounding_boxes

            # Class labels
            if result["images"]["img_od_class_labels"] is not None:
                path_img_class_labels = result_dir + "/{0}_0_1_0.png".format(id)
                cv2.imwrite(path_img_class_labels, result["images"]["img_od_class_labels"], [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                result["images"]["img_od_class_labels"] = path_img_class_labels

            # Masks
            if result["images"]["img_od_masks"] is not None:
                path_img_masks = result_dir + "/{0}_0_0_1.png".format(id)
                cv2.imwrite(path_img_masks, result["images"]["img_od_masks"], [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                result["images"]["img_od_masks"] = path_img_masks

            new_results.append(ODResult(result["info"], result["images"]))
            index += 1
            sig_progress.emit(index)

        self.results = new_results
        return True

    def set_results(self, results):
        self.results = results

    def sort_results(self, criteria):
        if "relevance" in criteria.lower():
            self.results = sorted(self.results, key=lambda r: r.get_relevance(), reverse=True)
        elif "kps/des" in criteria.lower():
            self.results = sorted(self.results, key=lambda r: len(r.get_keypoints()), reverse=True)
        elif "matches" in criteria.lower():
            self.results = sorted(self.results, key=lambda r: len(r.get_matches()), reverse=True)
        elif "avg. confidence" in criteria.lower():
            try:
                self.results = sorted(self.results, key=lambda r: r.get_avg_confidence(), reverse=True)
            except Exception:
                pass
        elif "detections" in criteria.lower():
            self.results = sorted(self.results, key=lambda r: r.get_num_detections(), reverse=True)
        elif "filename" in criteria.lower():
            self.results = sorted(self.results, key=lambda r: r.get_original_path())

    def get_result_at_index(self, index):
        return self.results[index]

    def get_result_by_id(self, id):
        for result in self.results:
            if result.get_id() == id:
                return result

    def get_original_path(self, result):
        return result.get_original_path()

    def get_id(self, result):
        return result.get_id()

    def get_relevance(self, result):
        if type(result) is not FMResult:
            return 0
        return result.get_relevance()

    def get_keypoints(self, result):
        if type(result) is not FMResult:
            return 0
        return result.get_keypoints()

    def get_descriptors(self, result):
        if type(result) is not FMResult:
            return 0
        return result.get_descriptors()

    def get_matches(self, result):
        if type(result) is not FMResult:
            return 0
        return result.get_matches()

    def get_avg_confidence(self, result):
        if type(result) is not ODResult:
            return 0
        return result.get_avg_confidence()

    def get_num_classes(self, result):
        if type(result) is not ODResult:
            return 0
        return result.get_num_classes()

    def get_num_detections(self, result):
        if type(result) is not ODResult:
            return 0
        return result.get_num_detections()

    # region Images
    def get_img_original(self, result, img_needed=True):
        img = result.get_img_original()
        if img_needed and type(img) is str:
            return self.__file_manager.open_image_file(img)
        return img

    def get_img_fm_bounding_box(self, result, img_needed=True):
        img = result.get_img_fm_bounding_box()
        if img_needed and type(img) is str:
            return self.__file_manager.open_image_file(img)
        return img

    def get_img_fm_circle_prediction(self, result, img_needed=True):
        img = result.get_img_fm_circle_prediction()
        if img_needed and type(img) is str:
            return self.__file_manager.open_image_file(img)
        return img

    def get_img_fm_keypoints(self, result, img_needed=True):
        img = result.get_img_fm_keypoints()
        if img_needed and type(img) is str:
            return self.__file_manager.open_image_file(img)
        return img

    def get_img_fm_matches(self, result, img_needed=True):
        img = result.get_img_fm_matches()
        if img_needed and type(img) is str:
            return self.__file_manager.open_image_file(img)
        return img

    def get_img_od_bounding_boxes(self, result, img_needed=True):
        img = result.get_img_od_bounding_boxes()
        if img_needed and type(img) is str:
            return self.__file_manager.open_image_file(img)
        return img

    def get_img_od_class_labels(self, result, img_needed=True):
        img = result.get_img_od_class_labels()
        if img_needed and type(img) is str:
            return self.__file_manager.open_image_file(img)
        return img

    def get_img_od_masks(self, result, img_needed=True):
        img = result.get_img_od_masks()
        if img_needed and type(img) is str:
            return self.__file_manager.open_image_file(img)
        return img
    # endregion
