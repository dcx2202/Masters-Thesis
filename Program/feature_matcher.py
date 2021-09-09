import cv2
import numpy as np


class FeatureMatcher:
    """
    This class is responsible for image processing using the SIFT, SURF or ORB feature matching algorithms
    """

    def __init__(self):
        self.__sift = cv2.xfeatures2d.SIFT_create()
        self.__surf = cv2.xfeatures2d.SURF_create()
        self.__surf_cuda = cv2.cuda.SURF_CUDA_create(400)
        self.__orb = cv2.ORB_create(nfeatures=100000)
        self.__orb_cuda = cv2.cuda.ORB_create(nfeatures=100000)
        self.__brisk = cv2.BRISK_create()
        self.__akaze = cv2.AKAZE_create()
        self.__brute_force_matcher_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.__brute_force_matcher_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.__num_matches = 30  # Minimum number of matches that have to be found to be considered relevant

    def compute_keypoints_descriptors(self, img, method, use_cuda):
        """
        Computes the keypoints and descriptors of the image passed as a parameter
        :param img: Image to be processed
        :param method: Method to be used for feature/descriptor detection/extraction (SIFT, SURF or ORB)
        :param use_cuda: True to use cuda, False otherwise
        :return: Returns the keypoints and descriptors of the image passed as a parameter
        """

        # Calculate keypoints and descriptors
        if use_cuda:
            if method == self.__surf_cuda:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray_cuda = cv2.cuda_GpuMat()
                img_gray_cuda.upload(img_gray)
                keypoints_cuda, descriptors_cuda = method.detectWithDescriptors(img=img_gray_cuda, mask=None)
                keypoints = method.downloadKeypoints(keypoints_cuda)  # convert Keypoints to CPU
                descriptors = descriptors_cuda.download()
            elif method == self.__orb_cuda:
                img_cuda = cv2.cuda_GpuMat()
                img_cuda.upload(img)
                img_gray_cuda = cv2.cuda.cvtColor(img_cuda, cv2.COLOR_RGB2GRAY)
                keypoints_cuda, descriptors_cuda = method.detectAndComputeAsync(image=img_gray_cuda, mask=None)
                keypoints = method.convert(keypoints_cuda)
                descriptors = descriptors_cuda.download()
            return keypoints, keypoints_cuda, descriptors, descriptors_cuda
        else:
            # Convert image to grayscale to ignore color
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculate keypoints and descriptors
            return method.detectAndCompute(img_gray, None)

    def process_matches(self, descriptors_1, descriptors_2, matcher):
        """
        Matches the descriptors passed as parameters
        :param descriptors_1: First set of descriptors
        :param descriptors_2: Second set of descriptors
        :param matcher: Matcher to be used (depends on feature extraction method)
        :return: Returns the matches found
        """
        return matcher.match(descriptors_1, descriptors_2)

    def process_reference(self, img_ref, active_method, use_cuda):
        active_method = active_method.get_name()
        if use_cuda and ("SURF" not in active_method and "ORB" not in active_method):
            return
        elif use_cuda:
            if "SURF" in active_method:
                method = self.__surf_cuda
            elif "ORB" in active_method:
                method = self.__orb_cuda
        else:
            if "SIFT" in active_method:
                method = self.__sift
            elif "SURF" in active_method:
                method = self.__surf
            elif "ORB" in active_method:
                method = self.__orb
            elif "BRISK" in active_method:
                method = self.__brisk
            elif "AKAZE" in active_method:
                method = self.__akaze

        return self.compute_keypoints_descriptors(img_ref, method, use_cuda)

    def process_result(self, img_ref, kp_ref, desc_ref, img_analysis, active_method, use_cuda):
        """
        Looks for the reference image in the analysis image by detecting keypoints and descriptors and matching them,
        obtaining relevant information and images depicting the features found
        :param img_ref: Reference image
        :param kp_ref: Pre processed reference keypoints
        :param desc_ref: Pre processed reference descriptors
        :param img_analysis: Image for analysis
        :param active_method: Method to be used for feature/descriptor detection/extraction (SIFT, SURF or ORB)
        :param use_cuda: True to use cuda, False otherwise
        :return: Returns the information regarding the analysis image, and a set of images that depict the features found
        """

        # Images that will portray the features found
        img_fm_bounding_box = np.zeros((img_analysis.shape[0], img_analysis.shape[1], 4), dtype="uint8")
        img_fm_circle_prediction = np.zeros((img_analysis.shape[0], img_analysis.shape[1], 4), dtype="uint8")
        img_fm_keypoints = np.zeros((img_analysis.shape[0], img_analysis.shape[1], 4), dtype="uint8")
        img_fm_matches = np.zeros((img_analysis.shape[0], img_analysis.shape[1], 4), dtype="uint8")

        # Method and matcher to use
        method = None
        matcher = None
        active_method = active_method.get_name()

        if use_cuda and ("SURF" not in active_method and "ORB" not in active_method):
            return
        elif use_cuda:
            if "SURF" in active_method:
                method = self.__surf_cuda
                matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_L2)
            elif "ORB" in active_method:
                method = self.__orb_cuda
                matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
        else:
            if "SIFT" in active_method or "SURF" in active_method:
                matcher = self.__brute_force_matcher_l2
                if "SIFT" in active_method:
                    method = self.__sift
                else:
                    method = self.__surf
            elif "ORB" in active_method or "BRISK" in active_method or "AKAZE" in active_method:
                matcher = self.__brute_force_matcher_hamming
                if "ORB" in active_method:
                    method = self.__orb
                elif "BRISK" in active_method:
                    method = self.__brisk
                elif "AKAZE" in active_method:
                    method = self.__akaze

        # Detect keypoints, compute descriptors and brute-force match
        if use_cuda:
            kp_analysis, kp_analysis_cuda, desc_analysis, desc_analysis_cuda = self.compute_keypoints_descriptors(
                img_analysis, method, use_cuda)
            matches = self.process_matches(desc_ref, desc_analysis_cuda, matcher)
        else:
            kp_analysis, desc_analysis = self.compute_keypoints_descriptors(img_analysis, method, use_cuda)
            matches = self.process_matches(desc_ref, desc_analysis, matcher)

        # Some of the images that portray the features found rely on the number of matches found. If enough matches were
        # found, this result is relevant and we can create those images
        if len(matches) >= self.__num_matches:
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw matches
            img_fm_matches = cv2.drawMatches(img1=img_ref,
                                             keypoints1=kp_ref,
                                             img2=img_fm_matches,
                                             keypoints2=kp_analysis,
                                             matches1to2=matches,
                                             outImg=None,
                                             flags=2)

            # Draw bounding box
            # good_matches = matches[:self.__num_matches]  # Consider only the best matches so that the outline is more accurate
            good_matches = matches
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_analysis[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = img_ref.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            value = img_analysis.shape[1] * img_analysis.shape[0]
            thickness = int(round(np.interp(value, [40000, 4000000], [1, 10])))
            img_fm_bounding_box = cv2.polylines(img_fm_bounding_box, [np.int32(dst)], True, (0, 255, 0, 255), thickness,
                                                cv2.LINE_AA)  # Draw bounding box in green

            # Calculate average point to draw circle guess
            avg_point = [0, 0]
            for match in good_matches:
                avg_point[0] += kp_analysis[match.trainIdx].pt[0]
                avg_point[1] += kp_analysis[match.trainIdx].pt[1]
            avg_point[0] = int(avg_point[0] / len(good_matches))
            avg_point[1] = int(avg_point[1] / len(good_matches))

            # Draw circle guess
            radius = int(round(np.interp(value, [40000, 4000000], [5, 100])))
            cv2.circle(img_fm_circle_prediction, tuple(avg_point), radius, (0, 255, 255, 255), -1)

            """# Offset the keypoints by the width of the reference image that is added on the left side
            offset_kp2 = []
            for kp in kp_analysis:
                offset_kp2.append(cv2.KeyPoint(kp.pt[0] + w, kp.pt[1], kp.size, kp.response, kp.octave, kp.class_id))

            # Draw offset keypoints
            img_fm_offset_keypoints = cv2.drawKeypoints(img_fm_offset_keypoints, offset_kp2, None,
                                                        color=(0, 0, 255))  # Draw keypoints"""

        # Draw keypoints only
        if len(kp_analysis) >= 0:
            img_fm_keypoints = cv2.drawKeypoints(img_fm_keypoints, kp_analysis, None, color=(0, 0, 255, 255))

        # Return information and images
        return {"relevance": 0, "keypoints": kp_analysis, "descriptors": desc_analysis, "matches": matches}, \
               {"img_fm_bounding_box": img_fm_bounding_box, "img_fm_circle_prediction": img_fm_circle_prediction,
                "img_fm_keypoints": img_fm_keypoints, "img_fm_matches": img_fm_matches}
