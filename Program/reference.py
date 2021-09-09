
class Reference:
    """
    This class represents a reference.
    A reference object has a path, an image, a region of interest, ...
    """

    def __init__(self, img_original, path):
        self.__path = path
        self.__keypoints = None
        self.__descriptors = None
        self.__img_original = img_original
        self.__img_region_of_interest = img_original

    # region Info

    def get_path(self):
        """:return: Returns this reference's path"""
        return self.__path

    def get_keypoints(self):
        """:return: Returns this reference's keypoints"""
        return self.__keypoints

    def set_keypoints(self, keypoints):
        """
        Sets this reference's keypoints
        :param: keypoints: New keypoints
        """
        self.__keypoints = keypoints

    def get_descriptors(self):
        """:return: Returns this reference's descriptors"""
        return self.__descriptors

    def set_descriptors(self, descriptors):
        """
        Sets this reference's descriptors
        :param: descriptors: New descriptors
        """
        self.__descriptors = descriptors

    # endregion

    # region Images

    def get_reference_region(self):
        """:return: Returns this reference's region of interest"""
        return self.__img_region_of_interest

    def set_reference_region(self, region):
        """
        Sets this reference's region of interest and resets keypoints/descriptors
        :param: region: New region of interest
        """
        self.__img_region_of_interest = region
        self.__keypoints = None
        self.__descriptors = None

    def get_img_original(self):
        """:return: Returns this reference's original image"""
        return self.__img_original

    # endregion
