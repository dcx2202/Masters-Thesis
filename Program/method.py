

class Method:
    """
    This class is responsible for image processing using the SIFT, SURF or ORB feature matching algorithms
    """

    def __init__(self, name, description, typ, cuda_support):
        self.__name = name
        self.__description = description
        self.__type = typ
        self.__cuda_support = cuda_support

    def get_name(self):
        return self.__name

    def get_description(self):
        return self.__description

    def get_type(self):
        return self.__type

    def get_cuda_support(self):
        return self.__cuda_support
