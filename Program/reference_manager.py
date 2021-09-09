
class ReferenceManager:
    """
    This class is responsible for managing the results and controlling access to them
    """

    def __init__(self, file_manager):
        self.__reference = None
        self.__file_manager = file_manager

    def get_reference(self):
        """:return: Returns the reference"""
        return self.__reference

    def set_reference(self, path):
        """
        Takes a path and tries to set up a reference using the file the path points to
        :param: path: Path pointing to a file
        :return: True if successful, False otherwise
        """

        # Try to open the file
        file = self.__file_manager.open_reference_file(path)

        if file:
            import reference
            self.__reference = reference.Reference(file[0], file[1])  # Create and update the reference
            return True
        return False
