import glob
import cv2

class FileManager:
    """
    Manages the opening of reference and analysis files
    """
    def __init__(self, accepted_formats):
        self.__formats = accepted_formats

    def open_reference_file(self, path):
        """
        Tries to read the file in the given path into an opencv image
        :param path: Path of the opened reference file
        :return: Returns a tuple (image, path) if successful, else returns None
        """

        # For each of the accepted formats, check if the file's format matches.
        # If it does then try reading it into an opencv image
        for f in self.__formats:
            if path.endswith("." + f):
                try:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)  # Try reading into an opencv image
                except:
                    return None
                file = (img, path)
                return file
        return None

    def open_analysis_files(self, path):
        """
        Tries to read the files in the given directory path into opencv images
        :param path: Path of the directory
        :return: Returns a tuple (image, path) if successful, else returns None
        """

        files = []  # List of files found

        # For each of the accepted formats, look for files of the same format
        for f in self.__formats:
            file_paths = glob.glob(path + "/*." + f)
            # For each file found, try reading it into an opencv image
            for p in file_paths:
                p = p.replace('\\', '/')
                if f == "mp4" or f == "mkv":
                    video = cv2.VideoCapture(p)
                    while True:
                        success, frame = video.read()
                        if not success:
                            break
                        files.append((frame, p))
                else:
                    try:
                        img = cv2.imread(p, cv2.IMREAD_COLOR)
                    except:
                        continue
                    if img is None:
                        continue
                    files.append((img, p))

        """file_paths = glob.glob(path + "/*.*")
        for file in file_paths:
            if ".mp4" in file or ".avi" in file or ".mkv" in file:
                video = cv2.VideoCapture(file)
                success, image = video.read()
                while success:
                    files.append((image, file))
                    success, image = video.read()"""

        # If files were found then return the list, else return None
        if len(files) > 0:
            return sorted(files, key=lambda r: r[1])
        else:
            return None

    def open_image_file(self, path):
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Try reading into an opencv image
        except:
            return None
        return img

    def get_accepted_formats(self):
        """:return: Returns the accepted formats"""
        return self.__formats
