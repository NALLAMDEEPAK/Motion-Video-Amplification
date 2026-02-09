class MAVException(Exception):
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def to_dict(self):
        return {'error': self.__class__.__name__, 'message': self.message, 'details': self.details}


class VideoError(MAVException):
    pass


class VideoNotFoundError(VideoError):
    def __init__(self, path: str = None):
        super().__init__("Video file not found", f"Path: {path}" if path else None)


class VideoReadError(VideoError):
    def __init__(self, reason: str = None):
        super().__init__("Failed to read video file", reason)


class InvalidVideoError(VideoError):
    def __init__(self, reason: str = None):
        super().__init__("Invalid or corrupted video file", reason)


class ROIError(MAVException):
    pass


class InvalidROIError(ROIError):
    def __init__(self, roi: tuple = None):
        super().__init__("Invalid ROI selection", f"ROI values: {roi}" if roi else "Selection was cancelled")


class ROICancelledError(ROIError):
    def __init__(self):
        super().__init__("ROI selection was cancelled by user")


class PolygonError(MAVException):
    pass


class InvalidPolygonError(PolygonError):
    def __init__(self, num_points: int = 0):
        super().__init__("Invalid polygon selection - need at least 3 points", f"Points selected: {num_points}")


class PolygonCancelledError(PolygonError):
    def __init__(self):
        super().__init__("Polygon selection was cancelled by user")


class ProcessingError(MAVException):
    pass


class NoDataError(ProcessingError):
    def __init__(self, data_type: str = "data"):
        super().__init__(f"No {data_type} available for processing", "Please upload and process a video first")


class InsufficientDataError(ProcessingError):
    def __init__(self, required: int = 0, actual: int = 0):
        super().__init__("Insufficient data points for analysis", f"Required: {required}, Available: {actual}")


class AnalysisError(MAVException):
    pass


class FFTError(AnalysisError):
    def __init__(self, reason: str = None):
        super().__init__("Failed to compute FFT", reason)


class ModalAnalysisError(AnalysisError):
    def __init__(self, reason: str = None):
        super().__init__("Failed to perform modal analysis", reason)


class PlottingError(MAVException):
    def __init__(self, reason: str = None):
        super().__init__("Failed to generate plot", reason)


class FileUploadError(MAVException):
    pass


class NoFileUploadedError(FileUploadError):
    def __init__(self):
        super().__init__("No video file was uploaded", "Please select a video file to upload")


class InvalidFileTypeError(FileUploadError):
    def __init__(self, filename: str = None, allowed: list = None):
        super().__init__("Invalid file type", f"File: {filename}. Allowed: {', '.join(allowed or [])}")


class FileTooLargeError(FileUploadError):
    def __init__(self, size: int = 0, max_size: int = 0):
        super().__init__("File size exceeds limit", f"Size: {size / (1024*1024):.1f}MB, Max: {max_size / (1024*1024):.1f}MB")
