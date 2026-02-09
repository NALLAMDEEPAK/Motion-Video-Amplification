import cv2
import numpy as np
import pandas as pd
import os
import logging
from typing import Tuple, List, Dict, Optional

from backend.utils.exceptions import (
    VideoNotFoundError, VideoReadError, InvalidVideoError,
    InvalidROIError, ROICancelledError, InvalidPolygonError, PolygonCancelledError,
    ProcessingError, NoDataError
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, threshold: int = 10):
        self.threshold = threshold
        self.displacement_data = []
        self.time_data = []
        self.fps = 0
        self.sampling_rate = 0
    
    def _validate_video_path(self, video_path: str) -> None:
        if not video_path:
            raise VideoNotFoundError("No video path provided")
        if not os.path.exists(video_path):
            raise VideoNotFoundError(video_path)
    
    def _validate_video_capture(self, cap: cv2.VideoCapture, video_path: str) -> None:
        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")
    
    def _validate_frame(self, ret: bool, frame: np.ndarray, context: str = "frame") -> None:
        if not ret or frame is None:
            raise VideoReadError(f"Failed to read {context}")
        if frame.size == 0:
            raise VideoReadError(f"Empty {context} received")
    
    def _validate_roi(self, roi: Tuple[int, int, int, int]) -> None:
        if roi is None:
            raise ROICancelledError()
        x, y, w, h = roi
        if w == 0 or h == 0:
            raise InvalidROIError(roi)
        if x < 0 or y < 0 or w < 0 or h < 0:
            raise InvalidROIError(roi)
    
    def _validate_points(self, points: List[Tuple[int, int]], min_points: int = 3) -> None:
        if points is None:
            raise PolygonCancelledError()
        if len(points) < min_points:
            raise InvalidPolygonError(len(points))
    
    def _safe_crop_frame(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise VideoReadError("Cannot crop empty frame")
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        cropped = frame[y:y+h, x:x+w]
        if cropped.size == 0:
            raise InvalidROIError((x, y, w, h))
        return cropped
    
    def select_roi(self, video_path: str) -> Tuple[int, int, int, int]:
        self._validate_video_path(video_path)
        cap = cv2.VideoCapture(video_path)
        self._validate_video_capture(cap, video_path)
        try:
            ret, frame = cap.read()
            self._validate_frame(ret, frame, "first frame")
            instruction_frame = frame.copy()
            cv2.putText(instruction_frame, "Draw ROI: Click & Drag | ENTER/SPACE: Confirm | C: Cancel | ESC: Exit", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            window_name = "Select Region of Interest - Press ENTER to confirm"
            roi = cv2.selectROI(window_name, instruction_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            self._validate_roi(roi)
            return roi
        except cv2.error as e:
            raise VideoReadError(str(e))
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def track_motion_roi(self, video_path: str, roi: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        self._validate_video_path(video_path)
        if roi is None:
            roi = self.select_roi(video_path)
        self._validate_roi(roi)
        x, y, w, h = roi
        cap = cv2.VideoCapture(video_path)
        self._validate_video_capture(cap, video_path)
        try:
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise VideoReadError("Cannot determine video frame count")
            self.sampling_rate = self.fps
            ret, prev_frame = cap.read()
            self._validate_frame(ret, prev_frame, "first frame")
            prev_roi = self._safe_crop_frame(prev_frame, x, y, w, h)
            prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
            roi_center_x = w // 2
            roi_center_y = h // 2
            displacement_data = []
            time_data = []
            frame_count = 0
            prev_cx, prev_cy = roi_center_x, roi_center_y
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_count += 1
                try:
                    roi_frame = self._safe_crop_frame(frame, x, y, w, h)
                    current_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(prev_gray, current_gray).astype(float)
                    total_diff = np.sum(diff)
                    if total_diff > 0:
                        y_coords, x_coords = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
                        cx = np.sum(x_coords * diff) / total_diff
                        cy = np.sum(y_coords * diff) / total_diff
                        displacement_x = cx - prev_cx
                        displacement_y = cy - prev_cy
                        prev_cx = cx
                        prev_cy = cy
                    else:
                        displacement_x = 0.0
                        displacement_y = 0.0
                    displacement_data.append((displacement_x, displacement_y))
                    time_data.append(frame_count / self.fps)
                    prev_gray = current_gray.copy()
                except cv2.error:
                    displacement_data.append((0.0, 0.0))
                    time_data.append(frame_count / self.fps)
            if len(displacement_data) == 0:
                raise NoDataError("displacement data")
            self.displacement_data = displacement_data
            self.time_data = time_data
            return {'displacement_data': displacement_data, 'time_data': time_data, 'sampling_rate': self.sampling_rate, 'fps': self.fps}
        except cv2.error as e:
            raise ProcessingError(str(e))
        finally:
            cap.release()
    
    def select_points_of_interest(self, video_path: str) -> Tuple[List[Tuple[int, int]], float, float]:
        self._validate_video_path(video_path)
        cap = cv2.VideoCapture(video_path)
        self._validate_video_capture(cap, video_path)
        try:
            ret, frame = cap.read()
            self._validate_frame(ret, frame, "first frame")
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / self.fps if total_frames > 0 and self.fps > 0 else 0
            self.sampling_rate = self.fps * duration
            points = []
            drawing = False
            cancelled = False
            def select_point(event, x, y, flags, param):
                nonlocal points, drawing
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    points.append((x, y))
                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        points.append((x, y))
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
            window_name = 'Draw Polygon - Q: Finish | R: Reset | ESC: Cancel'
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, select_point)
            while True:
                frame_copy = frame.copy()
                cv2.putText(frame_copy, "Draw polygon area | Q: Finish & Analyze | R: Reset | ESC: Cancel", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame_copy, f"Points: {len(points)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(frame_copy, points[i], points[i + 1], (0, 255, 0), 2)
                if drawing and len(points) > 0:
                    cv2.polylines(frame_copy, [np.array(points)], False, (0, 255, 0), 2)
                for point in points:
                    cv2.circle(frame_copy, point, 5, (0, 255, 0), -1)
                cv2.imshow(window_name, frame_copy)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    points = []
                elif key == 27:
                    cancelled = True
                    points = []
                    break
            cv2.destroyAllWindows()
            if cancelled:
                raise PolygonCancelledError()
            self._validate_points(points)
            return points, self.sampling_rate, self.fps
        except cv2.error as e:
            raise VideoReadError(str(e))
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def track_motion_polygon(self, video_path: str, points: List[Tuple[int, int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_video_path(video_path)
        self._validate_points(points)
        cap = cv2.VideoCapture(video_path)
        self._validate_video_capture(cap, video_path)
        try:
            ret, prev_frame = cap.read()
            self._validate_frame(ret, prev_frame, "first frame")
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            displacement_data = {f"Point_{i + 1}": [] for i in range(len(points))}
            time_data = []
            contour_displacement = []
            mask = np.zeros_like(prev_gray)
            pts = np.array(points, np.int32)
            cv2.fillPoly(mask, [pts], 255)
            prev_centroid = None
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_count += 1
                try:
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    current_gray_masked = cv2.bitwise_and(current_gray, current_gray, mask=mask)
                    prev_gray_masked = cv2.bitwise_and(prev_gray, prev_gray, mask=mask)
                    diff = cv2.absdiff(prev_gray_masked, current_gray_masked)
                    _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        layout_contour = max(contours, key=cv2.contourArea)
                        M_current = cv2.moments(layout_contour)
                        area = M_current["m00"] + 1e-5
                        cx_current = int(M_current["m10"] / area)
                        cy_current = int(M_current["m01"] / area)
                        contour_displacement.append((cx_current, cy_current))
                        current_centroid = (cx_current, cy_current)
                        if prev_centroid:
                            displacement_x = current_centroid[0] - prev_centroid[0]
                            displacement_y = current_centroid[1] - prev_centroid[1]
                            contour_displacement.append((displacement_x, displacement_y))
                        prev_centroid = current_centroid
                    for i, point in enumerate(points):
                        x, y = point
                        if 0 <= y < current_gray.shape[0] and 0 <= x < current_gray.shape[1]:
                            try:
                                current_point_value = current_gray[y, x]
                                matches = np.where(current_gray == current_point_value)
                                if len(matches[0]) > 0 and len(matches[1]) > 0:
                                    displacement_x = int(x - matches[1][0])
                                    displacement_y = int(y - matches[0][0])
                                    displacement_data[f"Point_{i + 1}"].append((displacement_x, displacement_y))
                            except (IndexError, ValueError):
                                displacement_data[f"Point_{i + 1}"].append((0, 0))
                    prev_gray = current_gray.copy()
                except cv2.error:
                    continue
            displacement_df = pd.DataFrame(displacement_data)
            displacement_df['Time'] = pd.Series(time_data)
            contour_displacement_df = pd.DataFrame(
                contour_displacement if contour_displacement else [(0, 0)],
                columns=['Contour_Displacement_X', 'Contour_Displacement_Y']
            )
            return displacement_df, contour_displacement_df
        except cv2.error as e:
            raise ProcessingError(str(e))
        finally:
            cap.release()
    
    def save_displacement_data(self, filepath: str = 'displacement_data.xlsx') -> str:
        if not self.displacement_data:
            raise NoDataError("displacement data")
        try:
            df = pd.DataFrame(self.displacement_data, columns=['Displacement_X', 'Displacement_Y'])
            df['Time'] = pd.Series(self.time_data)
            df.to_excel(filepath, index=False)
            return filepath
        except Exception as e:
            raise ProcessingError(f"Failed to save data: {str(e)}")
