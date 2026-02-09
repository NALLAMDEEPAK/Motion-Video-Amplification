import os
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AppState:
    _instance = None
    _initialized = False
    
    def __new__(cls, state_file: Path = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, state_file: Path = None):
        if AppState._initialized:
            return
        AppState._initialized = True
        from backend.config import Config
        self.state_file = state_file or Config.STATE_FILE
        self.reset()
        self._load_from_file()
    
    def reset(self):
        self.video_path = None
        self.displacement_data = None
        self.sampling_rate = None
        self.fps = None
        self.processing_option = None
        self.points = None
        self.displacement_df = None
        self.contour_df = None
        self.last_error = None
    
    def _save_to_file(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state_dict = {
                'video_path': self.video_path,
                'displacement_data': self.displacement_data,
                'sampling_rate': self.sampling_rate,
                'fps': self.fps,
                'processing_option': self.processing_option,
                'points': self.points,
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(state_dict, f)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def _load_from_file(self):
        try:
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    state_dict = pickle.load(f)
                self.video_path = state_dict.get('video_path')
                self.displacement_data = state_dict.get('displacement_data')
                self.sampling_rate = state_dict.get('sampling_rate')
                self.fps = state_dict.get('fps')
                self.processing_option = state_dict.get('processing_option')
                self.points = state_dict.get('points')
                logger.info(f"State loaded: data_points={len(self.displacement_data) if self.displacement_data else 0}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    def save(self):
        self._save_to_file()
    
    def has_video(self) -> bool:
        return self.video_path is not None and os.path.exists(self.video_path)
    
    def has_displacement_data(self) -> bool:
        return self.displacement_data is not None and len(self.displacement_data) > 0
    
    def has_multipoint_data(self) -> bool:
        return self.displacement_df is not None and not self.displacement_df.empty


state = AppState()
