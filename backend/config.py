import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    BACKEND_DIR = Path(__file__).resolve().parent
    FRONTEND_DIR = BASE_DIR / 'frontend'
    DATA_DIR = BASE_DIR / 'data'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mav-defense-vibration-2024')
    UPLOAD_FOLDER = DATA_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv', 'flv'}
    STATE_FILE = UPLOAD_FOLDER / '.mav_state.pkl'
    TEMPLATE_FOLDER = FRONTEND_DIR / 'templates'
    STATIC_FOLDER = FRONTEND_DIR
    ASSETS_FOLDER = FRONTEND_DIR / 'assets'
    
    @classmethod
    def init_app(cls, app):
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    DEBUG = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
