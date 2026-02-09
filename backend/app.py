import os
import logging
from flask import Flask, render_template
from backend.config import config


def create_app(config_name: str = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app_config = config.get(config_name, config['default'])
    
    app = Flask(
        __name__,
        template_folder=str(app_config.TEMPLATE_FOLDER),
        static_folder=str(app_config.STATIC_FOLDER),
        static_url_path='/static'
    )
    
    app.config.from_object(app_config)
    app.config['UPLOAD_FOLDER'] = app_config.UPLOAD_FOLDER
    app.config['ALLOWED_EXTENSIONS'] = app_config.ALLOWED_EXTENSIONS
    
    app_config.init_app(app)
    
    from backend.routes import analysis_bp, video_bp
    app.register_blueprint(video_bp)
    app.register_blueprint(analysis_bp)
    
    register_error_handlers(app)
    
    logger.info(f"MAV application created with config: {config_name}")
    
    return app


def register_error_handlers(app: Flask):
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('error.html', error_title='Page Not Found',
                              error_message='The page you are looking for does not exist.', error_code=404), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('error.html', error_title='Internal Server Error',
                              error_message='An unexpected error occurred.', error_code=500), 500

    @app.errorhandler(413)
    def file_too_large(error):
        return render_template('error.html', error_title='File Too Large',
                              error_message='File exceeds 500MB limit.', error_code=413), 413
