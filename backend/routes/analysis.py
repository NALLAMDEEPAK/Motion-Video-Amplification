import os
import logging
from flask import Blueprint, render_template, request, current_app
from werkzeug.utils import secure_filename

from backend.services import VideoProcessor, SignalAnalyzer, PlotGenerator
from backend.utils.exceptions import (
    NoFileUploadedError, InvalidFileTypeError,
    VideoNotFoundError, VideoReadError, InvalidROIError, ROICancelledError,
    InvalidPolygonError, PolygonCancelledError,
    NoDataError, InsufficientDataError, ModalAnalysisError
)
from backend.state import state

logger = logging.getLogger(__name__)
analysis_bp = Blueprint('analysis', __name__)

video_processor = VideoProcessor()
signal_analyzer = SignalAnalyzer()
plot_generator = PlotGenerator()


def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def create_error_html(title: str, message: str, details: str = None) -> str:
    details_html = f'<p style="font-size: 0.85rem; margin-top: 0.5rem;">{details}</p>' if details else ''
    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 400px; background: rgba(255, 107, 53, 0.1); border: 1px solid rgba(255, 107, 53, 0.3);
        border-radius: 16px; padding: 2rem; color: #ff6b35; font-family: 'Outfit', sans-serif; text-align: center;">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-bottom: 1rem;">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <h3 style="margin: 0 0 0.5rem; font-size: 1.5rem; font-weight: 600;">{title}</h3>
        <p style="margin: 0; color: #8888a0; max-width: 400px;">{message}</p>
        {details_html}
    </div>
    """


@analysis_bp.route('/timex', methods=['POST'])
def timex():
    try:
        use_sample = request.form.get('use_sample')
        if use_sample:
            from backend.config import Config
            sample_path = str(Config.FRONTEND_DIR / 'assets' / use_sample)
            if os.path.exists(sample_path):
                state.video_path = sample_path
                state.displacement_data = None
                state.save()
        elif 'video' in request.files:
            video_file = request.files['video']
            if video_file and video_file.filename:
                if not allowed_file(video_file.filename):
                    raise InvalidFileTypeError(video_file.filename, list(current_app.config['ALLOWED_EXTENSIONS']))
                filename = secure_filename(video_file.filename)
                video_path = os.path.join(str(current_app.config['UPLOAD_FOLDER']), filename)
                video_file.save(video_path)
                state.video_path = video_path
                state.displacement_data = None
                state.save()
        
        if not state.has_video():
            raise NoFileUploadedError()
        
        processing_option = request.form.get('processing_option', 'roi')
        state.processing_option = processing_option
        
        if processing_option == 'roi':
            result = video_processor.track_motion_roi(state.video_path)
            state.displacement_data = result['displacement_data']
            state.sampling_rate = result['sampling_rate']
            state.fps = result['fps']
            state.save()
            displacement_x = [x for x, _ in state.displacement_data]
            time_waveform = plot_generator.plot_time_waveform_x(displacement_x, state.fps)
        elif processing_option == 'points':
            points, sampling_rate, fps = video_processor.select_points_of_interest(state.video_path)
            state.points = points
            state.sampling_rate = sampling_rate
            state.fps = fps
            displacement_df, contour_df = video_processor.track_motion_polygon(state.video_path, points)
            state.displacement_df = displacement_df
            state.contour_df = contour_df
            state.save()
            time_waveform = plot_generator.plot_multipoint_analysis(displacement_df, int(sampling_rate), fps, 'time')
        else:
            raise ValueError(f"Unknown processing option: {processing_option}")
        
        return render_template('time_disx.html', time=time_waveform)
    except ROICancelledError:
        return render_template('time_disx.html', time=create_error_html("Selection Cancelled", "ROI selection was cancelled.", "Press ENTER or SPACE to confirm"))
    except InvalidROIError:
        return render_template('time_disx.html', time=create_error_html("Invalid Selection", "The selected region is invalid.", "Draw a rectangle with non-zero dimensions"))
    except PolygonCancelledError:
        return render_template('time_disx.html', time=create_error_html("Selection Cancelled", "Polygon selection was cancelled.", "Press Q to finish, R to reset, ESC to cancel"))
    except InvalidPolygonError:
        return render_template('time_disx.html', time=create_error_html("Invalid Polygon", "Not enough points selected.", "Select at least 3 points"))
    except NoFileUploadedError:
        return render_template('time_disx.html', time=create_error_html("No Video Uploaded", "Please upload a video file.", "Supported: MP4, AVI, MOV, MKV, WebM"))
    except InvalidFileTypeError as e:
        return render_template('time_disx.html', time=create_error_html("Invalid File Type", e.message, e.details))
    except (VideoNotFoundError, VideoReadError) as e:
        return render_template('time_disx.html', time=create_error_html("Video Error", e.message, e.details))
    except NoDataError:
        return render_template('time_disx.html', time=create_error_html("No Motion Detected", "No motion in selected region.", "Try a different region"))
    except Exception as e:
        logger.exception(f"Error in timex: {e}")
        return render_template('time_disx.html', time=create_error_html("Processing Error", str(e)))


@analysis_bp.route('/timey', methods=['POST'])
def timey():
    try:
        state._load_from_file()
        if not state.has_displacement_data():
            raise NoDataError("displacement data")
        displacement_y = [y for _, y in state.displacement_data]
        time_waveform = plot_generator.plot_time_waveform_y(displacement_y, state.fps)
        return render_template('time_disy.html', time=time_waveform)
    except NoDataError:
        return render_template('time_disy.html', time=create_error_html("No Data Available", "Upload and process a video first."))
    except Exception as e:
        logger.exception(f"Error in timey: {e}")
        return render_template('time_disy.html', time=create_error_html("Processing Error", str(e)))


@analysis_bp.route('/timem', methods=['POST'])
def timem():
    try:
        state._load_from_file()
        if not state.has_displacement_data():
            raise NoDataError("displacement data")
        displacement_x = [x for x, _ in state.displacement_data]
        displacement_y = [y for _, y in state.displacement_data]
        time_waveform = plot_generator.plot_time_waveform_magnitude(displacement_x, displacement_y, state.fps)
        return render_template('time_dism.html', time=time_waveform)
    except NoDataError:
        return render_template('time_dism.html', time=create_error_html("No Data Available", "Upload and process a video first."))
    except Exception as e:
        logger.exception(f"Error in timem: {e}")
        return render_template('time_dism.html', time=create_error_html("Processing Error", str(e)))


@analysis_bp.route('/fftx', methods=['POST'])
def fftx():
    try:
        state._load_from_file()
        if state.processing_option == 'points' and state.has_multipoint_data():
            time_waveform = plot_generator.plot_multipoint_analysis(state.displacement_df, int(state.sampling_rate), state.fps, 'fft')
        elif state.has_displacement_data():
            displacement_x = [x for x, _ in state.displacement_data]
            time_waveform = plot_generator.plot_fft_x(displacement_x, int(state.sampling_rate))
        else:
            raise NoDataError("displacement data")
        return render_template('fftx.html', time=time_waveform)
    except NoDataError:
        return render_template('fftx.html', time=create_error_html("No Data Available", "Upload and process a video first."))
    except Exception as e:
        logger.exception(f"Error in fftx: {e}")
        return render_template('fftx.html', time=create_error_html("FFT Error", str(e)))


@analysis_bp.route('/ffty', methods=['POST'])
def ffty():
    try:
        state._load_from_file()
        if not state.has_displacement_data():
            raise NoDataError("displacement data")
        displacement_y = [y for _, y in state.displacement_data]
        time_waveform = plot_generator.plot_fft_y(displacement_y, int(state.sampling_rate))
        return render_template('ffty.html', time=time_waveform)
    except NoDataError:
        return render_template('ffty.html', time=create_error_html("No Data Available", "Upload and process a video first."))
    except Exception as e:
        logger.exception(f"Error in ffty: {e}")
        return render_template('ffty.html', time=create_error_html("FFT Error", str(e)))


@analysis_bp.route('/fftm', methods=['POST'])
def fftm():
    try:
        state._load_from_file()
        if not state.has_displacement_data():
            raise NoDataError("displacement data")
        displacement_x = [x for x, _ in state.displacement_data]
        displacement_y = [y for _, y in state.displacement_data]
        time_waveform = plot_generator.plot_fft_magnitude(displacement_x, displacement_y, int(state.sampling_rate))
        return render_template('fftm.html', time=time_waveform)
    except NoDataError:
        return render_template('fftm.html', time=create_error_html("No Data Available", "Upload and process a video first."))
    except Exception as e:
        logger.exception(f"Error in fftm: {e}")
        return render_template('fftm.html', time=create_error_html("FFT Error", str(e)))


@analysis_bp.route('/mode', methods=['POST'])
def mode():
    try:
        state._load_from_file()
        if not state.has_video():
            raise NoDataError("video")
        modal_data = signal_analyzer.modal_analysis(state.video_path)
        time_waveform = plot_generator.plot_modal_analysis(modal_data)
        return render_template('mode.html', time=time_waveform)
    except NoDataError:
        return render_template('mode.html', time=create_error_html("No Video Available", "Upload a video first."))
    except InsufficientDataError:
        return render_template('mode.html', time=create_error_html("Insufficient Data", "Video too short for modal analysis.", "Use at least 10 frames"))
    except ModalAnalysisError as e:
        return render_template('mode.html', time=create_error_html("Modal Analysis Error", e.message, e.details))
    except Exception as e:
        logger.exception(f"Error in mode: {e}")
        return render_template('mode.html', time=create_error_html("Analysis Error", str(e)))


@analysis_bp.route('/psd', methods=['POST'])
def psd():
    try:
        state._load_from_file()
        if not state.has_video():
            raise NoDataError("video")
        modal_data = signal_analyzer.modal_analysis(state.video_path)
        time_waveform = plot_generator.plot_modal_analysis(modal_data)
        return render_template('psd.html', time=time_waveform)
    except NoDataError:
        return render_template('psd.html', time=create_error_html("No Video Available", "Upload a video first."))
    except Exception as e:
        logger.exception(f"Error in psd: {e}")
        return render_template('psd.html', time=create_error_html("PSD Error", str(e)))
