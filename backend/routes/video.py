import subprocess
import logging
from flask import Blueprint, render_template, current_app

logger = logging.getLogger(__name__)
video_bp = Blueprint('video', __name__)


def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@video_bp.route('/')
def home():
    return render_template('home.html')


@video_bp.route('/ovsa')
def ovsa():
    try:
        result = run_magnification_script()
        logger.info(f"Magnification result: {result[:100] if result else 'None'}...")
    except Exception as e:
        logger.error(f"Magnification script error: {e}")
    return render_template('ovsa.html')


def run_magnification_script():
    from backend.config import Config
    colab_script_path = Config.BASE_DIR / 'amplify.ipynb'
    
    if not colab_script_path.exists():
        return "Notebook file not found"
    
    try:
        from nbconvert import PythonExporter
        from nbformat import read
        
        with open(colab_script_path, 'r', encoding='utf-8') as nb_file:
            notebook = read(nb_file, as_version=4)
        
        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(notebook)
        script = script.replace("cd motion_magnification_learning-based", "")
        
        temp_script_path = Config.BASE_DIR / 'temp_script.py'
        with open(temp_script_path, 'w', encoding='utf-8') as py_file:
            py_file.write(script)
        
        command = ['python', str(temp_script_path)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(timeout=300)
        
        return output.decode('utf-8') if process.returncode == 0 else f"Error: {error.decode('utf-8')}"
    except subprocess.TimeoutExpired:
        return "Script execution timed out"
    except Exception as e:
        return f"Exception: {str(e)}"
