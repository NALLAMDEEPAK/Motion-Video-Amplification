import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Union
import pandas as pd
import logging

from backend.utils.exceptions import NoDataError, InsufficientDataError

logger = logging.getLogger(__name__)


class PlotGenerator:
    THEME = {
        'bg_color': '#0a0a0f', 'paper_color': '#0a0a0f', 'font_color': '#e0e0e0',
        'grid_color': 'rgba(255,255,255,0.08)', 'primary_color': '#00d4ff',
        'secondary_color': '#ff6b35', 'tertiary_color': '#a855f7',
        'success_color': '#22c55e', 'line_width': 2
    }
    
    ERROR_TEMPLATE = """
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 300px; background: rgba(255, 107, 53, 0.1); border: 1px solid rgba(255, 107, 53, 0.3);
        border-radius: 12px; padding: 2rem; color: #ff6b35; font-family: 'Outfit', sans-serif;">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <h3 style="margin: 1rem 0 0.5rem; font-size: 1.25rem;">{title}</h3>
        <p style="margin: 0; color: #8888a0; text-align: center;">{message}</p>
    </div>
    """
    
    def __init__(self, theme: Optional[Dict] = None):
        if theme:
            self.THEME.update(theme)
    
    def _create_error_html(self, title: str, message: str) -> str:
        return self.ERROR_TEMPLATE.format(title=title, message=message)
    
    def _validate_data(self, data: Union[List, np.ndarray], min_length: int = 1, name: str = "data") -> np.ndarray:
        if data is None:
            raise NoDataError(name)
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            raise NoDataError(name)
        if len(arr) < min_length:
            raise InsufficientDataError(min_length, len(arr))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    
    def _validate_fps(self, fps: float) -> float:
        if fps is None or fps <= 0 or not np.isfinite(fps):
            return 30.0
        return float(fps)
    
    def _apply_theme(self, fig: go.Figure, title: str = '', height: int = 500) -> go.Figure:
        try:
            fig.update_layout(
                title=dict(text=title, font=dict(size=20, color=self.THEME['font_color'], family='Inter, sans-serif'), x=0.5, xanchor='center'),
                font=dict(color=self.THEME['font_color'], family='Inter, sans-serif'),
                plot_bgcolor=self.THEME['bg_color'], paper_bgcolor=self.THEME['paper_color'],
                height=height, margin=dict(l=60, r=40, t=80, b=60),
                legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1, font=dict(size=12)),
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_family='Inter, sans-serif')
            )
            fig.update_xaxes(showline=True, linewidth=1, linecolor='rgba(255,255,255,0.2)', gridcolor=self.THEME['grid_color'], zeroline=False, tickfont=dict(size=11))
            fig.update_yaxes(showline=True, linewidth=1, linecolor='rgba(255,255,255,0.2)', gridcolor=self.THEME['grid_color'], zeroline=False, tickfont=dict(size=11))
            return fig
        except Exception:
            return fig
    
    def plot_time_waveform_x(self, displacement_x: List, fps: float) -> str:
        try:
            displacement_x = self._validate_data(displacement_x, min_length=2, name="displacement X data")
            fps = self._validate_fps(fps)
            timestamps = np.arange(0.0, len(displacement_x)/fps, 1.0/fps, dtype=float)[:len(displacement_x)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=displacement_x, mode='lines', name='Displacement X',
                line=dict(color=self.THEME['primary_color'], width=self.THEME['line_width']),
                hovertemplate='Time: %{x:.3f}s<br>Displacement: %{y:.2f}px<extra></extra>'))
            fig = self._apply_theme(fig, 'Displacement X vs Time', height=500)
            fig.update_xaxes(title_text='Time (seconds)')
            fig.update_yaxes(title_text='Displacement (pixels)')
            return plotly.offline.plot(fig, include_plotlyjs='cdn', output_type='div')
        except (NoDataError, InsufficientDataError) as e:
            return self._create_error_html("No Data Available", str(e.message))
        except Exception as e:
            return self._create_error_html("Plotting Error", str(e))
    
    def plot_time_waveform_y(self, displacement_y: List, fps: float) -> str:
        try:
            displacement_y = self._validate_data(displacement_y, min_length=2, name="displacement Y data")
            fps = self._validate_fps(fps)
            timestamps = np.arange(0.0, len(displacement_y)/fps, 1.0/fps, dtype=float)[:len(displacement_y)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=displacement_y, mode='lines', name='Displacement Y',
                line=dict(color=self.THEME['secondary_color'], width=self.THEME['line_width']),
                hovertemplate='Time: %{x:.3f}s<br>Displacement: %{y:.2f}px<extra></extra>'))
            fig = self._apply_theme(fig, 'Displacement Y vs Time', height=500)
            fig.update_xaxes(title_text='Time (seconds)')
            fig.update_yaxes(title_text='Displacement (pixels)')
            return plotly.offline.plot(fig, include_plotlyjs='cdn', output_type='div')
        except (NoDataError, InsufficientDataError) as e:
            return self._create_error_html("No Data Available", str(e.message))
        except Exception as e:
            return self._create_error_html("Plotting Error", str(e))
    
    def plot_time_waveform_magnitude(self, displacement_x: List, displacement_y: List, fps: float) -> str:
        try:
            displacement_x = self._validate_data(displacement_x, min_length=2, name="displacement X data")
            displacement_y = self._validate_data(displacement_y, min_length=2, name="displacement Y data")
            fps = self._validate_fps(fps)
            min_len = min(len(displacement_x), len(displacement_y))
            displacement_x = displacement_x[:min_len]
            displacement_y = displacement_y[:min_len]
            timestamps = np.arange(0.0, min_len/fps, 1.0/fps, dtype=float)[:min_len]
            magnitude = np.sqrt(np.square(displacement_x) + np.square(displacement_y))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=magnitude, mode='lines', name='Displacement Magnitude',
                line=dict(color=self.THEME['tertiary_color'], width=self.THEME['line_width']),
                fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)',
                hovertemplate='Time: %{x:.3f}s<br>Magnitude: %{y:.2f}px<extra></extra>'))
            fig = self._apply_theme(fig, 'Displacement Magnitude vs Time', height=500)
            fig.update_xaxes(title_text='Time (seconds)')
            fig.update_yaxes(title_text='Magnitude (pixels)')
            return plotly.offline.plot(fig, include_plotlyjs='cdn', output_type='div')
        except (NoDataError, InsufficientDataError) as e:
            return self._create_error_html("No Data Available", str(e.message))
        except Exception as e:
            return self._create_error_html("Plotting Error", str(e))
    
    def plot_fft_x(self, displacement_x: List, sampling_rate: float) -> str:
        try:
            displacement_x = self._validate_data(displacement_x, min_length=2, name="displacement X data")
            if sampling_rate is None or sampling_rate <= 0:
                sampling_rate = 30.0
            fft_result = np.fft.fft(displacement_x)
            frequency_bins = np.fft.fftfreq(len(displacement_x), d=1.0/sampling_rate)
            positive_mask = frequency_bins >= 0
            freq_positive = frequency_bins[positive_mask]
            fft_positive = np.abs(fft_result)[positive_mask]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freq_positive, y=fft_positive, mode='lines', name='FFT X',
                line=dict(color=self.THEME['primary_color'], width=self.THEME['line_width']),
                hovertemplate='Frequency: %{x:.2f} Hz<br>Amplitude: %{y:.2f}<extra></extra>'))
            fig = self._apply_theme(fig, 'FFT - Displacement X', height=500)
            fig.update_xaxes(title_text='Frequency (Hz)')
            fig.update_yaxes(title_text='Amplitude')
            return plotly.offline.plot(fig, include_plotlyjs='cdn', output_type='div')
        except (NoDataError, InsufficientDataError) as e:
            return self._create_error_html("No Data Available", str(e.message))
        except Exception as e:
            return self._create_error_html("FFT Error", str(e))
    
    def plot_fft_y(self, displacement_y: List, sampling_rate: float) -> str:
        try:
            displacement_y = self._validate_data(displacement_y, min_length=2, name="displacement Y data")
            if sampling_rate is None or sampling_rate <= 0:
                sampling_rate = 30.0
            fft_result = np.fft.fft(displacement_y)
            frequency_bins = np.fft.fftfreq(len(displacement_y), d=1.0/sampling_rate)
            positive_mask = frequency_bins >= 0
            freq_positive = frequency_bins[positive_mask]
            fft_positive = np.abs(fft_result)[positive_mask]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freq_positive, y=fft_positive, mode='lines', name='FFT Y',
                line=dict(color=self.THEME['secondary_color'], width=self.THEME['line_width']),
                hovertemplate='Frequency: %{x:.2f} Hz<br>Amplitude: %{y:.2f}<extra></extra>'))
            fig = self._apply_theme(fig, 'FFT - Displacement Y', height=500)
            fig.update_xaxes(title_text='Frequency (Hz)')
            fig.update_yaxes(title_text='Amplitude')
            return plotly.offline.plot(fig, include_plotlyjs='cdn', output_type='div')
        except (NoDataError, InsufficientDataError) as e:
            return self._create_error_html("No Data Available", str(e.message))
        except Exception as e:
            return self._create_error_html("FFT Error", str(e))
    
    def plot_fft_magnitude(self, displacement_x: List, displacement_y: List, sampling_rate: float) -> str:
        try:
            displacement_x = self._validate_data(displacement_x, min_length=2, name="displacement X data")
            displacement_y = self._validate_data(displacement_y, min_length=2, name="displacement Y data")
            if sampling_rate is None or sampling_rate <= 0:
                sampling_rate = 30.0
            min_len = min(len(displacement_x), len(displacement_y))
            displacement_x = displacement_x[:min_len]
            displacement_y = displacement_y[:min_len]
            magnitude = np.sqrt(np.square(displacement_x) + np.square(displacement_y))
            fft_result = np.fft.fft(magnitude)
            frequency_bins = np.fft.fftfreq(len(magnitude), d=1.0/sampling_rate)
            positive_mask = frequency_bins >= 0
            freq_positive = frequency_bins[positive_mask]
            fft_positive = np.abs(fft_result)[positive_mask]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freq_positive, y=fft_positive, mode='lines', name='FFT Magnitude',
                line=dict(color=self.THEME['tertiary_color'], width=self.THEME['line_width']),
                hovertemplate='Frequency: %{x:.2f} Hz<br>Amplitude: %{y:.2f}<extra></extra>'))
            fig = self._apply_theme(fig, 'FFT - Displacement Magnitude', height=500)
            fig.update_xaxes(title_text='Frequency (Hz)')
            fig.update_yaxes(title_text='Amplitude')
            return plotly.offline.plot(fig, include_plotlyjs='cdn', output_type='div')
        except (NoDataError, InsufficientDataError) as e:
            return self._create_error_html("No Data Available", str(e.message))
        except Exception as e:
            return self._create_error_html("FFT Error", str(e))
    
    def plot_multipoint_analysis(self, displacement_df: pd.DataFrame, sampling_rate: float, fps: float, analysis_type: str = 'time') -> str:
        try:
            if displacement_df is None or displacement_df.empty:
                raise NoDataError("displacement data")
            fps = self._validate_fps(fps)
            if sampling_rate is None or sampling_rate <= 0:
                sampling_rate = fps * 100
            point_columns = [col for col in displacement_df.columns if col.startswith('Point_')]
            num_points = len(point_columns)
            if num_points == 0:
                raise NoDataError("point data")
            colors = [self.THEME['primary_color'], self.THEME['secondary_color'], self.THEME['tertiary_color'], self.THEME['success_color'], '#f59e0b', '#ec4899']
            fig = make_subplots(rows=num_points, cols=1, subplot_titles=[f'Point {i+1}' for i in range(num_points)], shared_xaxes=True, vertical_spacing=0.08)
            for i, col_name in enumerate(point_columns):
                try:
                    data = displacement_df[col_name].tolist()
                    if not data:
                        continue
                    if isinstance(data[0], tuple):
                        displacement_x = [point[0] for point in data]
                        displacement_y = [point[1] for point in data]
                    else:
                        displacement_x = data
                        displacement_y = [0] * len(data)
                    displacement_x = np.nan_to_num(np.asarray(displacement_x, dtype=float), nan=0.0)
                    displacement_y = np.nan_to_num(np.asarray(displacement_y, dtype=float), nan=0.0)
                    displacement_magnitude = np.sqrt(np.square(displacement_x) + np.square(displacement_y))
                    if analysis_type == 'time':
                        time_data = np.arange(0.0, len(displacement_x)/fps, 1.0/fps, dtype=float)[:len(displacement_x)]
                        x_axis = time_data
                        y_data_x = displacement_x
                        y_data_y = displacement_y
                        y_data_mag = displacement_magnitude
                    else:
                        if len(displacement_x) < 2:
                            continue
                        fft_x = np.fft.fft(displacement_x)
                        freq_x = np.fft.fftfreq(len(displacement_x), d=1.0/sampling_rate)
                        fft_y = np.fft.fft(displacement_y)
                        fft_mag = np.fft.fft(displacement_magnitude)
                        start_idx = min(10, len(freq_x) // 4)
                        x_axis = freq_x[start_idx:]
                        y_data_x = np.abs(fft_x)[start_idx:]
                        y_data_y = np.abs(fft_y)[start_idx:]
                        y_data_mag = np.abs(fft_mag)[start_idx:]
                    color_idx = i % len(colors)
                    fig.add_trace(go.Scatter(x=x_axis, y=y_data_x, mode='lines', name=f'X - Point {i+1}', line=dict(color=colors[color_idx], width=1.5)), row=i+1, col=1)
                    fig.add_trace(go.Scatter(x=x_axis, y=y_data_y, mode='lines', name=f'Y - Point {i+1}', line=dict(color=colors[(color_idx+1) % len(colors)], width=1.5)), row=i+1, col=1)
                    fig.add_trace(go.Scatter(x=x_axis, y=y_data_mag, mode='lines', name=f'Mag - Point {i+1}', line=dict(color=colors[(color_idx+2) % len(colors)], width=1.5)), row=i+1, col=1)
                except Exception:
                    continue
            title = 'Displacement Components vs Time' if analysis_type == 'time' else 'FFT Analysis'
            fig = self._apply_theme(fig, title, height=200 * num_points + 100)
            return plotly.offline.plot(fig, include_plotlyjs='cdn', output_type='div')
        except (NoDataError, InsufficientDataError) as e:
            return self._create_error_html("No Data Available", str(e.message))
        except Exception as e:
            return self._create_error_html("Plotting Error", str(e))
    
    def plot_modal_analysis(self, modal_data: Dict) -> str:
        try:
            if modal_data is None:
                raise NoDataError("modal analysis data")
            modal_coor = modal_data.get('modal_coordinates', np.array([]))
            modal_shapes = modal_data.get('modal_shapes', np.array([]))
            original_shape = modal_data.get('original_shape', (100, 100))
            if not isinstance(modal_coor, np.ndarray):
                modal_coor = np.array(modal_coor) if modal_coor is not None else np.array([])
            if not isinstance(modal_shapes, np.ndarray):
                modal_shapes = np.array(modal_shapes) if modal_shapes is not None else np.array([])
            num_modes = min(6, max(len(modal_coor) if modal_coor.size > 0 else 0, len(modal_shapes) if modal_shapes.size > 0 else 0))
            if num_modes == 0:
                raise NoDataError("modal data")
            fig = make_subplots(rows=num_modes, cols=2,
                subplot_titles=[f'Mode {i+1} Coordinates' if j == 0 else f'Mode {i+1} Shape' for i in range(num_modes) for j in range(2)],
                horizontal_spacing=0.1, vertical_spacing=0.08)
            for i in range(num_modes):
                try:
                    if modal_coor.size > 0 and i < len(modal_coor):
                        coor_data = np.nan_to_num(np.abs(modal_coor[i]), nan=0.0)
                        fig.add_trace(go.Scatter(y=coor_data, mode='lines', name=f'Modal Coordinates {i + 1}',
                            line=dict(color=self.THEME['primary_color'], width=1.5)), row=i + 1, col=1)
                except Exception:
                    pass
                try:
                    if modal_shapes.size > 0 and i < len(modal_shapes):
                        shape_data = np.nan_to_num(np.real(modal_shapes[i]), nan=0.0)
                        try:
                            if shape_data.size == original_shape[0] * original_shape[1]:
                                shape_data = shape_data.reshape(original_shape)
                            else:
                                side = int(np.sqrt(shape_data.size))
                                if side * side == shape_data.size:
                                    shape_data = shape_data.reshape(side, side)
                                else:
                                    shape_data = shape_data.reshape(-1, 1)
                        except Exception:
                            shape_data = shape_data.reshape(-1, 1)
                        fig.add_trace(go.Heatmap(z=shape_data, colorscale='Viridis', name=f'Modal Shape {i + 1}', showscale=False), row=i + 1, col=2)
                except Exception:
                    pass
            fig = self._apply_theme(fig, 'Modal Analysis', height=200 * num_modes + 100)
            fig.update_layout(width=900)
            return plotly.offline.plot(fig, include_plotlyjs='cdn', output_type='div')
        except (NoDataError, InsufficientDataError) as e:
            return self._create_error_html("No Data Available", str(e.message))
        except Exception as e:
            return self._create_error_html("Plotting Error", str(e))
