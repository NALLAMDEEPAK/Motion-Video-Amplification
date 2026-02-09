import cv2
import numpy as np
import os
from scipy.linalg import fractional_matrix_power
from sklearn.decomposition import NMF
from typing import Tuple, List, Dict
import logging

from backend.utils.exceptions import (
    VideoNotFoundError, VideoReadError, InvalidVideoError,
    ProcessingError, NoDataError, InsufficientDataError,
    FFTError, ModalAnalysisError
)

logger = logging.getLogger(__name__)


class SignalAnalyzer:
    def __init__(self, num_components: int = 6):
        self.num_components = num_components
    
    def _validate_signal(self, signal: np.ndarray, min_length: int = 2, name: str = "signal") -> None:
        if signal is None:
            raise NoDataError(name)
        signal = np.asarray(signal)
        if signal.size == 0:
            raise NoDataError(name)
        if len(signal) < min_length:
            raise InsufficientDataError(min_length, len(signal))
    
    def _validate_sampling_rate(self, sampling_rate: float) -> float:
        if sampling_rate is None or sampling_rate <= 0:
            return 30.0
        return float(sampling_rate)
    
    def perform_fft(self, signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        try:
            self._validate_signal(signal, min_length=2, name="signal for FFT")
            sampling_rate = self._validate_sampling_rate(sampling_rate)
            signal = np.asarray(signal, dtype=float)
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            fft_result = np.fft.fft(signal)
            frequency_bins = np.fft.fftfreq(len(signal), d=1.0/sampling_rate)
            return fft_result, frequency_bins
        except (NoDataError, InsufficientDataError):
            raise
        except Exception as e:
            raise FFTError(str(e))
    
    def perform_psd(self, signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        try:
            self._validate_signal(signal, min_length=2, name="signal for PSD")
            sampling_rate = self._validate_sampling_rate(sampling_rate)
            signal = np.asarray(signal, dtype=float)
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            fft_result = np.fft.fft(signal)
            frequency_bins = np.fft.fftfreq(len(signal), d=1.0/sampling_rate)
            psd = (1.0 / len(signal)) * np.square(np.abs(fft_result))
            positive_frequencies = frequency_bins[:len(frequency_bins)//2]
            positive_psd = psd[:len(psd)//2]
            return positive_frequencies, positive_psd
        except (NoDataError, InsufficientDataError):
            raise
        except Exception as e:
            raise FFTError(str(e))
    
    def calculate_displacement_magnitude(self, displacement_x: np.ndarray, displacement_y: np.ndarray) -> np.ndarray:
        try:
            self._validate_signal(displacement_x, min_length=1, name="displacement X")
            self._validate_signal(displacement_y, min_length=1, name="displacement Y")
            x = np.asarray(displacement_x, dtype=float)
            y = np.asarray(displacement_y, dtype=float)
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            x = np.nan_to_num(x, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            return np.sqrt(np.square(x) + np.square(y))
        except (NoDataError, InsufficientDataError):
            raise
        except Exception as e:
            raise ProcessingError(str(e))
    
    def time_lagged_covariance(self, X: np.ndarray, num_lags: int) -> np.ndarray:
        try:
            if X is None or X.size == 0:
                raise NoDataError("covariance input data")
            X = np.asarray(X, dtype=float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            N = X.shape[0]
            T = X.shape[1]
            if T <= num_lags:
                raise InsufficientDataError(num_lags + 1, T)
            L = T - num_lags
            R = np.zeros([num_lags, N, N])
            def center(x):
                mean = x.mean(axis=1, keepdims=True)
                return x - mean
            X0 = center(X[:, 0:L])
            for k in range(num_lags):
                Xk = center(X[:, k:(k+L)])
                R[k] = (1.0/L) * (X0.dot(Xk.T))
                R[k] = 0.5 * (R[k] + R[k].T)
            return R
        except (NoDataError, InsufficientDataError):
            raise
        except Exception as e:
            raise ProcessingError(str(e))
    
    def eigenvalue_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if matrix is None or matrix.size == 0:
                raise NoDataError("matrix for eigenvalue decomposition")
            matrix = np.asarray(matrix, dtype=float)
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            diagonal_matrix = np.diag(eigenvalues)
            return eigenvectors, diagonal_matrix
        except NoDataError:
            raise
        except np.linalg.LinAlgError as e:
            raise ProcessingError(f"Eigenvalue decomposition failed: {str(e)}")
        except Exception as e:
            raise ProcessingError(str(e))
    
    def complex_pca(self, data: np.ndarray, num_components: int) -> np.ndarray:
        try:
            if data is None or data.size == 0:
                raise NoDataError("data for CPCA")
            data = np.asarray(data, dtype=float)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            max_components = min(data.shape)
            num_components = min(num_components, max_components)
            if num_components < 1:
                num_components = 1
            U, s, Vh = np.linalg.svd(data, full_matrices=False)
            U = U[:, :num_components]
            s = np.diag(s[:num_components])
            Vh = Vh[:num_components, :]
            cpca_components = U @ s @ Vh
            return cpca_components
        except NoDataError:
            raise
        except np.linalg.LinAlgError as e:
            raise ProcessingError(f"CPCA failed: {str(e)}")
        except Exception as e:
            raise ProcessingError(str(e))
    
    def estimate_diagonalizing_matrix(self, S: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if not S or len(S) == 0:
                raise NoDataError("matrices for diagonalization")
            S = [np.asarray(m, dtype=float) for m in S if m is not None and np.asarray(m).size > 0]
            if len(S) == 0:
                raise NoDataError("valid matrices for diagonalization")
            average_covariance_matrix = np.mean(S, axis=0)
            average_covariance_matrix = np.nan_to_num(average_covariance_matrix, nan=0.0)
            eigenvalues, eigenvectors = np.linalg.eigh(average_covariance_matrix)
            indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]
            orthogonal_matrix = eigenvectors
            diagonal_matrix = np.diag(eigenvalues)
            return orthogonal_matrix, diagonal_matrix
        except NoDataError:
            raise
        except np.linalg.LinAlgError as e:
            raise ProcessingError(f"Diagonalization failed: {str(e)}")
        except Exception as e:
            raise ProcessingError(str(e))
    
    def modal_analysis(self, video_path: str) -> Dict:
        if not video_path:
            raise VideoNotFoundError("No video path provided")
        if not os.path.exists(video_path):
            raise VideoNotFoundError(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")
        try:
            frames = []
            frame_count = 0
            max_frames = 1000
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                try:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray_frame)
                    frame_count += 1
                except cv2.error:
                    continue
            cap.release()
            if len(frames) < 10:
                raise InsufficientDataError(10, len(frames))
            frames_array = np.array(frames)
            original_shape = frames_array[0].shape
            flattened_frames = frames_array.reshape(frames_array.shape[0], -1).astype(float)
            flattened_frames = np.nan_to_num(flattened_frames, nan=0.0)
            mean_frame = np.mean(flattened_frames, axis=0)
            modal_response_matrix = flattened_frames - mean_frame
            min_val = np.min(modal_response_matrix)
            if min_val < 0:
                modal_response_matrix = modal_response_matrix - min_val
            try:
                nmf_model = NMF(
                    n_components=min(self.num_components, modal_response_matrix.shape[0] - 1, modal_response_matrix.shape[1] - 1),
                    init='nndsvda', solver='cd', max_iter=200
                )
                W = nmf_model.fit_transform(modal_response_matrix)
                H = nmf_model.components_
            except Exception:
                W = modal_response_matrix[:, :self.num_components] if modal_response_matrix.shape[1] >= self.num_components else modal_response_matrix
                H = np.eye(min(self.num_components, modal_response_matrix.shape[1]))
            k = min(self.num_components, modal_response_matrix.shape[0] - 1)
            if k < 2:
                k = 2
            try:
                R = self.time_lagged_covariance(modal_response_matrix, k)
            except Exception:
                R = np.zeros((k, k, k))
            try:
                Vx, Xw = self.eigenvalue_decomposition(R[0] if len(R) > 0 else np.eye(k))
            except Exception:
                Vx = np.eye(k)
                Xw = np.eye(k)
            try:
                Vs = self.complex_pca(Vx, self.num_components)
            except Exception:
                Vs = Vx
            try:
                Xw_reg = Xw + np.eye(Xw.shape[0]) * 1e-10
                Xw_inv = fractional_matrix_power(Xw_reg, -0.5)
                if Vs.shape[0] == Xw_inv.shape[1] and Vs.shape[1] == modal_response_matrix.shape[0]:
                    x_k = Xw_inv @ np.transpose(Vs) @ modal_response_matrix
                else:
                    x_k = modal_response_matrix[:k, :]
            except Exception:
                x_k = modal_response_matrix[:k, :] if modal_response_matrix.shape[0] >= k else modal_response_matrix
            U_list = []
            try:
                for i in range(1, min(k, Vs.shape[0])):
                    u, s, v = np.linalg.svd(Vs, full_matrices=False)
                    U_list.append(u)
            except Exception:
                U_list = [np.eye(min(Vs.shape))]
            try:
                if U_list:
                    orthogonal_matrix, diagonal_matrix = self.estimate_diagonalizing_matrix(U_list)
                else:
                    orthogonal_matrix = np.eye(k)
                    diagonal_matrix = np.eye(k)
            except Exception:
                orthogonal_matrix = np.eye(k)
                diagonal_matrix = np.eye(k)
            try:
                Xw_inv_pos = fractional_matrix_power(Xw_reg, 0.5)
                if Vs.shape[1] == Xw_inv_pos.shape[0] and Xw_inv_pos.shape[1] == orthogonal_matrix.shape[0]:
                    A = Vs @ Xw_inv_pos @ orthogonal_matrix
                else:
                    A = Vs
                if orthogonal_matrix.shape[1] == x_k.shape[0]:
                    y_k = np.transpose(orthogonal_matrix) @ x_k
                else:
                    y_k = x_k
            except Exception:
                A = Vs
                y_k = x_k
            modal_coor = []
            for i in range(1, min(k + 1, A.shape[1] if A.ndim > 1 else 1)):
                try:
                    modal_coor_i = A[:, i] if A.ndim > 1 else A
                    modal_coor.append(modal_coor_i)
                except IndexError:
                    break
            modal_coor = np.array(modal_coor) if modal_coor else np.array([[0]])
            modal_shapes = []
            for i in range(min(6, y_k.shape[0] if y_k.ndim > 1 else 1)):
                try:
                    modal_shapes_i = y_k[i, :] if y_k.ndim > 1 else y_k
                    modal_shapes.append(modal_shapes_i)
                except IndexError:
                    break
            modal_shapes = np.array(modal_shapes) if modal_shapes else np.array([[0]])
            try:
                if A.ndim > 1 and y_k.ndim > 1 and A.shape[1] == y_k.shape[0]:
                    y = np.dot(A, y_k)
                else:
                    y = y_k
            except Exception:
                y = y_k
            return {
                'modal_coordinates': modal_coor, 'modal_shapes': modal_shapes,
                'original_shape': original_shape, 'frames_array': frames_array,
                'y': y, 'A': A, 'y_k': y_k
            }
        except (VideoNotFoundError, VideoReadError, InvalidVideoError, InsufficientDataError):
            raise
        except Exception as e:
            raise ModalAnalysisError(str(e))
        finally:
            if cap.isOpened():
                cap.release()
