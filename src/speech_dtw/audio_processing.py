import numpy as np
from python_speech_features import mfcc, delta
from scipy.io import wavfile


def load_wav(path: str) -> tuple[int, np.ndarray]:
    """Load a WAV file and return sample rate and mono float32 signal."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    # Convert to float32 in range [-1, 1]
    if data.dtype.kind in {"i", "u"}:
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return rate, data


def compute_mfcc_features(
    signal: np.ndarray,
    sample_rate: int,
    numcep: int = 13,
    nfft: int = 2048,
    winlen: float = 0.025,
    winstep: float = 0.01,
) -> np.ndarray:
    """
    Compute 39-dimensional MFCC features (static + delta + delta-delta).

    Returns an array with shape (num_frames, 39).
    """
    base = mfcc(
        signal,
        samplerate=sample_rate,
        numcep=numcep,
        nfilt=26,
        nfft=nfft,
        winlen=winlen,
        winstep=winstep,
        preemph=0.97,
        appendEnergy=True,
    )
    delta_feat = delta(base, 2)
    delta_delta = delta(delta_feat, 2)
    return np.hstack((base, delta_feat, delta_delta)).astype(np.float32)
