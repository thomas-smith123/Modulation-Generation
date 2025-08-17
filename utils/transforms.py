from typing import Any, Callable, List, Literal, Optional, Tuple, Union
from torchsig.transforms import functional as F
from torchsig.transforms.functional import (
    FloatParameter,
    IntParameter,
    NumericParameter,
    to_distribution,
    uniform_continuous_distribution,
    uniform_discrete_distribution,
)
import numpy as np
from scipy import signal
class Spectrogram:
    """Calculates power spectral density over time

    Args:
        nperseg (:obj:`int`):
            Length of each segment. If window is str or tuple, is set to 256,
            and if window is array_like, is set to the length of the window.

        noverlap (:obj:`int`):
            Number of points to overlap between segments.
            If None, noverlap = nperseg // 8.

        nfft (:obj:`int`):
            Length of the FFT used, if a zero padded FFT is desired.
            If None, the FFT length is nperseg.

        window_fcn (:obj:`str`):
            Window to be used in spectrogram operation.
            Default value is 'np.blackman'.

        mode (:obj:`str`):
            Mode of the spectrogram to be computed.
            Default value is 'psd'.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Spectrogram with seg_size=256, overlap=64, nfft=256, window=blackman_harris
        >>> transform = ST.Spectrogram()
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=blackman_harris (2x oversampled in time)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64)
        >>> # Spectrogram with seg_size=128, overlap=0, nfft=128, window=blackman_harris (critically sampled)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=0)
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=blackman_harris (2x oversampled in frequency)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64, nfft=256)
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=rectangular
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64, nfft=256, window_fcn=np.ones)

    """

    def __init__(
        self,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        window_fcn: Callable[[int], np.ndarray] = np.hamming,#np.blackman,
        mode: str = "psd",
    ) -> None:
        super(Spectrogram, self).__init__()
        self.nperseg: int = nperseg
        self.noverlap: int = nperseg // 4 if noverlap is None else noverlap
        self.nfft: int = nperseg if nfft is None else nfft
        self.window_fcn = window_fcn
        self.mode = mode
        self.string = (
            self.__class__.__name__
            + "("
            + "nperseg={}, ".format(nperseg)
            + "noverlap={}, ".format(self.noverlap)
            + "nfft={}, ".format(self.nfft)
            + "window_fcn={}, ".format(window_fcn)
            + "mode={}".format(mode)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        ## 默认data为IQ数据
        data = F.spectrogram(
            ## nperseg为窗长，noverlap为重叠长度，nfft为FFT长度
            data, self.nperseg, self.noverlap, self.nfft, self.window_fcn, self.mode
        )
        if self.mode == "complex":
            new_tensor = np.zeros((2, data.shape[0], data.shape[1]), dtype=np.float32)
            new_tensor[0, :, :] = np.real(data).astype(np.float32)
            new_tensor[1, :, :] = np.imag(data).astype(np.float32)
            data = new_tensor
        return data
    
class Normalize:
    """Normalize a IQ vector with mean and standard deviation.

    Args:
        norm :obj:`string`:
            Type of norm with which to normalize

        flatten :obj:`flatten`:
            Specifies if the norm should be calculated on the flattened
            representation of the input tensor

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Normalize(norm=2) # normalize by l2 norm
        >>> transform = ST.Normalize(norm=1) # normalize by l1 norm
        >>> transform = ST.Normalize(norm=2, flatten=True) # normalize by l1 norm of the 1D representation

    """

    def __init__(
        self,
        norm: Optional[Union[int, float, Literal["fro", "nuc"]]] = 2,
        flatten: bool = False,
    ) -> None:
        super(Normalize, self).__init__()
        self.norm = norm
        self.flatten = flatten
        self.string: str = (
            self.__class__.__name__
            + "("
            + "norm={}, ".format(norm)
            + "flatten={}".format(flatten)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data):
        ## 默认数据为IQ数据
        data = F.normalize(data, self.norm, self.flatten)
        return data    