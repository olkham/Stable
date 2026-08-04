"""Signal-processing primitives, in pure NumPy.

Everything the smoothing filters need — Butterworth design, zero-phase
``filtfilt``, Savitzky-Golay coefficients, a banded Cholesky solver for the
smoothing spline — implemented directly rather than pulled from SciPy.

Why not just depend on SciPy? Because ``numpy`` is this project's only hard
dependency, and a stabiliser that refuses to run because a 40 MB scientific
stack is missing is a worse product than one that carries four hundred lines of
well-understood DSP. The routines here follow the same standard formulations
SciPy uses (bilinear-transformed Butterworth, odd-extension ``filtfilt`` with
steady-state initial conditions, Reinsch's smoothing spline), and the test suite
checks them against SciPy whenever it happens to be installed.

Edge handling
-------------
Every filter here extends the signal by **odd reflection** about its endpoints
(``2*x[0] - x[k]``) rather than repeating or mirroring it. A camera move is
locally a ramp, and odd extension is the only cheap padding that leaves a ramp
unchanged — mirror padding turns the last second of a pan into a fake
deceleration, which is exactly the artefact users notice at the end of a shot.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.errors import SmoothingError

__all__ = [
    "odd_extension",
    "butter_lowpass_sos",
    "sos_steady_state",
    "sosfilt",
    "sosfiltfilt",
    "savgol_coeffs",
    "savgol_filter",
    "gaussian_kernel",
    "smoothing_spline",
    "cholesky_banded",
    "cho_solve_banded",
]


def odd_extension(values: np.ndarray, pad: int) -> np.ndarray:
    """Extend ``values`` at both ends by odd reflection about the endpoints.

    Args:
        values: ``[T]`` float array.
        pad: Number of samples to add at each end. Clipped to ``T - 1``.

    Returns:
        A ``[T + 2 * pad]`` array whose extension continues the signal's slope
        instead of mirroring it back.
    """
    array = np.asarray(values, dtype=np.float64)
    length = array.shape[0]
    pad = int(min(max(pad, 0), max(length - 1, 0)))
    if pad == 0:
        return array.copy()
    left = 2.0 * array[0] - array[pad:0:-1]
    right = 2.0 * array[-1] - array[-2 : -pad - 2 : -1]
    return np.concatenate([left, array, right])


# -- Butterworth --------------------------------------------------------------


def butter_lowpass_sos(order: int, cutoff_hz: float, fps: float) -> np.ndarray:
    """Design a digital Butterworth low-pass as second-order sections.

    The standard route: analogue Butterworth prototype poles, frequency
    pre-warped and scaled to the cutoff, mapped to the unit disc by the bilinear
    transform, then paired into biquads. The DC gain is normalised numerically
    at the end so a constant input passes through at exactly unit gain — which
    matters here, because a gain error on the scale channel is a zoom error.

    Args:
        order: Filter order, ``>= 1``. Each order doubles the roll-off; the
            zero-phase :func:`sosfiltfilt` application doubles it again.
        cutoff_hz: -3 dB cutoff frequency in Hz.
        fps: Sample rate in frames per second.

    Returns:
        ``[n_sections, 6]`` float64 array of ``[b0, b1, b2, 1, a1, a2]`` rows.

    Raises:
        SmoothingError: If the order is below 1, the sample rate is not
            positive, or the cutoff is not strictly between 0 and Nyquist.
    """
    order = int(order)
    if order < 1:
        raise SmoothingError(f"butterworth order must be >= 1, got {order}")
    if fps <= 0.0:
        raise SmoothingError(f"fps must be positive to design a filter, got {fps}")
    nyquist = fps / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise SmoothingError(
            f"cutoff_hz must lie strictly between 0 and Nyquist ({nyquist:g} Hz), "
            f"got {cutoff_hz:g}"
        )

    normalised = cutoff_hz / nyquist
    index = np.arange(order)
    prototype = np.exp(1j * np.pi * (2 * index + order + 1) / (2 * order))
    # Bilinear pre-warp with the conventional fs = 2 normalisation.
    warped = 4.0 * math.tan(math.pi * normalised / 2.0)
    analogue = prototype * warped
    digital = (4.0 + analogue) / (4.0 - analogue)

    # Pair conjugate poles into biquads; the numerator of every section is the
    # low-pass (1 + z^-1) factor pair contributed by the transform's zeros at -1.
    poles = sorted(digital, key=lambda p: (round(float(p.real), 12), abs(p.imag)))
    sections: list[list[float]] = []
    used = [False] * order
    for i, pole in enumerate(poles):
        if used[i]:
            continue
        used[i] = True
        if abs(pole.imag) < 1e-12:
            sections.append([1.0, 1.0, 0.0, 1.0, -float(pole.real), 0.0])
            continue
        for j in range(i + 1, order):
            if not used[j] and abs(poles[j] - np.conj(pole)) < 1e-9:
                used[j] = True
                break
        sections.append(
            [1.0, 2.0, 1.0, 1.0, -2.0 * float(pole.real), float(abs(pole) ** 2)]
        )

    sos = np.asarray(sections, dtype=np.float64)
    gain = np.prod((sos[:, 0] + sos[:, 1] + sos[:, 2]) / (1.0 + sos[:, 4] + sos[:, 5]))
    if not math.isfinite(gain) or gain == 0.0:  # pragma: no cover - unreachable design
        raise SmoothingError("butterworth design collapsed; try a lower order")
    sos[0, :3] /= gain
    return sos


def sos_steady_state(sos: np.ndarray) -> np.ndarray:
    """Return the per-section state that holds a unit constant input steady.

    Starting :func:`sosfilt` from ``sos_steady_state(sos) * x[0]`` removes the
    startup transient a zero-initialised filter would show — the equivalent of
    SciPy's ``sosfilt_zi``.

    Args:
        sos: ``[n_sections, 6]`` sections from :func:`butter_lowpass_sos`.

    Returns:
        ``[n_sections, 2]`` float64 initial delay-line state.
    """
    sections = np.asarray(sos, dtype=np.float64)
    state = np.zeros((sections.shape[0], 2), dtype=np.float64)
    level = 1.0
    for index, (b0, b1, b2, _a0, a1, a2) in enumerate(sections):
        gain = (b0 + b1 + b2) / (1.0 + a1 + a2)
        second = level * (b2 - a2 * gain)
        state[index, 0] = level * (b1 - a1 * gain) + second
        state[index, 1] = second
        level *= gain
    return state


def sosfilt(sos: np.ndarray, values: np.ndarray, zi: np.ndarray | None = None) -> np.ndarray:
    """Apply a cascade of second-order sections forward over ``values``.

    Args:
        sos: ``[n_sections, 6]`` sections.
        values: ``[T]`` input signal.
        zi: Optional ``[n_sections, 2]`` initial state, e.g. from
            :func:`sos_steady_state` scaled by ``values[0]``.

    Returns:
        ``[T]`` float64 filtered signal. Causal, so it lags — use
        :func:`sosfiltfilt` for anything a viewer will see.
    """
    sections = np.asarray(sos, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64).copy()
    state = (
        np.zeros((sections.shape[0], 2), dtype=np.float64)
        if zi is None
        else np.asarray(zi, dtype=np.float64).copy()
    )
    for index in range(sections.shape[0]):
        b0, b1, b2, _a0, a1, a2 = sections[index]
        z1, z2 = float(state[index, 0]), float(state[index, 1])
        out = np.empty_like(signal)
        for n in range(signal.shape[0]):
            sample = float(signal[n])
            filtered = b0 * sample + z1
            z1 = b1 * sample - a1 * filtered + z2
            z2 = b2 * sample - a2 * filtered
            out[n] = filtered
        state[index, 0], state[index, 1] = z1, z2
        signal = out
    return signal


def sosfiltfilt(sos: np.ndarray, values: np.ndarray, *, padlen: int | None = None) -> np.ndarray:
    """Apply ``sos`` forward and backward — zero phase, no lag.

    Filtering twice squares the magnitude response and cancels the phase
    response exactly, which is what a stabiliser needs: a causal filter would
    slide the whole shot late by a fraction of a second.

    Args:
        sos: ``[n_sections, 6]`` sections.
        values: ``[T]`` input signal.
        padlen: Samples of odd extension at each end; ``None`` derives it from
            the cascade's effective tap count, clipped to the signal length.

    Returns:
        ``[T]`` float64 zero-phase filtered signal. Signals shorter than three
        samples are returned unchanged — there is nothing to filter.
    """
    sections = np.asarray(sos, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    length = signal.shape[0]
    if length < 3:
        return signal.astype(np.float64, copy=True)
    # Effective tap count of the cascade, discounting sections that are really
    # first-order (an odd-order design contributes one); three times that is the
    # usual settling allowance.
    taps = 2 * sections.shape[0] + 1
    taps -= int(min(np.count_nonzero(sections[:, 2] == 0), np.count_nonzero(sections[:, 5] == 0)))
    pad = 3 * taps if padlen is None else int(padlen)
    pad = min(max(pad, 0), length - 1)

    extended = odd_extension(signal, pad)
    state = sos_steady_state(sections)
    forward = sosfilt(sections, extended, state * float(extended[0]))
    reversed_signal = forward[::-1]
    backward = sosfilt(sections, reversed_signal, state * float(reversed_signal[0]))
    result = backward[::-1]
    return result[pad : result.shape[0] - pad] if pad else result


# -- Savitzky-Golay -----------------------------------------------------------


def savgol_coeffs(window: int, polyorder: int, position: int) -> np.ndarray:
    """Return the FIR weights of a Savitzky-Golay fit evaluated at ``position``.

    Args:
        window: Window length in samples.
        polyorder: Degree of the polynomial fitted over the window.
        position: Index within the window at which the fitted polynomial is
            evaluated. ``window // 2`` gives the usual centred filter; other
            values give the one-sided fits used at the signal's ends.

    Returns:
        ``[window]`` float64 weights ``c`` such that ``c @ x_window`` is the
        fitted value at ``position``.

    Raises:
        SmoothingError: If the window is not positive or ``polyorder`` is not
            below the window length.
    """
    window, polyorder, position = int(window), int(polyorder), int(position)
    if window < 1:
        raise SmoothingError(f"savgol window must be >= 1, got {window}")
    if not 0 <= polyorder < window:
        raise SmoothingError(
            f"savgol polyorder must satisfy 0 <= polyorder < window ({window}), got {polyorder}"
        )
    offsets = np.arange(window, dtype=np.float64) - float(position)
    design = offsets[:, None] ** np.arange(polyorder + 1, dtype=np.float64)[None, :]
    # Row 0 of the pseudo-inverse maps the window onto the fitted constant term,
    # which is the polynomial's value at the offset origin — i.e. at `position`.
    return np.asarray(np.linalg.pinv(design)[0], dtype=np.float64)


def savgol_filter(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Smooth a signal with a Savitzky-Golay filter.

    Interior samples use the centred filter; the half-window at each end is
    filled by evaluating a fit over the first (or last) whole window at the
    right offset, so the ends are polynomial extrapolations rather than being
    dragged toward zero. This matches SciPy's ``mode="interp"``.

    Args:
        values: ``[T]`` input signal.
        window: Window length in samples; forced odd and clipped to ``T``.
        polyorder: Polynomial degree, which must be below ``window``.

    Returns:
        ``[T]`` float64 smoothed signal. A curve too short to hold a usable
        window is returned unchanged — there is nothing to fit, which is a
        different situation from being asked for an impossible fit.

    Raises:
        SmoothingError: If ``polyorder`` is not below the requested ``window``.
    """
    signal = np.asarray(values, dtype=np.float64)
    length = signal.shape[0]
    window, polyorder = int(window), int(polyorder)
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        raise SmoothingError(
            f"savgol needs polyorder < window, got polyorder {polyorder} with window {window}"
        )
    if window > length:
        window = length if length % 2 == 1 else length - 1
    if window < 3 or window <= polyorder:
        return signal.copy()

    half = window // 2
    centred = savgol_coeffs(window, polyorder, half)
    result = np.correlate(signal, centred, mode="same")
    head = signal[:window]
    tail = signal[length - window :]
    for index in range(half):
        result[index] = float(savgol_coeffs(window, polyorder, index) @ head)
        result[length - 1 - index] = float(
            savgol_coeffs(window, polyorder, window - 1 - index) @ tail
        )
    return result


# -- Gaussian -----------------------------------------------------------------


def gaussian_kernel(
    sigma: float, *, truncate: float = 4.0, max_radius: int | None = None
) -> np.ndarray:
    """Return a normalised discrete Gaussian kernel.

    Args:
        sigma: Standard deviation in samples; must be positive.
        truncate: Kernel radius in standard deviations.
        max_radius: Optional cap on the radius, so a very wide filter on a short
            curve stays a (renormalised) weighted average instead of demanding
            more padding than the curve can supply.

    Returns:
        ``[2 * radius + 1]`` float64 weights summing to 1.

    Raises:
        SmoothingError: If ``sigma`` is not positive.
    """
    if sigma <= 0.0:
        raise SmoothingError(f"gaussian sigma must be positive, got {sigma}")
    radius = max(int(math.ceil(truncate * sigma)), 1)
    if max_radius is not None:
        radius = max(min(radius, int(max_radius)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


# -- banded linear algebra + smoothing spline ---------------------------------


def cholesky_banded(ab: np.ndarray) -> np.ndarray:
    """Cholesky-factor a symmetric positive-definite matrix in lower band form.

    Args:
        ab: ``[p + 1, n]`` lower-band storage, where ``ab[i, j] = A[j + i, j]``
            and ``p`` is the bandwidth (``p = 2`` for a pentadiagonal matrix).

    Returns:
        ``[p + 1, n]`` lower-band storage of ``L`` with ``A = L @ L.T``.

    Raises:
        SmoothingError: If the matrix is not positive definite.
    """
    band = np.asarray(ab, dtype=np.float64).copy()
    bandwidth, size = band.shape[0] - 1, band.shape[1]
    for column in range(size):
        total = band[0, column] - float(
            sum(
                band[column - k, k] ** 2
                for k in range(max(0, column - bandwidth), column)
                if column - k <= bandwidth
            )
        )
        if total <= 0.0:
            raise SmoothingError("banded matrix is not positive definite")
        pivot = math.sqrt(total)
        band[0, column] = pivot
        for offset in range(1, min(bandwidth, size - 1 - column) + 1):
            row = column + offset
            acc = band[offset, column] - float(
                sum(
                    band[row - k, k] * band[column - k, k]
                    for k in range(max(0, row - bandwidth), column)
                    if row - k <= bandwidth and column - k <= bandwidth
                )
            )
            band[offset, column] = acc / pivot
    return band


def cho_solve_banded(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``A x = rhs`` given ``A``'s banded Cholesky factor.

    Args:
        lower: ``[p + 1, n]`` factor from :func:`cholesky_banded`.
        rhs: ``[n]`` right-hand side.

    Returns:
        ``[n]`` float64 solution.
    """
    band = np.asarray(lower, dtype=np.float64)
    bandwidth, size = band.shape[0] - 1, band.shape[1]
    y = np.asarray(rhs, dtype=np.float64).copy()
    for i in range(size):
        for k in range(max(0, i - bandwidth), i):
            y[i] -= band[i - k, k] * y[k]
        y[i] /= band[0, i]
    x = y
    for i in range(size - 1, -1, -1):
        for offset in range(1, min(bandwidth, size - 1 - i) + 1):
            x[i] -= band[offset, i] * x[i + offset]
        x[i] /= band[0, i]
    return x


def smoothing_spline(values: np.ndarray, lam: float) -> np.ndarray:
    """Fit a natural cubic smoothing spline to a uniformly sampled signal.

    Reinsch's algorithm: minimise ``sum (y_i - f_i)^2 + lam * integral f''^2``.
    ``lam = 0`` returns the input untouched; large ``lam`` drives the result
    toward the least-squares straight line through the whole curve.

    Args:
        values: ``[T]`` input signal, sampled once per frame.
        lam: Non-negative roughness penalty.

    Returns:
        ``[T]`` float64 fitted signal.

    Raises:
        SmoothingError: If ``lam`` is negative.
    """
    if lam < 0.0:
        raise SmoothingError(f"spline lam must be >= 0, got {lam}")
    signal = np.asarray(values, dtype=np.float64)
    length = signal.shape[0]
    if lam == 0.0 or length < 4:
        return signal.astype(np.float64, copy=True)

    # With unit spacing the normal equations R + lam Q'Q are pentadiagonal with
    # constant bands, so the whole solve is O(T).
    size = length - 2
    band = np.zeros((3, size), dtype=np.float64)
    band[0, :] = 2.0 / 3.0 + 6.0 * lam
    band[1, : size - 1] = 1.0 / 6.0 - 4.0 * lam
    band[2, : size - 2] = lam
    rhs = signal[:-2] - 2.0 * signal[1:-1] + signal[2:]

    gamma = cho_solve_banded(cholesky_banded(band), rhs)
    padded = np.zeros(length + 2, dtype=np.float64)
    padded[2 : 2 + size] = gamma
    correction = padded[2:] - 2.0 * padded[1 : length + 1] + padded[:length]
    return signal - lam * correction
