import logging
import numpy as np


def findPeaks(threshold_low, threshold_high, trace, time):
    # Detect where the trace crosses above the low threshold
    above_threshold = trace > threshold_low

    # Prepare results
    accepted_peak_t = []
    accepted_peak_amp = []
    rejected_peak_t = []
    rejected_peak_amp = []
    max_t = abs(time[-1])

    if len(above_threshold) == 0:
        return accepted_peak_t, accepted_peak_amp, rejected_peak_t, rejected_peak_amp

    transitions = np.diff(above_threshold.astype(int))
    peak_starts = np.where(transitions == 1)[0] + 1  # Rising edges (+1)
    peak_ends = np.where(transitions == -1)[0] + 1  # Falling edges (+1)

    # Batch processing peaks
    for start, end in zip(peak_starts, peak_ends):
        if end - start < 2:
            continue
        # Find max index within the current peak range
        peak_range = trace[start:end]
        max_idx = start + np.argmax(peak_range)

        # Ensure we have enough data points for parabolic fitting
        if max_idx > 0 and max_idx < len(trace) - 1:
            x = time[max_idx - 1 : max_idx + 2] / max_t
            y = trace[max_idx - 1 : max_idx + 2]

            # Parabola fitting using the predefined function
            a, b, c = ParabolaFit(x, y)
            t = -b / (2 * a)
            amp = a * t**2 + b * t + c
            t = t * max_t

            # Classify peaks by threshold
            if amp < threshold_high:
                accepted_peak_t.append(t)
                accepted_peak_amp.append(amp)
            else:
                rejected_peak_t.append(t)
                rejected_peak_amp.append(amp)

    return accepted_peak_t, accepted_peak_amp, rejected_peak_t, rejected_peak_amp


def ParabolaFit(x, y):
    # Fit a parabola y=ax**2+bx+c to input data

    V = np.vstack([x**2, x, np.ones(len(x))]).T
    a, b, c = np.linalg.lstsq(V, y, rcond=-1)[0]
    return a, b, c
