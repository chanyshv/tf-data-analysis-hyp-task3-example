import numpy as np
from scipy.stats import mannwhitneyu

chat_id = 506238275


def solution(x: np.array, y: np.array) -> bool:
    _, p_value = mannwhitneyu(x, y, use_continuity=True, method='asymptotic')

    return p_value < 0.09
