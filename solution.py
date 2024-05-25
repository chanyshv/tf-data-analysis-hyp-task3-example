import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

chat_id = 506238275


def solution(x: np.array, y: np.array) -> bool:
    _, p_value = mannwhitneyu(x, y, use_continuity=True, method='asymptotic')
    return p_value < 0.125


def test_solution_with_multiple_samples(num_tests=10_000):
    data = pd.read_csv('hyp3_historical_data.csv')

    data_array = list(map(float, data.values[0][1:]))

    results = []

    for _ in range(num_tests):
        sample_1 = np.random.choice(data_array, 500, replace=False)
        sample_2 = np.random.choice(data_array, 500, replace=False)

        result = solution(sample_1, sample_2)
        results.append(result)

    errs = sum(results)
    print(errs / num_tests)


# test_solution_with_multiple_samples()
