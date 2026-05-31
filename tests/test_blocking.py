import numpy as np

from nqs_vmc.statistics.blocking import blocking_analysis, blocking_error


def test_blocking_analysis_returns_rows():
    data = np.arange(16, dtype=float)
    rows = blocking_analysis(data)
    assert len(rows) >= 4
    assert rows[0]["n_blocks"] == 16
    assert rows[1]["block_size"] == 2
    assert blocking_error(data) >= 0
