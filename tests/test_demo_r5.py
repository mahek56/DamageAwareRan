from pathlib import Path

import pytest

from scripts.replay_demo import main


def test_replay_demo_writes_png(tmp_path):
    out = tmp_path / 'figures' / 'replay.png'
    status = main(
        [
            '--data-dir',
            'data/processed_stub',
            '--scores-dir',
            'results/scores',
            '--models',
            'bag',
            'gru',
            '--out',
            str(out),
        ]
    )
    assert status == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_replay_demo_unknown_sample_id_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match='Unknown sample_id'):
        main(
            [
                '--data-dir',
                'data/processed_stub',
                '--scores-dir',
                'results/scores',
                '--models',
                'bag',
                '--sample-id',
                'missing-sample',
                '--out',
                str(tmp_path / 'replay.png'),
            ]
        )
