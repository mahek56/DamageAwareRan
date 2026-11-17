import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / 'src'):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _data_dir() -> Path:
    raw = os.environ.get('DAMAGEAWARE_DATA_DIR', str(ROOT / 'data' / 'processed_stub'))
    data_dir = Path(raw)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return data_dir


@pytest.fixture(scope='session')
def data_dir():
    path = _data_dir()
    if not path.exists():
        pytest.skip(f'DATA_DIR is missing: {path}')
    return path


if not _data_dir().exists():
    pytest.skip('DAMAGEAWARE_DATA_DIR is missing; stub data must be generated first.', allow_module_level=True)
