import numpy as np, pytest
from damageaware.contracts import default_checkpoints, load_meta, load_trace, N_CAP
import sys; sys.path.insert(0, 'scripts')
import make_stub_data as m
@pytest.fixture(scope='module')
def d(tmp_path_factory):
    p = tmp_path_factory.mktemp('stub'); m.make(p, n=120); return p
def test_meta_columns(d):
    meta = load_meta(d)
    for c in ['sample_id', 'label', 'split', 'split_random', 'cluster_id', 'n_events', 'total_damage_full']: assert c in meta.columns
def test_trace_shapes(d):
    meta = load_meta(d)
    for sid in meta.sample_id[:10]:
        t = load_trace(d, sid); assert t['tok'].shape == t['dmg'].shape and len(t['tok']) <= N_CAP
def test_no_cluster_crosses_splits(d):
    meta = load_meta(d); assert (meta.groupby('cluster_id').split.nunique() == 1).all()
def test_checkpoints():
    c = default_checkpoints(); assert c == sorted(c) and c[0] == 8 and c[-1] == N_CAP
def test_total_damage_matches_trace(d):
    meta = load_meta(d)
    max_checkpoint = max(default_checkpoints())
    for _, r in meta.head(10).iterrows():
        trace_dmg = load_trace(d, r.sample_id)['dmg']
        assert trace_dmg.sum() <= r.total_damage_full
        if r.n_events <= max_checkpoint:
            assert trace_dmg.sum() == r.total_damage_full
