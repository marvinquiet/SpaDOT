import anndata as ad
import numpy as np
import pandas as pd
from argparse import Namespace
import matplotlib.pyplot as plt

from SpaDOT.utils._analyze_utils import (
    _annotation_matches, _transition_order, plot_domains, plot_OT,
)


def test_cluster_color_uses_largest_annotation_overlap():
    data = ad.AnnData(
        X=np.ones((5, 1)),
        obs=pd.DataFrame({
            'timepoint': ['t0'] * 5,
            'kmeans': ['0', '0', '0', '1', '1'],
            'annotation': ['Heart', 'Heart', 'Liver', 'Liver', 'Heart'],
        }, index=[str(i) for i in range(5)]),
    )
    matches, colors = _annotation_matches(data)
    assert matches == {'t0_0': 'Heart', 't0_1': 'Heart'}
    assert colors['Heart'] != colors['Liver']


def test_transition_order_aligns_labels_without_changing_values():
    mass = np.array([[0.2, 0.8], [0.7, 0.3]])
    rows = ['t0_0', 't0_1']
    columns = ['t1_1', 't1_0']
    matches = {'t0_0': 'Heart', 't0_1': 'Liver',
               't1_0': 'Heart', 't1_1': 'Liver'}
    row_order, column_order = _transition_order(mass, rows, columns, rows, matches)
    assert row_order == rows
    assert column_order == ['t1_0', 't1_1']
    reordered = mass[:, [columns.index(name) for name in column_order]]
    np.testing.assert_array_equal(np.sort(reordered.ravel()), np.sort(mass.ravel()))


def test_domain_csv_and_original_single_panel_layout(tmp_path, monkeypatch):
    data = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame({
            'timepoint': ['t0', 't0'],
            'kmeans': ['0', '0'],
            'annotation': ['A\nB', 'A\nB'],
        }, index=['spot0', 'spot1']),
        obsm={'spatial': np.array([[0.0, 0.0], [1.0, 1.0]])},
    )
    saved = []
    original_savefig = plt.savefig

    def capture_layout(path, *args, **kwargs):
        figure = plt.gcf()
        saved.append((len(figure.axes), tuple(figure.get_size_inches()),
                      figure.axes[0].get_title()))
        return original_savefig(path, *args, **kwargs)

    monkeypatch.setattr(plt, 'savefig', capture_layout)
    plot_domains(Namespace(output_dir=str(tmp_path), prefix=''), data)
    assert saved == [(1, (5.0, 5.0), 'Time point: t0')]
    table = pd.read_csv(tmp_path / 'domains.csv')
    assert table['matched_annotation'].tolist() == ['A B', 'A B']
    assert len((tmp_path / 'domains.csv').read_text().splitlines()) == 3
    mapping = pd.read_csv(tmp_path / 'domain_display_mapping.csv', dtype=str)
    assert mapping[['domain_id', 'display_name']].values.tolist() == [['t0_0', 't0_0']]
    assert mapping['matched_annotation'].tolist() == ['A B']


def test_transition_plot_uses_original_conditional_score_and_preserves_raw_mass(tmp_path, monkeypatch):
    mass = np.array([[0.1, 0.4], [0.3, 0.2]])
    transition = ad.AnnData(
        X=mass,
        obs=pd.DataFrame(index=['t0_0', 't0_1']),
        var=pd.DataFrame(index=['t1_0', 't1_1']),
    )
    transition.write_h5ad(tmp_path / 'transition_table_0_1.h5ad')
    domains = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame({
            'timepoint': ['t0', 't0', 't1', 't1'],
            'day': [0, 0, 1, 1],
            'kmeans': ['0', '1', '0', '1'],
            'annotation': ['A', 'B', 'A', 'B'],
        }, index=['a', 'b', 'c', 'd']),
    )
    saved = []
    original_savefig = plt.savefig

    def capture_layout(path, *args, **kwargs):
        figure = plt.gcf()
        saved.append((len(figure.axes), tuple(figure.get_size_inches()),
                      figure.axes[0].get_title()))
        return original_savefig(path, *args, **kwargs)

    monkeypatch.setattr(plt, 'savefig', capture_layout)
    labels = {'t0_0': '1', 't0_1': '2', 't1_0': '1', 't1_1': '2'}
    plot_OT(Namespace(output_dir=str(tmp_path), prefix='', display_labels=labels), domains)
    assert saved == [(2, (1.6, 1.6), 'Transition Probability Dotplot')]
    ordered = pd.read_csv(tmp_path / 'transition_mass_ordered_0_1.csv', index_col=0)
    np.testing.assert_allclose(np.sort(ordered.to_numpy().ravel()), np.sort(mass.ravel()))
    score = pd.read_csv(tmp_path / 'transition_probability_min_ordered_0_1.csv', index_col=0)
    expected = np.minimum(mass / mass.sum(axis=1, keepdims=True),
                          mass / mass.sum(axis=0, keepdims=True))
    np.testing.assert_allclose(score.to_numpy(), expected)
    named = pd.read_csv(tmp_path / 'transition_probability_min_named_0_1.csv', index_col=0)
    assert named.index.astype(str).tolist() == ['1', '2']
    assert named.columns.astype(str).tolist() == ['1', '2']
    np.testing.assert_allclose(named.to_numpy(), expected)
    assert (tmp_path / 'transition_dotplot_0_1.png').is_file()
    assert (tmp_path / 'transition_probability_min_0_1.png').is_file()
    assert not (tmp_path / 'transition_raw_mass_0_1.png').exists()
