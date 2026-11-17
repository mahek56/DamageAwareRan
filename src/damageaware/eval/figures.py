"""Headless plots for damage distributions and model rankings."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def _alpha_tag(alpha):
    return str(alpha).replace('.', 'p')


def write_figures(figure_data, out_dir):
    """Save a damage CDF and median ranking scatter for every alpha."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for alpha, models in figure_data.items():
        if not models:
            continue
        tag = _alpha_tag(alpha)

        figure, axis = plt.subplots()
        for model, values in sorted(models.items()):
            damage = np.sort(np.asarray(values['damage'], dtype=np.float64))
            if not damage.size:
                continue
            cdf = np.arange(1, damage.size + 1) / damage.size
            axis.step(damage, cdf, where='post', label=model)
        axis.set_xlabel('Damage before alert')
        axis.set_ylabel('Cumulative fraction')
        axis.set_title(
            f'Damage CDF (alpha={alpha}); never-caught traces use worst-case damage'
        )
        axis.legend()
        figure.tight_layout()
        cdf_path = out_dir / f'damage_cdf_alpha_{tag}.png'
        figure.savefig(cdf_path)
        plt.close(figure)
        paths.append(cdf_path)

        figure, axis = plt.subplots()
        for model, values in sorted(models.items()):
            axis.scatter(
                values['median_trace_fraction'],
                values['median_damage'],
                label=model,
            )
            axis.annotate(
                model,
                (values['median_trace_fraction'], values['median_damage']),
            )
        axis.set_xlabel('Median trace fraction at alert')
        axis.set_ylabel('Median damage before alert')
        axis.set_title(f'Model ranking comparison (alpha={alpha})')
        axis.legend()
        figure.tight_layout()
        ranking_path = out_dir / f'ranking_alpha_{tag}.png'
        figure.savefig(ranking_path)
        plt.close(figure)
        paths.append(ranking_path)
    return paths
