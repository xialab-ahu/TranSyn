"""Replot saved coordinates without refitting UMAP or rerunning the models."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

OUT = Path(__file__).resolve().parent


def main():
    rows = []
    for transfer in [True, False]:
        for modality in ['DNA', 'RNA']:
            prefix = f'transfer_{transfer}_{modality}'
            frames = {split: pd.read_csv(OUT / f'{prefix}_{split}_embedding.csv')
                      for split in ['train', 'test']}
            both = pd.concat(list(frames.values()))[['UMAP1', 'UMAP2']].to_numpy()
            # Same limits and equal distance scale for the paired train/test plots.
            lo, hi = both.min(axis=0), both.max(axis=0)
            center = (lo + hi) / 2
            half = max(hi - lo) / 2 + 1
            for split, frame in frames.items():
                xy = frame[['UMAP1', 'UMAP2']].to_numpy()
                labels = frame.label.to_numpy()
                score = silhouette_score(xy, labels, metric='euclidean')
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                for label, color in [(0, '#92ccd7'), (1, '#e70f1e')]:
                    mask = labels == label
                    ax.scatter(*xy[mask].T, s=70, color=color, alpha=.8,
                               edgecolors='none')
                ax.text(.04, .96,
                        # f'Silhouette score = {score:.4f}',
                        f'score: {score:.4f}',
                        transform=ax.transAxes, ha='left', va='top', fontsize=28,
                        color='#121B3E',
                        # bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2)
                        )
                ax.set_xlim(center[0]-half, center[0]+half)
                ax.set_ylim(center[1]-half, center[1]+half)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                fig.tight_layout(pad=.3)
                stem = f'{prefix}_{split}_umap'
                for ext in ['png', 'pdf']:
                    fig.savefig(OUT / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
                plt.close(fig)
                rows.append(dict(transfer=transfer, modality=modality, split=split,
                                 n_samples=len(labels), space='umap_2d',
                                 metric='euclidean', silhouette=score,
                                 figure=f'{stem}.png'))
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'figure_silhouette_scores.csv', index=False)
    heldout = pd.read_csv(OUT / 'held_out_separation.csv')
    for row in results[results.split == 'test'].itertuples():
        reference = heldout[(heldout.transfer == row.transfer) &
                            (heldout.modality == row.modality) &
                            (heldout.space == 'umap_2d')].silhouette.item()
        assert np.isclose(row.silhouette, reference, atol=1e-6)
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
