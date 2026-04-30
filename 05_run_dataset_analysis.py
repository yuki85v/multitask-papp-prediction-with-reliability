import os
import itertools
from collections import Counter
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Settings
    input_path = 'training_data/'
    out_dir = 'results_from_manuscript/dataset_analysis/'
    os.makedirs(out_dir, exist_ok=True)

    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)

    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']


    # Per-assay compound counts
    rows = []
    for ct in target_cols:
        rows.append({'Assay': ct, 'N_compounds': int(merge_df[ct].notna().sum())})
    rows.append({'Assay': 'Union', 'N_compounds': int(merge_df[target_cols].notna().any(axis=1).sum())})
    rows.append({'Assay': 'Total_rows', 'N_compounds': int(len(merge_df))})
    pd.DataFrame(rows).to_csv(f'{out_dir}Table_S1_assay_counts.csv', index=False)


    # Per-compound assay coverage
    n_per_compound = merge_df[target_cols].notna().sum(axis=1).values
    counter = Counter(n_per_compound.tolist())
    coverage_df = pd.DataFrame([
        {'N_assays_measured': k, 'N_compounds': int(counter.get(k, 0))}
        for k in range(len(target_cols) + 1)
    ])
    coverage_df.to_csv(f'{out_dir}Table_S2_assay_coverage_per_compound.csv', index=False)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(coverage_df['N_assays_measured'].values, coverage_df['N_compounds'].values,
           color='#4c72b0', edgecolor='black')
    for k, n in zip(coverage_df['N_assays_measured'].values, coverage_df['N_compounds'].values):
        if n > 0:
            ax.text(k, n, str(n), ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Number of assays measured per compound')
    ax.set_ylabel('Number of compounds')
    ax.set_xticks(coverage_df['N_assays_measured'].values)
    ax.set_title('Assay coverage per compound')
    fig.tight_layout()
    plt.savefig(f'{out_dir}Fig_S1_assay_coverage_per_compound.png', dpi=300)
    plt.close()


    # Intersection counts across non-empty subsets of assays
    masks = {ct: merge_df[ct].notna().values for ct in target_cols}
    inter_rows = []
    for r in range(1, len(target_cols) + 1):
        for combo in itertools.combinations(target_cols, r):
            in_combo = np.logical_and.reduce([masks[a] for a in combo])
            if len(combo) < len(target_cols):
                out_combo = np.logical_or.reduce([masks[a] for a in target_cols if a not in combo])
                exact = int(np.sum(in_combo & ~out_combo))
            else:
                exact = int(np.sum(in_combo))
            inclusive = int(np.sum(in_combo))
            inter_rows.append({
                'Combination': '+'.join(combo),
                'Size': r,
                'Exact_intersection': exact,
                'Inclusive_intersection': inclusive,
            })
    inter_df = pd.DataFrame(inter_rows).sort_values(['Size', 'Exact_intersection'], ascending=[True, False])
    inter_df.to_csv(f'{out_dir}Table_S3_intersection_counts.csv', index=False)


    # Pairwise overlap heatmap (inclusive)
    n_assays = len(target_cols)
    overlap_mat = np.zeros((n_assays, n_assays), dtype=int)
    for i, a in enumerate(target_cols):
        for j, b in enumerate(target_cols):
            overlap_mat[i, j] = int(np.sum(masks[a] & masks[b]))
    pd.DataFrame(overlap_mat, index=target_cols, columns=target_cols).to_csv(
        f'{out_dir}Table_S4_pairwise_overlap_counts.csv'
    )

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(overlap_mat, cmap='Blues')
    ax.set_xticks(range(n_assays)); ax.set_yticks(range(n_assays))
    ax.set_xticklabels(target_cols, rotation=45, ha='right'); ax.set_yticklabels(target_cols)
    for i in range(n_assays):
        for j in range(n_assays):
            color = 'white' if overlap_mat[i, j] > overlap_mat.max() * 0.5 else 'black'
            ax.text(j, i, str(overlap_mat[i, j]), ha='center', va='center', color=color, fontsize=9)
    ax.set_title('Pairwise compound overlap (inclusive)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(f'{out_dir}Fig_S3_pairwise_overlap_heatmap.png', dpi=300)
    plt.close()


    # Pairwise label correlation (on compounds measured in both assays)
    corr_mat = np.full((n_assays, n_assays), np.nan)
    count_mat = np.zeros((n_assays, n_assays), dtype=int)
    for i, a in enumerate(target_cols):
        for j, b in enumerate(target_cols):
            if i == j:
                count_mat[i, j] = int(merge_df[a].notna().sum())
                corr_mat[i, j] = 1.0
                continue
            both = merge_df[[a, b]].dropna()
            count_mat[i, j] = len(both)
            if len(both) >= 5:
                corr_mat[i, j] = both[a].corr(both[b], method='pearson')
    pd.DataFrame(corr_mat, index=target_cols, columns=target_cols).to_csv(
        f'{out_dir}Table_S5_pairwise_label_correlation.csv'
    )
    pd.DataFrame(count_mat, index=target_cols, columns=target_cols).to_csv(
        f'{out_dir}Table_S5b_pairwise_label_correlation_N.csv'
    )

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_assays)); ax.set_yticks(range(n_assays))
    ax.set_xticklabels(target_cols, rotation=45, ha='right'); ax.set_yticklabels(target_cols)
    for i in range(n_assays):
        for j in range(n_assays):
            val = corr_mat[i, j]; n = count_mat[i, j]
            txt = f'N={n}' if np.isnan(val) else f'{val:.2f}\nN={n}'
            color = 'white' if (not np.isnan(val) and abs(val) > 0.5) else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=8)
    ax.set_title('Pairwise Pearson correlation of log Papp\n(on compounds measured in both assays)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(f'{out_dir}Fig_S4_pairwise_label_correlation.png', dpi=300)
    plt.close()


    # Chemical-space relatedness: nearest-neighbour Tanimoto on Morgan fingerprints
    fp_df_path = f'{input_path}/fp_df.csv'
    if os.path.exists(fp_df_path):
        fp_df = pd.read_csv(fp_df_path)
        fp_cols = [c for c in fp_df.columns if c.startswith('fp_')]
        fp_arr = fp_df[fp_cols].values.astype(np.float32)
        compid_to_idx = {cid: k for k, cid in enumerate(fp_df['COMPID'].values)}
        bit_counts = fp_arr.sum(axis=1).astype(np.float32)

        assay_idx = {}
        for a in target_cols:
            ids = merge_df.loc[merge_df[a].notna(), 'COMPID'].values
            assay_idx[a] = np.array([compid_to_idx[c] for c in ids if c in compid_to_idx], dtype=int)

        asym_mean = np.full((n_assays, n_assays), np.nan)
        asym_median = np.full((n_assays, n_assays), np.nan)
        frac_gt_05 = np.full((n_assays, n_assays), np.nan)
        for i, a in enumerate(target_cols):
            for j, b in enumerate(target_cols):
                if i == j:
                    asym_mean[i, j] = 1.0; asym_median[i, j] = 1.0; frac_gt_05[i, j] = 1.0
                    continue
                src = assay_idx[a]
                tgt = np.setdiff1d(assay_idx[b], src, assume_unique=False)
                if len(src) == 0 or len(tgt) == 0:
                    continue
                src_fp = fp_arr[src]
                tgt_fp = fp_arr[tgt]
                src_bits = bit_counts[src]; tgt_bits = bit_counts[tgt]
                inter = src_fp @ tgt_fp.T
                union = src_bits[:, None] + tgt_bits[None, :] - inter
                with np.errstate(divide='ignore', invalid='ignore'):
                    sim = np.where(union > 0, inter / union, 0.0)
                maxes = sim.max(axis=1)
                asym_mean[i, j] = float(maxes.mean())
                asym_median[i, j] = float(np.median(maxes))
                frac_gt_05[i, j] = float(np.mean(maxes > 0.5))

        sym_mean = np.full((n_assays, n_assays), np.nan)
        for i in range(n_assays):
            for j in range(n_assays):
                if i == j:
                    sym_mean[i, j] = 1.0
                    continue
                sym_mean[i, j] = np.nanmean([asym_mean[i, j], asym_mean[j, i]])

        pd.DataFrame(sym_mean, index=target_cols, columns=target_cols).to_csv(
            f'{out_dir}Table_S5c_chemspace_relatedness_symmetric_mean.csv'
        )
        pd.DataFrame(asym_mean, index=target_cols, columns=target_cols).to_csv(
            f'{out_dir}Table_S5d_chemspace_relatedness_asym_mean.csv'
        )
        pd.DataFrame(asym_median, index=target_cols, columns=target_cols).to_csv(
            f'{out_dir}Table_S5e_chemspace_relatedness_asym_median.csv'
        )
        pd.DataFrame(frac_gt_05, index=target_cols, columns=target_cols).to_csv(
            f'{out_dir}Table_S5f_chemspace_frac_nn_over_0p5.csv'
        )

        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(sym_mean, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(n_assays)); ax.set_yticks(range(n_assays))
        ax.set_xticklabels(target_cols, rotation=45, ha='right'); ax.set_yticklabels(target_cols)
        for i in range(n_assays):
            for j in range(n_assays):
                val = sym_mean[i, j]
                txt = 'NA' if np.isnan(val) else f'{val:.2f}'
                color = 'white' if (not np.isnan(val) and val < 0.6) else 'black'
                ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)
        ax.set_title('Chemical-space relatedness\n(symmetric mean NN Tanimoto, Morgan r=2)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.savefig(f'{out_dir}Fig_S4b_chemspace_relatedness.png', dpi=300)
        plt.close()


    # Physicochemical property distributions
    prop_rows = []
    for smi in merge_df['SMILES'].values:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            prop_rows.append({'logP': np.nan, 'MW': np.nan, 'TPSA': np.nan, 'HBD': np.nan, 'HBA': np.nan})
            continue
        prop_rows.append({
            'logP': Crippen.MolLogP(mol),
            'MW': Descriptors.MolWt(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
        })
    physchem_df = pd.DataFrame(prop_rows)
    physchem_df['COMPID'] = merge_df['COMPID'].values
    for ct in target_cols:
        physchem_df[ct] = merge_df[ct].values
    physchem_df.to_csv(f'{out_dir}Table_S6_physchem_per_compound.csv', index=False)

    props = ['logP', 'MW', 'TPSA', 'HBD', 'HBA']
    summary_rows = []
    for ct in target_cols:
        sub = physchem_df[physchem_df[ct].notna()]
        row = {'Assay': ct, 'N': len(sub)}
        for p in props:
            vals = sub[p].dropna()
            row[f'{p}_mean'] = float(vals.mean())
            row[f'{p}_median'] = float(vals.median())
            row[f'{p}_std'] = float(vals.std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(f'{out_dir}Table_S7_physchem_summary_by_assay.csv', index=False)

    fig, axes = plt.subplots(1, len(props), figsize=(4.2 * len(props), 4.5))
    for k, p in enumerate(props):
        data = []
        labels = []
        for ct in target_cols:
            vals = physchem_df.loc[physchem_df[ct].notna(), p].dropna().values
            if len(vals) > 0:
                data.append(vals); labels.append(ct)
        parts = axes[k].violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        for b in parts['bodies']:
            b.set_alpha(0.6)
        axes[k].set_xticks(range(1, len(labels) + 1))
        axes[k].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        axes[k].set_title(p)
        axes[k].grid(True, axis='y', linestyle=':', alpha=0.5)
    fig.suptitle('Physicochemical property distributions by assay', y=1.02)
    fig.tight_layout()
    plt.savefig(f'{out_dir}Fig_S5_physchem_violin.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Label distribution summary
    label_rows = []
    for ct in target_cols:
        vals = merge_df[ct].dropna().values
        label_rows.append({
            'Assay': ct, 'N': len(vals),
            'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)),
            'median': float(np.median(vals)),
            'q25': float(np.quantile(vals, 0.25)), 'q75': float(np.quantile(vals, 0.75)),
            'min': float(np.min(vals)), 'max': float(np.max(vals)),
            'skew': float(pd.Series(vals).skew()), 'kurtosis': float(pd.Series(vals).kurtosis()),
        })
    pd.DataFrame(label_rows).to_csv(f'{out_dir}Table_S8_label_distribution_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.get_cmap('tab10').colors
    for idx, ct in enumerate(target_cols):
        vals = merge_df[ct].dropna().values
        ax.hist(vals, bins=np.arange(-3, 4.1, 0.25),
                alpha=0.45, label=f'{ct} (N={len(vals)})', color=colors[idx])
    ax.set_xlabel('log Papp')
    ax.set_ylabel('Count')
    ax.set_title('Label distribution across assays')
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(f'{out_dir}Fig_S6_label_distribution.png', dpi=300)
    plt.close()
