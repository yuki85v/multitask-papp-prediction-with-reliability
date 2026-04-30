import os
import pickle
import pandas as pd
import numpy as np
import yaml
import torch
from torch import nn
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score, brier_score_loss, mean_absolute_error,
    r2_score, roc_auc_score, root_mean_squared_error,
)
from dgllife.utils import SMILESToBigraph
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import matplotlib.pyplot as plt

from utils_for_admet_model.models import GraphEncoderGCN, DNN
from utils_for_admet_model.utils import fix_seed


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if __name__ == "__main__":

    fix_seed(42)

    # Settings
    input_path = 'training_data/'
    model_path = 'trained_model/'
    ad_dir = f'{model_path}/random_split/multi_task_cv_ad'
    out_dir = 'results_from_manuscript/revision_visualizations/'
    os.makedirs(out_dir, exist_ok=True)

    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    log10_2 = float(np.log10(2.0))

    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)


    # Uncertainty / calibration metrics per assay
    rows = []
    reliability_rows = []
    enrichment_rows = []

    fig_rel, axes_rel = plt.subplots(1, 5, figsize=(22, 4), sharex=True, sharey=True)
    fig_enr, axes_enr = plt.subplots(1, 5, figsize=(22, 4), sharex=True, sharey=True)

    for i, target_col in enumerate(target_cols):
        err_model_path = f'{ad_dir}/error_model/err_model_{target_col}.pkl'
        test_ad_path = f'{ad_dir}/test_ad_df_{target_col}.csv'
        if not (os.path.exists(err_model_path) and os.path.exists(test_ad_path)):
            continue
        err_model = pickle.load(open(err_model_path, 'rb'))
        test_ad_df = pd.read_csv(test_ad_path, dtype={'COMPID': str})

        err_features = [
            f'{target_col}_pred_std',
            f'{target_col}_wRMSD1',
            f'{target_col}_wRMSD2',
            f'{target_col}_SIM1',
            f'{target_col}_SIM5',
            f'{target_col}_pred_mean',
        ]
        X = test_ad_df[err_features]
        ue = test_ad_df['UE'].values
        y_within2f = (ue < log10_2).astype(int)
        classes = list(err_model.classes_)
        within_idx = classes.index(0) if 0 in classes else 0
        reliability = err_model.predict_proba(X)[:, within_idx]

        try:
            auroc = float(roc_auc_score(y_within2f, reliability))
        except ValueError:
            auroc = float('nan')
        try:
            pr_auc = float(average_precision_score(y_within2f, reliability))
        except ValueError:
            pr_auc = float('nan')
        brier = float(brier_score_loss(y_within2f, reliability))

        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        n = len(reliability)
        for k in range(len(bins) - 1):
            lo, hi = bins[k], bins[k + 1]
            if k == len(bins) - 2:
                mask = (reliability >= lo) & (reliability <= hi)
            else:
                mask = (reliability >= lo) & (reliability < hi)
            if mask.sum() == 0:
                continue
            conf = float(reliability[mask].mean())
            acc = float(y_within2f[mask].mean())
            ece += (mask.sum() / n) * abs(conf - acc)

        rows.append({
            'Assay': target_col, 'N_test': int(n),
            'frac_within_2fold': float(y_within2f.mean()),
            'AUROC': auroc, 'PR_AUC': pr_auc, 'Brier': brier, 'ECE': float(ece),
        })

        # Reliability diagram data
        bin_mids, bin_confs, bin_accs, bin_counts = [], [], [], []
        for k in range(len(bins) - 1):
            lo, hi = bins[k], bins[k + 1]
            mask = (reliability >= lo) & (reliability <= hi) if k == len(bins) - 2 else (reliability >= lo) & (reliability < hi)
            if mask.sum() == 0:
                continue
            bin_mids.append((lo + hi) / 2)
            bin_confs.append(float(reliability[mask].mean()))
            bin_accs.append(float(y_within2f[mask].mean()))
            bin_counts.append(int(mask.sum()))
        for lo, hi, c, acc, cnt in zip(bins[:-1], bins[1:], bin_confs, bin_accs, bin_counts):
            reliability_rows.append({
                'Assay': target_col, 'bin_lo': float(lo), 'bin_hi': float(hi),
                'mean_confidence': c, 'mean_accuracy': acc, 'n_in_bin': cnt,
            })

        axes_rel[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='perfect')
        axes_rel[i].bar(bin_mids, bin_accs, width=0.08, alpha=0.6, edgecolor='black', label='observed acc')
        axes_rel[i].plot(bin_confs, bin_accs, 'ro-', markersize=4, label='bin mean conf')
        axes_rel[i].set_title(f'{target_col}\nAUROC={auroc:.3f}, ECE={ece:.3f}')
        axes_rel[i].set_xlim([0, 1]); axes_rel[i].set_ylim([0, 1])
        axes_rel[i].set_xlabel('Reliability score (p within 2-fold)')
        if i == 0:
            axes_rel[i].set_ylabel('Fraction actually within 2-fold')
        axes_rel[i].legend(fontsize=7, loc='upper left')

        # Coverage-vs-RMSE curve
        y_obs = test_ad_df[target_col].values
        y_pred = test_ad_df[f'{target_col}_pred_mean'].values
        order = np.argsort(-reliability)
        cum_rmse = []
        cum_coverage = []
        for k in range(1, n + 1):
            sel = order[:k]
            rmse_k = float(root_mean_squared_error(y_obs[sel], y_pred[sel]))
            cum_rmse.append(rmse_k)
            cum_coverage.append(k / n)
            if k % max(1, n // 20) == 0 or k == n:
                enrichment_rows.append({
                    'Assay': target_col, 'coverage_frac': k / n, 'k': k,
                    'RMSE_cum': rmse_k,
                    'R2_cum': float(r2_score(y_obs[sel], y_pred[sel])) if len(sel) > 1 else float('nan'),
                    'MAE_cum': float(mean_absolute_error(y_obs[sel], y_pred[sel])),
                })

        axes_enr[i].plot(cum_coverage, cum_rmse, 'b-')
        axes_enr[i].axhline(
            float(root_mean_squared_error(y_obs, y_pred)),
            color='red', linestyle='--', label='overall RMSE',
        )
        axes_enr[i].set_title(f'{target_col}\nN_test={n}')
        axes_enr[i].set_xlabel('Cumulative coverage (top-k by reliability)')
        if i == 0:
            axes_enr[i].set_ylabel('Cumulative RMSE')
        axes_enr[i].legend(fontsize=8)

    fig_rel.suptitle('Error-model reliability diagrams', y=1.02)
    fig_rel.tight_layout()
    plt.savefig(f'{out_dir}Fig_U1_reliability_diagrams.png', dpi=300, bbox_inches='tight')
    plt.close(fig_rel)

    fig_enr.suptitle('Coverage vs RMSE as reliability threshold tightens', y=1.02)
    fig_enr.tight_layout()
    plt.savefig(f'{out_dir}Fig_U2_coverage_vs_rmse.png', dpi=300, bbox_inches='tight')
    plt.close(fig_enr)

    pd.DataFrame(rows).to_csv(f'{out_dir}Table_U1_uncertainty_metrics.csv', index=False)
    pd.DataFrame(reliability_rows).to_csv(f'{out_dir}Table_U2_reliability_bins.csv', index=False)
    pd.DataFrame(enrichment_rows).to_csv(f'{out_dir}Table_U3_coverage_rmse.csv', index=False)


    # Parity plots with 2-fold bands and gray-scale fold colouring (10-fold CV)
    for split_pattern, fig_name, table_name, title in [
        ('random_split_cv', 'Fig_P1_parity_with_errorbars_random.png',
         'Table_P1_parity_summary_random.csv', 'MT-GCN 10-fold CV parity (random split)'),
        ('scaffold_split_cv', 'Fig_P2_parity_with_errorbars_scaffold.png',
         'Table_P2_parity_summary_scaffold.csv', 'MT-GCN 10-fold CV parity (scaffold split)'),
    ]:
        frames = []
        for fold in range(1, 11):
            fp = f'{model_path}/{split_pattern}/fold_{fold}/multi_task_cv/test_pred_group_df.csv'
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            df['Fold'] = fold
            frames.append(df)
        if not frames:
            continue
        pred_df = pd.concat(frames, ignore_index=True)
        pred_df = pred_df.merge(merge_df[['COMPID'] + target_cols], on='COMPID', how='left')

        n_folds = 10
        cmap = plt.get_cmap('Greys')
        fold_colors = [cmap(0.20 + 0.70 * (k / (n_folds - 1))) for k in range(n_folds)]

        fig, axes = plt.subplots(1, 5, figsize=(25, 5.3))
        summary_rows = []
        for i, target_col in enumerate(target_cols):
            sub = pred_df[~pred_df[target_col].isna()]
            if sub.empty:
                continue
            obs_all = sub[target_col].values
            pred_all = sub[f'{target_col}_pred_mean'].values
            std_all = sub[f'{target_col}_pred_std'].values
            fold_all = sub['Fold'].values.astype(int)

            data_lo = float(min(obs_all.min(), pred_all.min()) - 0.3)
            lo = max(-6.0, data_lo)
            hi = float(max(obs_all.max(), pred_all.max()) + 0.3)

            for k in range(1, n_folds + 1):
                mask = fold_all == k
                if not mask.any():
                    continue
                axes[i].errorbar(
                    obs_all[mask], pred_all[mask], yerr=std_all[mask],
                    fmt='o', markersize=3, alpha=0.75,
                    color=fold_colors[k - 1], ecolor=fold_colors[k - 1],
                    capsize=0, elinewidth=0.4, markeredgewidth=0.0,
                    label=f'fold {k}',
                )

            xs = np.array([lo, hi])
            axes[i].plot(xs, xs, 'k--', alpha=0.8, linewidth=0.8)
            axes[i].plot(xs, xs + log10_2, 'r--', alpha=0.6, linewidth=0.8)
            axes[i].plot(xs, xs - log10_2, 'r--', alpha=0.6, linewidth=0.8)
            axes[i].set_xlim([lo, hi]); axes[i].set_ylim([lo, hi])
            axes[i].set_aspect('equal', adjustable='box')
            axes[i].set_xlabel('Observed log Papp')
            axes[i].set_ylabel('Predicted log Papp')
            axes[i].set_title(f"{target_col} (N={len(obs_all)})", fontsize=11)
            if i == 0:
                axes[i].legend(fontsize=6, loc='lower right', ncol=2, frameon=False)

            rmse = float(root_mean_squared_error(obs_all, pred_all))
            r2 = float(r2_score(obs_all, pred_all))
            mae = float(mean_absolute_error(obs_all, pred_all))
            frac2 = float(np.mean(np.abs(pred_all - obs_all) <= log10_2))
            summary_rows.append({
                'Split': split_pattern, 'Assay': target_col, 'N_total': int(len(obs_all)),
                'RMSE': rmse, 'R2': r2, 'MAE': mae, 'frac_within_2fold': frac2,
            })

        fig.suptitle(title, y=1.0, fontsize=12)
        fig.tight_layout()
        plt.savefig(f'{out_dir}{fig_name}', dpi=300, bbox_inches='tight')
        plt.close()
        pd.DataFrame(summary_rows).to_csv(f'{out_dir}{table_name}', index=False)


    # t-SNE / UMAP embedding comparison (holdout)
    test_id = pd.read_csv(f'{ad_dir}/test_id.csv', dtype={'COMPID': str})['COMPID'].values
    holdout = merge_df[merge_df['COMPID'].isin(test_id)].reset_index(drop=True)
    fp_df = pd.read_csv(f'{input_path}/fp_df.csv')
    fp_cols = [c for c in fp_df.columns if c.startswith('fp_')]
    holdout = holdout.merge(fp_df[['COMPID'] + fp_cols], on='COMPID', how='left')

    with open(f'{ad_dir}/study_params.yaml') as f:
        study_params = yaml.safe_load(f)
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    smiles_to_graph = SMILESToBigraph(node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)

    encoders = []
    for cv in range(5):
        enc = GraphEncoderGCN(
            in_feats=atom_featurizer.feat_size('h'),
            hidden_feats=study_params['gcn_hidden_feats'],
            activation=[nn.LeakyReLU(), nn.LeakyReLU()],
            dropout=study_params['gcn_dropout'],
        ).to(device)
        enc.load_state_dict(torch.load(f'{ad_dir}/cv{cv}/encoder.pth', map_location=device, weights_only=True))
        enc.eval()
        encoders.append(enc)

    readout_dim = 2 * study_params['gcn_hidden_feats'][-1]
    readouts = np.zeros((len(holdout), readout_dim))
    with torch.no_grad():
        for k, smi in enumerate(holdout['SMILES'].values):
            g = smiles_to_graph(smi)
            if g is None:
                readouts[k, :] = np.nan; continue
            g = g.to(device)
            fs = []
            for enc in encoders:
                f = enc(g, g.ndata['h'].float()).cpu().numpy().ravel()
                fs.append(f)
            readouts[k, :] = np.mean(np.stack(fs, axis=0), axis=0)
    fp_matrix = holdout[fp_cols].values.astype(np.float32)

    tsne_mt = TSNE(n_components=2, random_state=42, init='pca', perplexity=30, learning_rate='auto').fit_transform(readouts)
    tsne_fp = TSNE(n_components=2, random_state=42, init='pca', perplexity=30, learning_rate='auto').fit_transform(fp_matrix)

    have_umap = False
    umap_mt = None; umap_fp = None
    try:
        import umap
        umap_mt = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(readouts)
        umap_fp = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(fp_matrix)
        have_umap = True
    except ImportError:
        pass

    primary_assay = []
    for _, r in holdout.iterrows():
        measured = [a for a in target_cols if not pd.isna(r[a])]
        if len(measured) == 0:
            primary_assay.append('none')
        elif len(measured) > 1:
            primary_assay.append('multi')
        else:
            primary_assay.append(measured[0])
    primary_assay = np.array(primary_assay)

    colors = {'Caco-2': '#1f77b4', 'LLC-PK1': '#2ca02c', 'MDCK': '#ff7f0e',
              'PAMPA': '#d62728', 'RRCK': '#9467bd', 'multi': '#000000', 'none': '#999999'}

    n_rows = 2 if have_umap else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    for ax, pts, title in [
        (axes[0, 0], tsne_mt, 't-SNE of MT-GCN readouts'),
        (axes[0, 1], tsne_fp, 't-SNE of Morgan FPs'),
    ]:
        for lab, c in colors.items():
            mask = (primary_assay == lab)
            if mask.sum() == 0:
                continue
            ax.scatter(pts[mask, 0], pts[mask, 1], s=10, alpha=0.6,
                       c=c, label=f'{lab} (N={int(mask.sum())})', edgecolor='none')
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc='best')
    if have_umap:
        for ax, pts, title in [
            (axes[1, 0], umap_mt, 'UMAP of MT-GCN readouts'),
            (axes[1, 1], umap_fp, 'UMAP of Morgan FPs'),
        ]:
            for lab, c in colors.items():
                mask = (primary_assay == lab)
                if mask.sum() == 0:
                    continue
                ax.scatter(pts[mask, 0], pts[mask, 1], s=10, alpha=0.6,
                           c=c, label=f'{lab} (N={int(mask.sum())})', edgecolor='none')
            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=7, loc='best')
    fig.suptitle('Holdout test compounds — embeddings coloured by assay membership', y=1.01)
    fig.tight_layout()
    plt.savefig(f'{out_dir}Fig_V1_embedding_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    emb_df = pd.DataFrame({
        'COMPID': holdout['COMPID'].values,
        'primary_assay': primary_assay,
        'tsne_mt_x': tsne_mt[:, 0], 'tsne_mt_y': tsne_mt[:, 1],
        'tsne_fp_x': tsne_fp[:, 0], 'tsne_fp_y': tsne_fp[:, 1],
    })
    if have_umap:
        emb_df['umap_mt_x'] = umap_mt[:, 0]; emb_df['umap_mt_y'] = umap_mt[:, 1]
        emb_df['umap_fp_x'] = umap_fp[:, 0]; emb_df['umap_fp_y'] = umap_fp[:, 1]
    emb_df.to_csv(f'{out_dir}Table_V1_embeddings.csv', index=False)


    # Subsampling quantitative analysis (paired-seed deltas).
    # Prefer the *_with_mae files; fall back to the legacy filenames.
    sub_dir = 'results_from_manuscript/subsampled_seeds/'
    if os.path.isdir(sub_dir):
        loaded = {}
        targets = [
            (['RandomForest_tuned_with_mae', 'RandomForest_tuned',
              'RandomForest_with_mae', 'RandomForest'], 'RandomForest'),
            (['single_task_cv_with_mae', 'single_task_cv'], 'single_task_cv'),
            (['multi_task_cv_with_mae', 'multi_task_cv'], 'multi_task_cv'),
        ]
        for names, unified_name in targets:
            df = None
            for name in names:
                p = f'{sub_dir}score_summary_{name}.csv'
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    break
            if df is None:
                continue
            df = df[pd.to_numeric(df['Seed'], errors='coerce').notna()].copy()
            df['Seed'] = df['Seed'].astype(np.int64)
            df['N'] = pd.to_numeric(df['N'], errors='coerce').astype('Int64')
            df = df.dropna(subset=['N'])
            df['N'] = df['N'].astype(int)
            df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
            df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
            df = df.dropna(subset=['RMSE', 'R2'])
            if 'MAE' in df.columns:
                df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')
            else:
                df['MAE'] = np.nan
            df['Model'] = unified_name
            loaded[unified_name] = df

        if len(loaded) == 3:
            combined = pd.concat(loaded.values(), ignore_index=True)
            combined.to_csv(f'{out_dir}Table_Q1_subsampling_all_seeds.csv', index=False)

            agg = combined.groupby(['N', 'Model']).agg(
                RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
                R2_mean=('R2', 'mean'), R2_std=('R2', 'std'),
                MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'),
                n_seeds=('RMSE', 'count'),
            ).reset_index()
            agg.to_csv(f'{out_dir}Table_Q2_subsampling_mean_std.csv', index=False)

            pivot_rmse = combined.pivot_table(index=['N', 'Seed'], columns='Model', values='RMSE').reset_index()
            pivot_r2 = combined.pivot_table(index=['N', 'Seed'], columns='Model', values='R2').reset_index()
            pivot_mae = combined.pivot_table(index=['N', 'Seed'], columns='Model', values='MAE').reset_index()
            delta_rows = []
            for (N, seed), g in pivot_rmse.groupby(['N', 'Seed']):
                r2_row = pivot_r2[(pivot_r2['N'] == N) & (pivot_r2['Seed'] == seed)]
                mae_row = pivot_mae[(pivot_mae['N'] == N) & (pivot_mae['Seed'] == seed)]
                if r2_row.empty:
                    continue
                try:
                    row = {
                        'N': int(N), 'Seed': int(seed),
                        'dRMSE_MT_minus_ST': float(g['multi_task_cv'].iloc[0] - g['single_task_cv'].iloc[0]),
                        'dRMSE_MT_minus_RF': float(g['multi_task_cv'].iloc[0] - g['RandomForest'].iloc[0]),
                        'dR2_MT_minus_ST': float(r2_row['multi_task_cv'].iloc[0] - r2_row['single_task_cv'].iloc[0]),
                        'dR2_MT_minus_RF': float(r2_row['multi_task_cv'].iloc[0] - r2_row['RandomForest'].iloc[0]),
                    }
                    if not mae_row.empty:
                        try:
                            row['dMAE_MT_minus_ST'] = float(mae_row['multi_task_cv'].iloc[0] - mae_row['single_task_cv'].iloc[0])
                            row['dMAE_MT_minus_RF'] = float(mae_row['multi_task_cv'].iloc[0] - mae_row['RandomForest'].iloc[0])
                        except (KeyError, ValueError):
                            row['dMAE_MT_minus_ST'] = np.nan
                            row['dMAE_MT_minus_RF'] = np.nan
                    else:
                        row['dMAE_MT_minus_ST'] = np.nan
                        row['dMAE_MT_minus_RF'] = np.nan
                except KeyError:
                    continue
                delta_rows.append(row)
            delta_df = pd.DataFrame(delta_rows)
            if not delta_df.empty:
                delta_df.to_csv(f'{out_dir}Table_Q3_subsampling_deltas_per_seed.csv', index=False)
                delta_agg = delta_df.groupby('N').agg(
                    dRMSE_MT_minus_ST_mean=('dRMSE_MT_minus_ST', 'mean'),
                    dRMSE_MT_minus_ST_std=('dRMSE_MT_minus_ST', 'std'),
                    dRMSE_MT_minus_RF_mean=('dRMSE_MT_minus_RF', 'mean'),
                    dRMSE_MT_minus_RF_std=('dRMSE_MT_minus_RF', 'std'),
                    dR2_MT_minus_ST_mean=('dR2_MT_minus_ST', 'mean'),
                    dR2_MT_minus_ST_std=('dR2_MT_minus_ST', 'std'),
                    dR2_MT_minus_RF_mean=('dR2_MT_minus_RF', 'mean'),
                    dR2_MT_minus_RF_std=('dR2_MT_minus_RF', 'std'),
                    dMAE_MT_minus_ST_mean=('dMAE_MT_minus_ST', 'mean'),
                    dMAE_MT_minus_ST_std=('dMAE_MT_minus_ST', 'std'),
                    dMAE_MT_minus_RF_mean=('dMAE_MT_minus_RF', 'mean'),
                    dMAE_MT_minus_RF_std=('dMAE_MT_minus_RF', 'std'),
                    n_seeds=('Seed', 'count'),
                ).reset_index()
                delta_agg.to_csv(f'{out_dir}Table_Q4_subsampling_delta_mean_std.csv', index=False)

                # Optional: full Caco-2 baseline as a single point
                full_path = f'{sub_dir}score_summary_full_caco2_baselines.csv'
                full_deltas = None
                if os.path.exists(full_path):
                    full_df = pd.read_csv(full_path).set_index('Model')
                    try:
                        full_n = int(full_df.loc['MT-GCN_full', 'N_trainval'])
                        full_deltas = {
                            'N': full_n,
                            'dRMSE_MT_minus_ST': float(full_df.loc['MT-GCN_full', 'RMSE'] - full_df.loc['ST-GCN_full', 'RMSE']),
                            'dRMSE_MT_minus_RF': float(full_df.loc['MT-GCN_full', 'RMSE'] - full_df.loc['RF_tuned_full', 'RMSE']),
                            'dR2_MT_minus_ST': float(full_df.loc['MT-GCN_full', 'R2'] - full_df.loc['ST-GCN_full', 'R2']),
                            'dR2_MT_minus_RF': float(full_df.loc['MT-GCN_full', 'R2'] - full_df.loc['RF_tuned_full', 'R2']),
                            'dMAE_MT_minus_ST': float(full_df.loc['MT-GCN_full', 'MAE'] - full_df.loc['ST-GCN_full', 'MAE']),
                            'dMAE_MT_minus_RF': float(full_df.loc['MT-GCN_full', 'MAE'] - full_df.loc['RF_tuned_full', 'MAE']),
                        }
                    except KeyError:
                        full_deltas = None

                has_mae = delta_agg['dMAE_MT_minus_ST_mean'].notna().any()
                n_panels = 3 if has_mae else 2
                fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
                if n_panels == 1:
                    axes = [axes]
                panels = [
                    ('dRMSE', 'ΔRMSE (negative = MT wins)'),
                    ('dR2', 'ΔR² (positive = MT wins)'),
                ]
                if has_mae:
                    panels.append(('dMAE', 'ΔMAE (negative = MT wins)'))
                for ax, (metric_prefix, ylab) in zip(axes, panels):
                    for comp, color in (('MT_minus_ST', '#1f77b4'), ('MT_minus_RF', '#d62728')):
                        mean_col = f'{metric_prefix}_{comp}_mean'
                        std_col = f'{metric_prefix}_{comp}_std'
                        ax.errorbar(
                            delta_agg['N'], delta_agg[mean_col], yerr=delta_agg[std_col],
                            marker='o', capsize=3, color=color, label=comp.replace('_', ' '),
                        )
                        if full_deltas is not None:
                            key = f'{metric_prefix}_{comp}'
                            ax.plot(
                                [full_deltas['N']], [full_deltas[key]],
                                marker='*', markersize=14, color=color, linestyle='none',
                                markeredgecolor='black', markeredgewidth=0.5,
                                label=f'{comp.replace("_", " ")} (full Caco-2)',
                            )
                    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
                    ax.set_xscale('log')
                    ax.set_xlabel('Caco-2 subsample size N')
                    ax.set_ylabel(ylab)
                    ax.legend(); ax.grid(True, alpha=0.3)
                fig.suptitle('Caco-2 subsampling: multitask advantage vs training size', y=1.02)
                fig.tight_layout()
                plt.savefig(f'{out_dir}Fig_Q1_subsampling_deltas.png', dpi=300, bbox_inches='tight')
                plt.close()
