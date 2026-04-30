import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from utils_for_admet_model.execute_model import execute_mt_cv_train


# Leave-one-assay-out (LOAO) ablation of the MT-GCN.
# For each of the five assays A in turn we re-train MT-GCN on the remaining
# four assays (dnn_output_dim=4), using the same 90/10 random_split trainval
# and test set as the deployed multi_task_cv_ad model. The comparison vs. the
# full five-assay MT-GCN quantifies how much each dropped assay contributes
# to the representation the other four assays share.
if __name__ == "__main__":

    # Settings
    input_path = 'training_data/'
    split_pattern = 'random_split'
    model_path = 'trained_model/'

    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)

    trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/trainval_id.csv', dtype={'COMPID': str})['COMPID'].values
    test_id = pd.read_csv(f'{model_path}/{split_pattern}/test_id.csv', dtype={'COMPID': str})['COMPID'].values

    all_target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']

    dfs = []
    for assay_to_drop in all_target_cols:
        remaining = [a for a in all_target_cols if a != assay_to_drop]
        model_output_path = f'{model_path}/{split_pattern}/multi_task_cv_loao/leave_out_{assay_to_drop}/'

        if os.path.exists(f'{model_output_path}test_pred_group_df.csv'):
            pred_group_df = pd.read_csv(f'{model_output_path}test_pred_group_df.csv')
        else:
            study_params = {
                'target_col': remaining,
                'batch_size': 64,
                'gcn_hidden_feats': [256, 256],
                'gcn_dropout': [0.1, 0.1],
                'dnn_hidden_dims': [256, 64],
                'dnn_dropout': [0.2, 0.2],
                'dnn_output_dim': len(remaining),
                'gcn_lr': 5e-5,
                'dnn_lr': 5e-5,
                'scheduler': False,
                'warmup_epochs': None,
                'effective_epochs': None,
                'eta_min': None,
                'earlystopping_patience': 100,
                'multi_task_loss_weighted': False,
            }
            execute_mt_cv_train(
                model_output_path, merge_df, remaining, trainval_id, test_id, study_params
            )
            pred_group_df = pd.read_csv(f'{model_output_path}test_pred_group_df.csv')

        label_df = merge_df.set_index('COMPID').loc[test_id].reset_index()
        xylim = np.array([-5, 5])
        fig, axes = plt.subplots(1, len(remaining), figsize=(25, 5))
        for i, col in enumerate(remaining):
            tmp_mask = ~np.isnan(label_df[col].values)
            tmp_obs = label_df[col].values[tmp_mask]
            tmp_pred = pred_group_df[col+'_pred_mean'].values[tmp_mask]
            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)
            mae = mean_absolute_error(tmp_obs, tmp_pred)
            dfs.append(pd.DataFrame({
                'Dropped_assay': [assay_to_drop],
                'Target': [col],
                'RMSE': [rmse],
                'R2': [r2],
                'MAE': [mae],
            }))

            axes[i].scatter(tmp_obs, tmp_pred, alpha=0.8, color='black')
            axes[i].plot(xylim, xylim, 'k--')
            axes[i].set_xlabel('Observed')
            axes[i].set_ylabel('Predicted')
            axes[i].set_title(f"{col} (R² = {r2:.3f}, RMSE = {rmse:.3f})")
            axes[i].set_xlim(xylim)
            axes[i].set_ylim(xylim)
            axes[i].set_aspect('equal', adjustable='box')

        fig.tight_layout()
        plt.savefig(f'{model_output_path}pred_vs_obs.png', dpi=300)
        plt.close()

    score_df = pd.concat(dfs, ignore_index=True)
    score_df.to_csv(f'{model_path}/{split_pattern}/multi_task_cv_loao/Table_LOAO_per_assay_metrics.csv', index=False)


    # Compare against the full five-assay MT-GCN baseline on the same test set
    full_pred_df = pd.read_csv(f'{model_path}/{split_pattern}/multi_task_cv_ad/test_pred_group_df.csv')
    label_df = merge_df.set_index('COMPID').loc[test_id].reset_index()
    baseline_rows = []
    for col in all_target_cols:
        tmp_mask = ~np.isnan(label_df[col].values)
        tmp_obs = label_df[col].values[tmp_mask]
        tmp_pred = full_pred_df[col+'_pred_mean'].values[tmp_mask]
        baseline_rows.append({
            'Target': col,
            'RMSE_full': float(root_mean_squared_error(tmp_obs, tmp_pred)),
            'R2_full': float(r2_score(tmp_obs, tmp_pred)),
            'MAE_full': float(mean_absolute_error(tmp_obs, tmp_pred)),
        })
    baseline_df = pd.DataFrame(baseline_rows)

    delta_rows = []
    for assay_to_drop in all_target_cols:
        remaining = [a for a in all_target_cols if a != assay_to_drop]
        sub = score_df[score_df['Dropped_assay'] == assay_to_drop]
        for col in remaining:
            r = sub[sub['Target'] == col].iloc[0]
            b = baseline_df[baseline_df['Target'] == col].iloc[0]
            delta_rows.append({
                'Dropped_assay': assay_to_drop,
                'Target': col,
                'RMSE_loao': r['RMSE'],
                'RMSE_full': b['RMSE_full'],
                'dRMSE_loao_minus_full': float(r['RMSE'] - b['RMSE_full']),
                'R2_loao': r['R2'],
                'R2_full': b['R2_full'],
                'dR2_loao_minus_full': float(r['R2'] - b['R2_full']),
                'MAE_loao': r['MAE'],
                'MAE_full': b['MAE_full'],
                'dMAE_loao_minus_full': float(r['MAE'] - b['MAE_full']),
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(f'{model_path}/{split_pattern}/multi_task_cv_loao/Table_LOAO_vs_full_deltas.csv', index=False)


    # Assay importance summary: how much the other four assays lose on
    # average when each assay is dropped (larger positive dRMSE mean =
    # more important assay to drop)
    importance_rows = []
    for assay_to_drop in all_target_cols:
        sub = delta_df[delta_df['Dropped_assay'] == assay_to_drop]
        importance_rows.append({
            'Dropped_assay': assay_to_drop,
            'mean_dRMSE_on_others': float(sub['dRMSE_loao_minus_full'].mean()),
            'max_dRMSE_on_others': float(sub['dRMSE_loao_minus_full'].max()),
            'mean_dR2_on_others': float(sub['dR2_loao_minus_full'].mean()),
            'min_dR2_on_others': float(sub['dR2_loao_minus_full'].min()),
            'n_remaining_assays': int(len(sub)),
        })
    importance_df = pd.DataFrame(importance_rows)
    importance_df = importance_df.sort_values('mean_dRMSE_on_others', ascending=False).reset_index(drop=True)
    importance_df.to_csv(f'{model_path}/{split_pattern}/multi_task_cv_loao/Table_LOAO_assay_importance.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(importance_df['Dropped_assay'].values, importance_df['mean_dRMSE_on_others'].values,
                color='#4c72b0', edgecolor='black')
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Dropped assay')
    axes[0].set_ylabel('Mean ΔRMSE on other 4 assays (LOAO − full)')
    axes[0].set_title('Assay importance: ΔRMSE on remaining assays')
    for k, v in enumerate(importance_df['mean_dRMSE_on_others'].values):
        axes[0].text(k, v, f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

    axes[1].bar(importance_df['Dropped_assay'].values, importance_df['mean_dR2_on_others'].values,
                color='#c44e52', edgecolor='black')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Dropped assay')
    axes[1].set_ylabel('Mean ΔR² on other 4 assays (LOAO − full)')
    axes[1].set_title('Assay importance: ΔR² on remaining assays')
    for k, v in enumerate(importance_df['mean_dR2_on_others'].values):
        axes[1].text(k, v, f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

    fig.tight_layout()
    plt.savefig(f'{model_path}/{split_pattern}/multi_task_cv_loao/Fig_LOAO_assay_importance.png', dpi=300)
    plt.close()
