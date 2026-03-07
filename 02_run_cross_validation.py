import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from utils_for_admet_model.execute_model import execute_mt_cv_train, execute_st_cv_train


if __name__ == "__main__":

    # Settings
    input_path = 'training_data/'
    split_pattern = 'random_split_cv'
    model_path = 'trained_model/'
    
    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)
    fp_df = pd.read_csv(f'{input_path}/fp_df.csv')

    # Random Forest
    score_dfs = []
    for fold in range(10): 
        trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/trainval_id.csv')['COMPID'].values
        test_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/test_id.csv')['COMPID'].values

        model_output_path = f'{model_path}/{split_pattern}/fold_{fold+1}/RandomForest/'
        os.makedirs(model_output_path, exist_ok=True)

        target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
        for target_col in target_cols:

            fp_merged = pd.merge(
                merge_df[['COMPID', target_col]].dropna(subset=[target_col]), 
                fp_df,
                on='COMPID', 
                how='left'
            )

            X_train = fp_merged[fp_merged['COMPID'].isin(trainval_id)].drop(columns=['COMPID', target_col])
            y_train = fp_merged[fp_merged['COMPID'].isin(trainval_id)][target_col].values
            X_test = fp_merged[fp_merged['COMPID'].isin(test_id)].drop(columns=['COMPID', target_col])
            y_test = fp_merged[fp_merged['COMPID'].isin(test_id)][target_col].values

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            pickle.dump(model, open(f'{model_output_path}rf_model_{target_col}.pkl', 'wb'))
            y_pred = model.predict(X_test)
            pd.DataFrame({
                'COMPID': fp_merged[fp_merged['COMPID'].isin(test_id)]['COMPID'].values,
                'LABEL': y_test,
                'PRED': y_pred
            }).to_csv(f'{model_output_path}pred_df_rf_model_{target_col}.csv', index=False)

        xylim = np.array([-5, 5])
        fig, axes = plt.subplots(1, len(target_cols), figsize=(25, 5))
        for i, target_col in enumerate(target_cols):
            pred_df = pd.read_csv(f'{model_output_path}pred_df_rf_model_{target_col}.csv')
            tmp_obs = pred_df['LABEL'].values
            tmp_pred = pred_df['PRED'].values
            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)
            tmp_df = pd.DataFrame({'Target': [target_col], 'Fold': [fold+1], 'RMSE': [rmse], 'R2': [r2]})
            score_dfs.append(tmp_df)

            axes[i].scatter(tmp_obs, tmp_pred, alpha=0.8, color='black')
            axes[i].plot(xylim, xylim, 'k--')
            axes[i].set_xlabel('Observed')
            axes[i].set_ylabel('Predicted')
            axes[i].set_title(f"{target_col} (RMSE = {rmse:.3f}, R² = {r2:.3f})")
            axes[i].set_xlim(xylim)
            axes[i].set_ylim(xylim)
            axes[i].set_aspect('equal', adjustable='box')

        fig.tight_layout()
        plt.savefig(f'{model_output_path}pred_vs_obs.png')
        plt.close()

    score_df = pd.concat(score_dfs, ignore_index=True)
    mean_df = (
        score_df.groupby("Target", as_index=False)[["RMSE", "R2"]]
        .mean(numeric_only=True)
    )
    mean_df["Fold"] = "mean"
    std_df = (
        score_df.groupby("Target", as_index=False)[["RMSE", "R2"]]
        .std(ddof=1, numeric_only=True)
    )
    std_df["Fold"] = "std"

    score_df = pd.concat([score_df, mean_df, std_df], ignore_index=True)

    # Reorder the "Fold" column to ensure the order is 1, 2, ..., 10, mean, std for each Target
    fold_order = list(map(str, range(1, 11))) + ["mean", "std"]
    score_df["Fold"] = score_df["Fold"].astype(str)
    score_df["Fold"] = pd.Categorical(score_df["Fold"], categories=fold_order, ordered=True)

    score_df = score_df.sort_values(["Target", "Fold"]).reset_index(drop=True)
    score_df.to_csv(f'{model_path}/{split_pattern}/Table01_CV_RF_metrics.csv', index=False)


    # Single-task GCN
    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    dfs = []
    for fold in range(10):
        trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/trainval_id.csv')['COMPID'].values
        test_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/test_id.csv')['COMPID'].values

        model_output_path = f'{model_path}/{split_pattern}/fold_{fold+1}/single_task_cv/'
        os.makedirs(model_output_path, exist_ok=True)

        for target_col in target_cols:
            target_model_output_path = f'{model_output_path}{target_col}/'

            study_params = {
                'target_col': target_col,
                'batch_size': 64,
                'gcn_hidden_feats': [256, 256],
                'gcn_dropout': [0.1, 0.1],
                'dnn_hidden_dims': [256, 64],
                'dnn_dropout': [0.2, 0.2],
                'dnn_output_dim': 1,
                'gcn_lr': 1e-4,
                'dnn_lr': 1e-4,
                'scheduler': False,
                'warmup_epochs': None,
                'effective_epochs': None,
                'eta_min': None,
                'earlystopping_patience': 100,
                'multi_task_loss_weighted': None,
            }

            train_results, test_results = execute_st_cv_train(
                target_model_output_path, merge_df, target_col, trainval_id, test_id, study_params
            )

        label_df = merge_df.set_index('COMPID').loc[test_id].reset_index()

        target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
        xylim = np.array([-5, 5])
        fig, axes = plt.subplots(1, len(target_cols), figsize=(25, 5))
        for i, col in enumerate(target_cols):
            pred_df = pd.read_csv(f'{model_output_path}{col}/test_pred_group_df.csv')    
            tmp_merge = pred_df.merge(label_df[['COMPID', col]], on='COMPID', how='left')
            tmp_obs = tmp_merge[col].values
            tmp_pred = tmp_merge[col+'_pred_mean'].values
            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)
            tmp_df = pd.DataFrame({'Target': [col], 'Fold': [fold+1], 'RMSE': [rmse], 'R2': [r2]})
            dfs.append(tmp_df)

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
    mean_df = (
        score_df.groupby("Target", as_index=False)[["RMSE", "R2"]]
        .mean(numeric_only=True)
    )
    mean_df["Fold"] = "mean"
    std_df = (
        score_df.groupby("Target", as_index=False)[["RMSE", "R2"]]
        .std(ddof=1, numeric_only=True)
    )
    std_df["Fold"] = "std"
    score_df = pd.concat([score_df, mean_df, std_df], ignore_index=True)
    fold_order = list(map(str, range(1, 11))) + ["mean", "std"]
    score_df["Fold"] = score_df["Fold"].astype(str)
    score_df["Fold"] = pd.Categorical(score_df["Fold"], categories=fold_order, ordered=True)
    score_df = score_df.sort_values(["Target", "Fold"]).reset_index(drop=True)
    score_df.to_csv(f'{model_path}/{split_pattern}/Table02_CV_ST-GCN_metrics.csv', index=False)


    # Multi-task GCN
    dfs = []
    for fold in range(10):
        trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/trainval_id.csv')['COMPID'].values
        test_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/test_id.csv')['COMPID'].values

        model_output_path = f'{model_path}/{split_pattern}/fold_{fold+1}/multi_task_cv/'

        target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
        study_params = {
            'target_col': target_cols,
            'batch_size': 64,
            'gcn_hidden_feats': [256, 256],
            'gcn_dropout': [0.1, 0.1],
            'dnn_hidden_dims': [256, 64],
            'dnn_dropout': [0.2, 0.2],
            'dnn_output_dim': len(target_cols),
            'gcn_lr': 5e-5,
            'dnn_lr': 5e-5,
            'scheduler': False,
            'warmup_epochs': None,
            'effective_epochs': None,
            'eta_min': None,
            'earlystopping_patience': 100,
            'multi_task_loss_weighted': False,
        }

        train_results, test_results = execute_mt_cv_train(
            model_output_path, merge_df, target_cols, trainval_id, test_id, study_params
        )
        pred_group_df = pd.read_csv(f'{model_output_path}test_pred_group_df.csv')
        label_df = merge_df.set_index('COMPID').loc[test_id].reset_index()

        xylim = np.array([-5, 5])
        fig, axes = plt.subplots(1, len(target_cols), figsize=(25, 5))
        for i, col in enumerate(target_cols):    
            tmp_mask = ~np.isnan(label_df[col].values)
            tmp_obs = label_df[col].values[tmp_mask]
            tmp_pred = pred_group_df[col+'_pred_mean'].values[tmp_mask]
            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)
            tmp_df = pd.DataFrame({'Target': [col], 'Fold': [fold+1], 'RMSE': [rmse], 'R2': [r2]})
            dfs.append(tmp_df)

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
    mean_df = (
        score_df.groupby("Target", as_index=False)[["RMSE", "R2"]]
        .mean(numeric_only=True)
    )
    mean_df["Fold"] = "mean"
    std_df = (
        score_df.groupby("Target", as_index=False)[["RMSE", "R2"]]
        .std(ddof=1, numeric_only=True)
    )
    std_df["Fold"] = "std"
    score_df = pd.concat([score_df, mean_df, std_df], ignore_index=True)
    fold_order = list(map(str, range(1, 11))) + ["mean", "std"]
    score_df["Fold"] = score_df["Fold"].astype(str)
    score_df["Fold"] = pd.Categorical(score_df["Fold"], categories=fold_order, ordered=True)
    score_df = score_df.sort_values(["Target", "Fold"]).reset_index(drop=True)
    score_df.to_csv(f'{model_path}/{split_pattern}/Table03_CV_MT-GCN_metrics.csv', index=False)