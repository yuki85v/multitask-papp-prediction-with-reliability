import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from utils_for_admet_model.execute_model import execute_mt_cv_train, execute_st_cv_train


if __name__ == "__main__":

    root = np.random.SeedSequence(42)
    seeds = [int(s.generate_state(1)[0]) for s in root.spawn(5)]

    # Settings
    input_path = 'training_data/'
    split_pattern = 'random_split'
    model_path = 'trained_model/'

    # Random Forest
    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)
    fp_df = pd.read_csv(f'{input_path}/fp_df.csv', dtype={'COMPID': str})

    trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/trainval_id.csv', dtype={'COMPID': str})['COMPID'].values
    test_id = pd.read_csv(f'{model_path}/{split_pattern}/test_id.csv', dtype={'COMPID': str})['COMPID'].values

    target_col = 'Caco-2'
    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    
    root = np.random.SeedSequence(42)
    seeds = [int(s.generate_state(1)[0]) for s in root.spawn(5)]
    for seed in seeds:
        caco2_id = merge_df[~merge_df['Caco-2'].isna()]['COMPID'].values
        trainval_caco2_id = np.intersect1d(trainval_id, caco2_id)
        rng = np.random.default_rng(seed)
        perm_ids = rng.permutation(trainval_caco2_id)

        for N in [100, 200, 500, 1000, 2500]:
            allowed = set(perm_ids[:N])
            trainval_merge_df = merge_df[merge_df['COMPID'].isin(trainval_id)].copy()
            test_merge_df = merge_df[merge_df['COMPID'].isin(test_id)].copy()
            mask_disallow = ~trainval_merge_df["COMPID"].isin(allowed)
            trainval_merge_df.loc[mask_disallow, "Caco-2"] = np.nan
            trainval_merge_df = trainval_merge_df.dropna(subset=target_cols, how="all")
            sampled_merge_df = pd.concat([trainval_merge_df, test_merge_df], ignore_index=True)

            model_output_path = f'{model_path}/{split_pattern}/RandomForest_subsampled_seeds/N_sample_{N}/seed_{seed}/'
            os.makedirs(model_output_path, exist_ok=True)

            fp_merged = pd.merge(
                sampled_merge_df[['COMPID', target_col]].dropna(subset=[target_col]), 
                fp_df,
                on='COMPID', 
                how='left'
            )
            fp_merged.to_csv(f'{model_output_path}fp_merged_{target_col}.csv', index=False)

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
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            pred_df = pd.read_csv(f'{model_output_path}pred_df_rf_model_{target_col}.csv')

            tmp_obs = pred_df['LABEL'].values
            tmp_pred = pred_df['PRED'].values
            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)

            ax.scatter(tmp_obs, tmp_pred, alpha=0.8, color='black')
            ax.plot(xylim, xylim, 'k--')
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted')
            ax.set_title(f"{target_col} (R² = {r2:.3f}, RMSE = {rmse:.3f})")
            ax.set_xlim(xylim)
            ax.set_ylim(xylim)
            ax.set_aspect('equal', adjustable='box')

            fig.tight_layout()
            plt.savefig(f'{model_output_path}pred_vs_obs.png')
            plt.close()

    # Summarize scores
    task = 'RandomForest'
    col = 'Caco-2'
    score_dfs = []
    for N in [100, 200, 500, 1000, 2500]:
        for seed in seeds:
            model_output_path = f'{model_path}/{split_pattern}/{task}_subsampled_seeds/N_sample_{N}/seed_{seed}/'
            pred_df = pd.read_csv(
                f'{model_output_path}/pred_df_rf_model_{col}.csv', dtype={'COMPID': str}
            )
            tmp_obs = pred_df['LABEL'].values
            tmp_pred = pred_df['PRED'].values
            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)
            tmp_df = pd.DataFrame({'N': [N], 'Seed': [seed], 'RMSE': [rmse], 'R2': [r2]})
            score_dfs.append(tmp_df)
            
    score_df = pd.concat(score_dfs, ignore_index=True)
    
    mean_df = (
        score_df.groupby("N", as_index=False)[["RMSE", "R2"]]
        .mean(numeric_only=True)
    )
    mean_df["Seed"] = "mean"
    std_df = (
        score_df.groupby("N", as_index=False)[["RMSE", "R2"]]
        .std(ddof=1, numeric_only=True)
    )
    std_df["Seed"] = "std"

    score_df = pd.concat([score_df, mean_df, std_df], ignore_index=True)

    # Reorder the "Seed" column to have the original seeds followed by "mean" and "std"
    seed_order = list(map(str, seeds)) + ["mean", "std"]
    score_df["Seed"] = score_df["Seed"].astype(str)
    score_df["Seed"] = pd.Categorical(score_df["Seed"], categories=seed_order, ordered=True)

    score_df = score_df.sort_values(["N", "Seed"]).reset_index(drop=True)
    score_df.to_csv(f'{model_path}/{split_pattern}/{task}_subsampled_seeds/Table02_score_summary_{task}.csv', index=False)


    # Single-Task GCN
    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)

    trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/trainval_id.csv', dtype={'COMPID': str})['COMPID'].values
    test_id = pd.read_csv(f'{model_path}/{split_pattern}/test_id.csv', dtype={'COMPID': str})['COMPID'].values

    target_col = 'Caco-2'
    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']

    root = np.random.SeedSequence(42)
    seeds = [int(s.generate_state(1)[0]) for s in root.spawn(5)]
    for seed in seeds:

        caco2_id = merge_df[~merge_df['Caco-2'].isna()]['COMPID'].values
        trainval_caco2_id = np.intersect1d(trainval_id, caco2_id)
        rng = np.random.default_rng(seed)
        perm_ids = rng.permutation(trainval_caco2_id)

        for N in [100, 200, 500, 1000, 2500]:
            allowed = set(perm_ids[:N])
            trainval_merge_df = merge_df[merge_df['COMPID'].isin(trainval_id)].copy()
            test_merge_df = merge_df[merge_df['COMPID'].isin(test_id)].copy()
            mask_disallow = ~trainval_merge_df["COMPID"].isin(allowed)
            trainval_merge_df.loc[mask_disallow, "Caco-2"] = np.nan
            trainval_merge_df = trainval_merge_df.dropna(subset=target_cols, how="all")
            sampled_merge_df = pd.concat([trainval_merge_df, test_merge_df], ignore_index=True)

            model_output_path = f'{model_path}/{split_pattern}/single_task_cv_subsampled_seeds/N_sample_{N}/seed_{seed}/'

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
                model_output_path, sampled_merge_df, target_col, trainval_id, test_id, study_params
            )
            sampled_merge_df.to_csv(f'{model_output_path}sampled_merge_df.csv', index=False)

            label_df = merge_df.set_index('COMPID').loc[test_id].reset_index()
            xylim = np.array([-5, 5])
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            pred_df = pd.read_csv(f'{model_output_path}/test_pred_group_df.csv', dtype={'COMPID': str})    
            tmp_merge = pred_df.merge(label_df[['COMPID', target_col]], on='COMPID', how='left')
            tmp_obs = tmp_merge[target_col].values
            tmp_pred = tmp_merge[target_col+'_pred_mean'].values
            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)

            ax.scatter(tmp_obs, tmp_pred, alpha=0.8, color='black')
            ax.plot(xylim, xylim, 'k--')
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted')
            ax.set_title(f"{target_col} (R² = {r2:.3f}, RMSE = {rmse:.3f})")
            ax.set_xlim(xylim)
            ax.set_ylim(xylim)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            plt.savefig(f'{model_output_path}pred_vs_obs.png', dpi=300)
            plt.close()

    # Summarize scores
    task = 'single_task_cv'
    col = 'Caco-2'
    score_dfs = []
    for N in [100, 200, 500, 1000, 2500]:
        for seed in seeds:
            model_output_path = f'{model_path}/{split_pattern}/{task}_subsampled_seeds/N_sample_{N}/seed_{seed}/'
            test_id = pd.read_csv(
                f'{model_output_path}/test_id.csv', dtype={'COMPID': str}
            )['COMPID'].values
            label_df = merge_df.set_index('COMPID').loc[test_id].reset_index()
            pred_df = pd.read_csv(
                f'{model_output_path}/test_pred_group_df.csv', dtype={'COMPID': str}
            )
            
            tmp_merge = pred_df.merge(label_df[['COMPID', col]], on='COMPID', how='left')
            tmp_obs = tmp_merge[col].values
            tmp_pred = tmp_merge[col+'_pred_mean'].values

            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)
            tmp_df = pd.DataFrame({'N': [N], 'Seed': [seed], 'RMSE': [rmse], 'R2': [r2]})
            score_dfs.append(tmp_df)
            
    score_df = pd.concat(score_dfs, ignore_index=True)
    mean_df = (
        score_df.groupby("N", as_index=False)[["RMSE", "R2"]]
        .mean(numeric_only=True)
    )
    mean_df["Seed"] = "mean"
    std_df = (
        score_df.groupby("N", as_index=False)[["RMSE", "R2"]]
        .std(ddof=1, numeric_only=True)
    )
    std_df["Seed"] = "std"

    score_df = pd.concat([score_df, mean_df, std_df], ignore_index=True)

    seed_order = list(map(str, seeds)) + ["mean", "std"]
    score_df["Seed"] = score_df["Seed"].astype(str)
    score_df["Seed"] = pd.Categorical(score_df["Seed"], categories=seed_order, ordered=True)

    score_df = score_df.sort_values(["N", "Seed"]).reset_index(drop=True)
    score_df.to_csv(f'{model_path}/{split_pattern}/{task}_subsampled_seeds/Table02_score_summary_{task}.csv', index=False)


    # Multi-Task GCN
    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)

    trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/trainval_id.csv', dtype={'COMPID': str})['COMPID'].values
    test_id = pd.read_csv(f'{model_path}/{split_pattern}/test_id.csv', dtype={'COMPID': str})['COMPID'].values

    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']

    root = np.random.SeedSequence(42)
    seeds = [int(s.generate_state(1)[0]) for s in root.spawn(5)]
    for seed in seeds:
        caco2_id = merge_df[~merge_df['Caco-2'].isna()]['COMPID'].values
        trainval_caco2_id = np.intersect1d(trainval_id, caco2_id)
        rng = np.random.default_rng(seed)
        perm_ids = rng.permutation(trainval_caco2_id)

        for N in [100, 200, 500, 1000, 2500]:
            allowed = set(perm_ids[:N])
            trainval_merge_df = merge_df[merge_df['COMPID'].isin(trainval_id)].copy()
            test_merge_df = merge_df[merge_df['COMPID'].isin(test_id)].copy()
            mask_disallow = ~trainval_merge_df["COMPID"].isin(allowed)
            trainval_merge_df.loc[mask_disallow, "Caco-2"] = np.nan
            trainval_merge_df = trainval_merge_df.dropna(subset=target_cols, how="all")
            sampled_merge_df = pd.concat([trainval_merge_df, test_merge_df], ignore_index=True)

            model_output_path = f'{model_path}/{split_pattern}/multi_task_cv_subsampled_seeds/N_sample_{N}/seed_{seed}/'

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
                model_output_path, sampled_merge_df, target_cols, trainval_id, test_id, study_params
            )
            sampled_merge_df.to_csv(f'{model_output_path}sampled_merge_df.csv', index=False)

            label_df = merge_df.set_index('COMPID').loc[test_id].reset_index()
            pred_group_df = pd.read_csv(f'{model_output_path}test_pred_group_df.csv')
            xylim = np.array([-5, 5])
            fig, axes = plt.subplots(1, len(target_cols), figsize=(25, 5))
            for i, col in enumerate(target_cols):    
                tmp_mask = ~np.isnan(label_df[col].values)
                tmp_obs = label_df[col].values[tmp_mask]
                tmp_pred = pred_group_df[col+'_pred_mean'].values[tmp_mask]
                r2 = r2_score(tmp_obs, tmp_pred)
                rmse = root_mean_squared_error(tmp_obs, tmp_pred)

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

    # Summarize scores
    task = 'multi_task_cv'
    col = 'Caco-2'
    score_dfs = []
    for N in [100, 200, 500, 1000, 2500]:
        for seed in seeds:
            model_output_path = f'{model_path}/{split_pattern}/{task}_subsampled_seeds/N_sample_{N}/seed_{seed}/'
            test_id = pd.read_csv(
                f'{model_output_path}/test_id.csv', dtype={'COMPID': str}
            )['COMPID'].values
            label_df = merge_df.set_index('COMPID').loc[test_id].reset_index()
            pred_df = pd.read_csv(
                f'{model_output_path}/test_pred_group_df.csv', dtype={'COMPID': str}
            )
            
            tmp_mask = ~np.isnan(label_df[col].values)
            tmp_obs = label_df[col].values[tmp_mask]
            tmp_pred = pred_df[col+'_pred_mean'].values[tmp_mask]

            r2 = r2_score(tmp_obs, tmp_pred)
            rmse = root_mean_squared_error(tmp_obs, tmp_pred)
            tmp_df = pd.DataFrame({'N': [N], 'Seed': [seed], 'RMSE': [rmse], 'R2': [r2]})
            score_dfs.append(tmp_df)
            
    score_df = pd.concat(score_dfs, ignore_index=True)
    mean_df = (
        score_df.groupby("N", as_index=False)[["RMSE", "R2"]]
        .mean(numeric_only=True)
    )
    mean_df["Seed"] = "mean"
    std_df = (
        score_df.groupby("N", as_index=False)[["RMSE", "R2"]]
        .std(ddof=1, numeric_only=True)
    )
    std_df["Seed"] = "std"

    score_df = pd.concat([score_df, mean_df, std_df], ignore_index=True)

    seed_order = list(map(str, seeds)) + ["mean", "std"]
    score_df["Seed"] = score_df["Seed"].astype(str)
    score_df["Seed"] = pd.Categorical(score_df["Seed"], categories=seed_order, ordered=True)

    score_df = score_df.sort_values(["N", "Seed"]).reset_index(drop=True)
    score_df.to_csv(f'{model_path}/{split_pattern}/{task}_subsampled_seeds/Table02_score_summary_{task}.csv', index=False)
