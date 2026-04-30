import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Settings
    input_path = 'training_data/'
    model_path = 'trained_model/'

    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)
    fp_df = pd.read_csv(f'{input_path}/fp_df.csv')

    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [None, 20],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 0.3],
    }
    gs_n_jobs = int(os.environ.get('GS_N_JOBS', '4'))

    for split_pattern in ['random_split_cv', 'scaffold_split_cv']:

        # Tuned Random Forest with nested GridSearchCV
        score_dfs = []
        best_params_dfs = []
        for fold in range(10):
            trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/trainval_id.csv')['COMPID'].values
            test_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/test_id.csv')['COMPID'].values

            model_output_path = f'{model_path}/{split_pattern}/fold_{fold+1}/RandomForest_tuned/'
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

                gs = GridSearchCV(
                    estimator=RandomForestRegressor(random_state=42, n_jobs=1),
                    param_grid=param_grid,
                    cv=3,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=gs_n_jobs,
                    refit=True,
                )
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                pickle.dump(model, open(f'{model_output_path}rf_model_{target_col}.pkl', 'wb'))
                y_pred = model.predict(X_test)
                pd.DataFrame({
                    'COMPID': fp_merged[fp_merged['COMPID'].isin(test_id)]['COMPID'].values,
                    'LABEL': y_test,
                    'PRED': y_pred
                }).to_csv(f'{model_output_path}pred_df_rf_model_{target_col}.csv', index=False)

                best_params_dfs.append(pd.DataFrame({
                    'Target': [target_col],
                    'Fold': [fold+1],
                    'n_estimators': [gs.best_params_['n_estimators']],
                    'max_depth': [gs.best_params_['max_depth']],
                    'min_samples_leaf': [gs.best_params_['min_samples_leaf']],
                    'max_features': [gs.best_params_['max_features']],
                    'best_cv_rmse': [float(-gs.best_score_)],
                }))

            xylim = np.array([-5, 5])
            fig, axes = plt.subplots(1, len(target_cols), figsize=(25, 5))
            for i, target_col in enumerate(target_cols):
                pred_df = pd.read_csv(f'{model_output_path}pred_df_rf_model_{target_col}.csv')
                tmp_obs = pred_df['LABEL'].values
                tmp_pred = pred_df['PRED'].values
                r2 = r2_score(tmp_obs, tmp_pred)
                rmse = root_mean_squared_error(tmp_obs, tmp_pred)
                mae = mean_absolute_error(tmp_obs, tmp_pred)
                tmp_df = pd.DataFrame({'Target': [target_col], 'Fold': [fold+1], 'RMSE': [rmse], 'R2': [r2], 'MAE': [mae]})
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
            score_df.groupby("Target", as_index=False)[["RMSE", "R2", "MAE"]]
            .mean(numeric_only=True)
        )
        mean_df["Fold"] = "mean"
        std_df = (
            score_df.groupby("Target", as_index=False)[["RMSE", "R2", "MAE"]]
            .std(ddof=1, numeric_only=True)
        )
        std_df["Fold"] = "std"
        score_df = pd.concat([score_df, mean_df, std_df], ignore_index=True)
        fold_order = list(map(str, range(1, 11))) + ["mean", "std"]
        score_df["Fold"] = score_df["Fold"].astype(str)
        score_df["Fold"] = pd.Categorical(score_df["Fold"], categories=fold_order, ordered=True)
        score_df = score_df.sort_values(["Target", "Fold"]).reset_index(drop=True)
        score_df.to_csv(f'{model_path}/{split_pattern}/Table01_CV_RF_tuned_metrics.csv', index=False)

        best_params_df = pd.concat(best_params_dfs, ignore_index=True)
        best_params_df.to_csv(f'{model_path}/{split_pattern}/Table01_CV_RF_tuned_best_params.csv', index=False)
