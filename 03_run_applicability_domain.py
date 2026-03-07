import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.PandasTools import AddMoleculeColumnToFrame
from rdkit.DataStructs import BulkTanimotoSimilarity
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import shap
from utils_for_admet_model.execute_model import execute_mt_cv_train
from utils_for_admet_model.applicability_domain import calc_ad_metrics


if __name__ == "__main__":

    # Settings
    input_path = 'training_data/'
    split_pattern = 'random_split'
    model_path = 'trained_model/'
    
    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)
    model_output_path = f'{model_path}/{split_pattern}/multi_task_cv_ad/'

    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    trainval_id = pd.read_csv(
        f'{model_path}/{split_pattern}/trainval_id.csv', dtype={'COMPID': str}
    )['COMPID'].values
    test_id = pd.read_csv(
        f'{model_path}/{split_pattern}/test_id.csv', dtype={'COMPID': str}
    )['COMPID'].values

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

    Kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (cv_trainval_idx, cv_test_idx) in enumerate(Kf.split(trainval_id)):
        cv_trainval_id = trainval_id[cv_trainval_idx]
        cv_test_id = trainval_id[cv_test_idx]
        
        train_results, test_results = execute_mt_cv_train(
            f'{model_output_path}error_model/cv{fold}/', merge_df, target_cols, 
            cv_trainval_id, cv_test_id, study_params
        )

    # Calculate applicability domain metrics for the main model and error models
    calc_ad_metrics(model_output_path, merge_df)
    for fold in range(10):
        calc_ad_metrics(f'{model_output_path}error_model/cv{fold}/', merge_df)

    # Develop error classification models and plot the relationship between reliability scores and unsigned errors
    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    for i, target_col in enumerate(target_cols):
        err_dataset = pd.DataFrame()
        for fold in range(10):
            fold_data = pd.read_csv(
                f'{model_output_path}error_model/cv{fold}/test_ad_df_{target_col}.csv', 
                dtype={'COMPID': str}
            )
            err_dataset = pd.concat([err_dataset, fold_data], ignore_index=True)
            
        err_features = [
            f'{target_col}_pred_std', 
            f'{target_col}_wRMSD1', 
            f'{target_col}_wRMSD2', 
            f'{target_col}_SIM1', 
            f'{target_col}_SIM5', 
            f'{target_col}_pred_mean'
        ]
        err_target = 'UE'

        err_X = err_dataset[err_features]
        err_y = [0 if i < np.log10(2) else 1 for i in err_dataset[err_target]]

        err_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        err_model.fit(err_X, err_y)
        pickle.dump(err_model, open(f'{model_output_path}error_model/err_model_{target_col}.pkl', 'wb'))

        test_ad_df = pd.read_csv(f'{model_output_path}test_ad_df_{target_col}.csv', dtype={'COMPID': str})
        test_ad_df['pred_UE'] = err_model.predict(test_ad_df[err_features])
        test_ad_df['proba_UE'] = err_model.predict_proba(test_ad_df[err_features])[:, 0]
        accuracy = accuracy_score(
            [0 if i < np.log10(2) else 1 for i in test_ad_df[err_target]],
            test_ad_df['pred_UE']
        )

        axes[i].set_title(f'{target_col} (Accuracy: {accuracy:.3f})', fontsize=14)
        sns.regplot(
            ax=axes[i], data=test_ad_df, x='proba_UE', y='UE', 
            scatter_kws={'color': 'None', 'edgecolor': 'black'}, 
            line_kws={'linewidth': 2, 'color': 'red', 'label': 'regression line'}
        )
        axes[i].axhline(np.log10(2), 0, 1, color='black', linestyle='--', label='2-fold error')
        axes[i].axvline(0.5, color='black', linestyle='-', linewidth=0.5)
        axes[i].set_xlim([0, 1])
        if i == 0:
            axes[i].set_ylabel('Unsigned Error', fontsize=14)
        else:
            axes[i].set_ylabel('')
        axes[i].set_xlabel('Reliability Score', fontsize=14)
        axes[i].legend()
    fig.tight_layout()
    plt.savefig(f'{model_output_path}Figure03_err_pred_vs_ue.png', dpi=300)
    plt.close()

    # Generate SHAP summary plots for the error classification models
    for target_col in target_cols:
        error_model = pickle.load(open(f'{model_output_path}error_model/err_model_{target_col}.pkl', 'rb'))
        test_ad_df = pd.read_csv(f'{model_output_path}test_ad_df_{target_col}.csv', dtype={'COMPID': str})
        err_features = [
            f'{target_col}_pred_std', 
            f'{target_col}_wRMSD1', 
            f'{target_col}_wRMSD2', 
            f'{target_col}_SIM1', 
            f'{target_col}_SIM5', 
            f'{target_col}_pred_mean'
        ]
        err_target = 'UE'
        err_X_test = test_ad_df[err_features].copy()
        err_X_test.rename(columns={
            f'{target_col}_pred_std': 'PRED_STD', 
            f'{target_col}_wRMSD1': 'wRMSD1', 
            f'{target_col}_wRMSD2': 'wRMSD2', 
            f'{target_col}_SIM1': 'SIM1', 
            f'{target_col}_SIM5': 'SIM5', 
            f'{target_col}_pred_mean': 'PRED'
        }, inplace=True)
        explainer = shap.TreeExplainer(error_model)
        shap_values = explainer.shap_values(err_X_test)

        shap.summary_plot(
            shap_values[:, :, 0],
            features=err_X_test,
            cmap='bwr',
            alpha=0.9,
            show=False
        )
        plt.title(f'SHAP Summary Plot for {target_col} Error Model', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{model_output_path}error_model/Figure04_shap_summary_{target_col}.png', dpi=300)
        plt.close()


    # Analyze the relationship between reliability score thresholds and prediction accuracy
    acc_df = pd.DataFrame(columns=['Target', 'Threshold','Total_Num_Data', 'Num_Data', 'R2', 'RMSE'])
    for i, target_col in enumerate(target_cols):
        err_dataset = pd.DataFrame()
        for fold in range(10):
            fold_data = pd.read_csv(f'{model_output_path}error_model/cv{fold}/test_ad_df_{target_col}.csv', dtype={'COMPID': str})
            err_dataset = pd.concat([err_dataset, fold_data], ignore_index=True)
            
        err_features = [
            f'{target_col}_pred_std', 
            f'{target_col}_wRMSD1', 
            f'{target_col}_wRMSD2', 
            f'{target_col}_SIM1', 
            f'{target_col}_SIM5', 
            f'{target_col}_pred_mean'
        ]
        err_target = 'UE'

        err_X = err_dataset[err_features]
        err_y = [0 if i < np.log10(2) else 1 for i in err_dataset[err_target]]

        err_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        err_model.fit(err_X, err_y)
        pickle.dump(err_model, open(f'{model_output_path}error_model/err_model_{target_col}.pkl', 'wb'))

        test_ad_df = pd.read_csv(f'{model_output_path}test_ad_df_{target_col}.csv', dtype={'COMPID': str})
        test_ad_df['pred_UE'] = err_model.predict(test_ad_df[err_features])
        test_ad_df['proba_UE'] = err_model.predict_proba(test_ad_df[err_features])[:, 0]

        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            total_num_data = test_ad_df.shape[0]
            tmp_df = test_ad_df.query(f'proba_UE >= {threshold}')
            num_data = tmp_df.shape[0]
            r2 = r2_score(tmp_df[target_col], tmp_df[f"{target_col}_pred_mean"])
            rmse = root_mean_squared_error(tmp_df[target_col], tmp_df[f"{target_col}_pred_mean"])
            acc_df = pd.concat([
                acc_df,
                pd.DataFrame({
                    'Target': [target_col],
                    'Threshold': [threshold],
                    'Total_Num_Data': [total_num_data],
                    'Fraction': [num_data / total_num_data],
                    'Num_Data': [num_data],
                    'R2': [r2],
                    'RMSE': [rmse]
                })
            ], ignore_index=True)
    acc_df.to_csv(f'{model_output_path}error_model/accuracy_vs_threshold.csv', index=False)

    AddMoleculeColumnToFrame(merge_df, 'SMILES')
    merge_df_train = merge_df[merge_df['COMPID'].isin(trainval_id)]
    merge_df_test = merge_df[merge_df['COMPID'].isin(test_id)]

    fps_train = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in merge_df_train.ROMol]
    fps_test = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in merge_df_test.ROMol]

    fp_test_max, fp_test_mean, fp_test_n5 = [], [], []
    for i in range(merge_df_test.shape[0]):
        fp_sim = BulkTanimotoSimilarity(fps_test[i], fps_train)
        fp_test_max.append(np.max(fp_sim))
        fp_test_mean.append(np.mean(fp_sim))
        fp_test_n5.append(np.mean(sorted(fp_sim)[-5:]))

    fp_df_test = pd.DataFrame(index=merge_df_test['COMPID'].values)
    fp_df_test.index.name = 'COMPID'
    fp_df_test['FP_MAX'] = fp_test_max
    fp_df_test['FP_MEAN'] = fp_test_mean
    fp_df_test['FP_N5'] = fp_test_n5

    acc_df = pd.DataFrame(columns=['Target', 'Threshold','Total_Num_Data', 'Num_Data', 'R2', 'RMSE'])
    for i, target_col in enumerate(target_cols):
        test_ad_df = pd.read_csv(f'{model_output_path}test_ad_df_{target_col}.csv', dtype={'COMPID': str})
        test_ad_df['FP_MAX'] = fp_df_test.loc[test_ad_df['COMPID'].values]['FP_MAX'].values

        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            total_num_data = test_ad_df.shape[0]
            tmp_df = test_ad_df.query(f'FP_MAX >= {threshold}')
            num_data = tmp_df.shape[0]
            r2 = r2_score(tmp_df[target_col], tmp_df[f"{target_col}_pred_mean"])
            rmse = root_mean_squared_error(tmp_df[target_col], tmp_df[f"{target_col}_pred_mean"])
            acc_df = pd.concat([
                acc_df,
                pd.DataFrame({
                    'Target': [target_col],
                    'Threshold': [threshold],
                    'Total_Num_Data': [total_num_data],
                    'Fraction': [num_data / total_num_data],
                    'Num_Data': [num_data],
                    'R2': [r2],
                    'RMSE': [rmse]
                })
            ], ignore_index=True)
    acc_df.to_csv(f'{model_output_path}error_model/accuracy_vs_tanimoto_threshold.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex='col', sharey='row')
    # Reliability Score
    acc_df = pd.read_csv(f'{model_output_path}error_model/accuracy_vs_threshold.csv')
    for target_col in target_cols:
        tmp_df = acc_df[acc_df['Target'] == target_col]
        axes[0, 0].plot(tmp_df['Threshold'], tmp_df['RMSE'], marker='o', label=target_col)
        axes[1, 0].plot(tmp_df['Threshold'], tmp_df['Fraction'], marker='o', label=target_col)
    # Tanimoto index
    acc_df = pd.read_csv(f'{model_output_path}error_model/accuracy_vs_tanimoto_threshold.csv')
    for target_col in target_cols:
        tmp_df = acc_df[acc_df['Target'] == target_col]
        axes[0, 1].plot(tmp_df['Threshold'], tmp_df['RMSE'], marker='o', label=target_col)
        axes[1, 1].plot(tmp_df['Threshold'], tmp_df['Fraction'], marker='o', label=target_col)

    labels = ['(A)', '(C)', '(B)', '(D)']
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(
            0.97, 0.95, labels[i], transform=ax.transAxes,
            ha='right', va='top', fontsize=16, fontweight='bold'
        )
    axes[1, 0].legend()
    axes[0, 0].set_ylabel('RMSE')
    axes[1, 0].set_ylabel('Coverage Fraction')
    axes[1, 0].set_xlabel('Reliability Score Threshold')
    axes[1, 1].set_xlabel('Similarity Score Threshold')
    fig.tight_layout()
    plt.savefig(f'{model_output_path}error_model/Figure05_rmse_coverage_vs_thresholds.png', dpi=300)
    plt.close()
