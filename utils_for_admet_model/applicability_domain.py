import numpy as np
import pandas as pd
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from utils_for_admet_model.execute_model import run_infer


def calc_ad_metrics(model_output_path, merge_df):
    trainval_id = pd.read_csv(f'{model_output_path}trainval_id.csv', dtype=str)['COMPID'].values
    test_id = pd.read_csv(f'{model_output_path}test_id.csv', dtype=str)['COMPID'].values

    with open(f'{model_output_path}study_params.yaml', 'r') as f:
        study_params = yaml.safe_load(f)

    target_cols = study_params['target_col']
    pred_df = pd.DataFrame()
    feat_df = pd.DataFrame()
    for fold in range(5):
        config = {
            'save_path': f'{model_output_path}cv{fold}/',
            'study_params': study_params
        }

        outputs = run_infer(merge_df, config)

        tmp_pred_df = pd.DataFrame(outputs['preds'], columns=target_cols)
        tmp_pred_df.insert(loc=0, column='COMPID', value=outputs['compids'])
        tmp_pred_df.insert(loc=0, column='FOLD', value=fold)
        pred_df = pd.concat([pred_df, tmp_pred_df], ignore_index=True)

        tmp_feat_df = pd.DataFrame(
            outputs['graph_feats'], 
            columns=[f'feat_{i}' for i in range(outputs['graph_feats'].shape[1])]
        )
        tmp_feat_df.insert(loc=0, column='COMPID', value=outputs['compids'])
        tmp_feat_df.insert(loc=0, column='FOLD', value=fold)
        feat_df = pd.concat([feat_df, tmp_feat_df], axis=0)

    pred_df.to_csv(f'{model_output_path}whole_pred_df.csv', index=False)

    # Calculate mean and SD of CV predictions for test set
    pred_group_df = pred_df.groupby('COMPID').agg(
        {
            'Caco-2': ['mean', 'std'],
            'LLC-PK1': ['mean', 'std'],
            'MDCK': ['mean', 'std'],
            'PAMPA': ['mean', 'std'],
            'RRCK': ['mean', 'std']
        }
    )
    pred_group_df.columns = [f"{col[0]}_pred_{col[1]}" for col in pred_group_df.columns]
    pred_group_df = pred_group_df.loc[merge_df['COMPID']].reset_index()
    pred_group_df.to_csv(f'{model_output_path}whole_pred_group_df.csv', index=False)

    feat_group_df = feat_df.groupby('COMPID').mean().reset_index().drop(columns=['FOLD'])

    # Calculate Similarity 
    trainval_feat_df = feat_group_df[feat_group_df['COMPID'].isin(trainval_id)]
    test_feat_df = feat_group_df[feat_group_df['COMPID'].isin(test_id)]

    # Extract features for similarity calculation
    test_feats = test_feat_df.drop(columns=['COMPID']).values
    trainval_feats = trainval_feat_df.drop(columns=['COMPID']).values

    # Cosine similarity matrix (shape: [number of trainval, number of test])
    cos_sim_matrix = cosine_similarity(trainval_feats, test_feats)
    cos_sim_matrix = (cos_sim_matrix + 1) / 2  # Scale to [0, 1]
    cos_sim_df = pd.DataFrame(
        cos_sim_matrix,
        index=trainval_feat_df['COMPID'],
        columns=['SIM_' + idx for idx in test_feat_df['COMPID']]
    ).reset_index()
    
    # DA Metrics Calculation
    ad_df = pd.merge(merge_df, pred_group_df, on='COMPID', how='inner')
    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    for target_col in target_cols:
        trainval_ad_df = ad_df[ad_df['COMPID'].isin(trainval_id)]
        trainval_ad_df = trainval_ad_df[['COMPID', target_col, target_col+'_pred_mean']]
        trainval_ad_df = trainval_ad_df.dropna(subset=[target_col])
        trainval_ad_df = trainval_ad_df.merge(cos_sim_df, on='COMPID', how='inner')

        test_ad_df = ad_df[ad_df['COMPID'].isin(test_id)]
        test_ad_df = test_ad_df[['COMPID', target_col, target_col+'_pred_mean', target_col+'_pred_std']]
        test_ad_df = test_ad_df.dropna(subset=[target_col])
        test_ad_df['UE'] = (test_ad_df[target_col] - test_ad_df[f'{target_col}_pred_mean']).abs()

        for compid in test_ad_df['COMPID'].unique():
            tmp_trainval_ad_df = trainval_ad_df[['COMPID', target_col, target_col+'_pred_mean', 'SIM_'+compid]]
            tmp_trainval_ad_df = tmp_trainval_ad_df.nlargest(5, 'SIM_'+compid)

            # SIM5: Average similarity of the top 5 most similar training compounds
            sim5 = tmp_trainval_ad_df['SIM_'+compid].mean()
            
            # SIM1: Similarity of the most similar training compound
            sim1 = tmp_trainval_ad_df['SIM_'+compid].max()

            # wRMSD1: Similarity-weighted RMSD between the test prediction and experimental values of the nearest training compounds
            diff1 = (
                tmp_trainval_ad_df[target_col].values -
                test_ad_df.loc[test_ad_df['COMPID'] == compid, f'{target_col}_pred_mean'].values[0]
            )
            w = tmp_trainval_ad_df[f'SIM_{compid}'].values
            wRMSD1 = np.sqrt(np.sum(w * (diff1 ** 2)) / np.sum(w))
            
            # wRMSD2: Similarity-weighted RMSD between predictions and experimental values of the nearest training compounds
            diff2 = (
                tmp_trainval_ad_df[target_col].values -
                tmp_trainval_ad_df[f'{target_col}_pred_mean'].values
            )
            wRMSD2 = np.sqrt(np.sum(w * (diff2 ** 2)) / np.sum(w))
            
            test_ad_df.loc[test_ad_df['COMPID'] == compid, target_col+'_wRMSD1'] = wRMSD1
            test_ad_df.loc[test_ad_df['COMPID'] == compid, target_col+'_wRMSD2'] = wRMSD2
            test_ad_df.loc[test_ad_df['COMPID'] == compid, target_col+'_SIM1'] = sim1
            test_ad_df.loc[test_ad_df['COMPID'] == compid, target_col+'_SIM5'] = sim5

        test_ad_df.to_csv(f'{model_output_path}test_ad_df_{target_col}.csv', index=False)

    return None