import os
import time
import datetime
import shutil
import yaml
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from dgllife.utils import SMILESToBigraph
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

from utils_for_admet_model.datasets import ADMETDataset, admet_collate_fn
from utils_for_admet_model.models import GraphEncoderGCN, DNN
from utils_for_admet_model.dataloader_loop import train_loop, eval_loop
from utils_for_admet_model.utils import fix_seed, EarlyStopping


# Caco-2 pre-trained -> per-target head + encoder fine-tuning with small encoder lr
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if __name__ == "__main__":

    # Settings
    input_path = 'training_data/'
    model_path = 'trained_model/'

    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)

    study_params = {
        'target_col': None,
        'batch_size': 64,
        'gcn_hidden_feats': [256, 256],
        'gcn_dropout': [0.1, 0.1],
        'dnn_hidden_dims': [256, 64],
        'dnn_dropout': [0.2, 0.2],
        'dnn_output_dim': 1,
        'gcn_lr': 1e-5,
        'dnn_lr': 1e-4,
        'scheduler': False,
        'warmup_epochs': None,
        'effective_epochs': None,
        'eta_min': None,
        'earlystopping_patience': 50,
        'multi_task_loss_weighted': None,
    }
    max_epoch = 500

    for split_pattern in ['random_split_cv', 'scaffold_split_cv']:

        target_cols = ['LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
        dfs = []
        for fold in range(10):
            trainval_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/trainval_id.csv')['COMPID'].values
            test_id = pd.read_csv(f'{model_path}/{split_pattern}/fold_{fold+1}/test_id.csv')['COMPID'].values

            caco2_pretrained_dir = f'{model_path}/{split_pattern}/fold_{fold+1}/single_task_cv/Caco-2/'
            model_output_path = f'{model_path}/{split_pattern}/fold_{fold+1}/transfer_learning_caco2_finetune/'
            os.makedirs(model_output_path, exist_ok=True)

            for target_col in target_cols:
                target_model_output_path = f'{model_output_path}{target_col}/'
                os.makedirs(target_model_output_path, exist_ok=True)
                params = dict(study_params)
                params['target_col'] = target_col
                with open(f'{target_model_output_path}study_params.yaml', 'w') as f:
                    yaml.dump(params, f, allow_unicode=True)

                atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
                bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
                smiles_to_graph = SMILESToBigraph(
                    node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
                )

                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                for cv_fold, (train_idx, val_idx) in enumerate(kf.split(trainval_id)):
                    fix_seed(42)
                    save_path = f'{target_model_output_path}cv{cv_fold}/'
                    os.makedirs(save_path, exist_ok=True)

                    train_id = trainval_id[train_idx]
                    val_id = trainval_id[val_idx]
                    pd.Series(train_id, name='COMPID').to_csv(f'{save_path}train_id.csv', index=False)
                    pd.Series(val_id, name='COMPID').to_csv(f'{save_path}val_id.csv', index=False)

                    sub_df = merge_df[merge_df[target_col].notna()]
                    train_dataset = ADMETDataset(
                        smiles_to_graph,
                        sub_df[sub_df['COMPID'].isin(train_id)],
                        target_col,
                    )
                    val_dataset = ADMETDataset(
                        smiles_to_graph,
                        sub_df[sub_df['COMPID'].isin(val_id)],
                        target_col,
                    )
                    if len(train_dataset) == 0 or len(val_dataset) == 0:
                        continue

                    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=params['batch_size'],
                        shuffle=True,
                        collate_fn=admet_collate_fn,
                        drop_last=True,
                    )
                    val_loader = DataLoader(
                        dataset=val_dataset,
                        batch_size=len(val_dataset),
                        shuffle=False,
                        collate_fn=admet_collate_fn,
                    )

                    encoder = GraphEncoderGCN(
                        in_feats=atom_featurizer.feat_size('h'),
                        hidden_feats=params['gcn_hidden_feats'],
                        activation=[nn.LeakyReLU(), nn.LeakyReLU()],
                        dropout=params['gcn_dropout'],
                    ).to(device)
                    encoder.load_state_dict(torch.load(
                        f'{caco2_pretrained_dir}cv{cv_fold}/encoder.pth',
                        map_location=device, weights_only=True,
                    ))
                    decoder = DNN(
                        input_dim=2 * params['gcn_hidden_feats'][-1],
                        hidden_dims=params['dnn_hidden_dims'],
                        output_dim=params['dnn_output_dim'],
                        dropout=params['dnn_dropout'],
                    ).to(device)

                    optimizer = optim.AdamW(params=[
                        {
                            'params': encoder.parameters(),
                            'lr': params['gcn_lr'],
                            'initial_lr': params['gcn_lr'],
                        },
                        {
                            'params': decoder.parameters(),
                            'lr': params['dnn_lr'],
                            'initial_lr': params['dnn_lr'],
                        }
                    ], weight_decay=1e-3)
                    loss_func = nn.MSELoss().to(device)

                    earlystopping = EarlyStopping(
                        patience=params['earlystopping_patience'],
                        verbose=False, save_model=True,
                        save_path=[save_path + 'encoder.pth', save_path + 'decoder.pth'],
                    )

                    start_time = time.time()
                    for epoch in range(max_epoch):
                        encoder.train()
                        decoder.train()
                        train_loss = train_loop(
                            encoder, decoder, train_loader, loss_func, optimizer, device,
                            None, epoch, None,
                        )
                        val_loss, _ = eval_loop(encoder, decoder, val_loader, loss_func, device)
                        earlystopping(val_loss, [encoder, decoder])
                        if earlystopping.early_stop:
                            break
                    td = datetime.timedelta(seconds=time.time() - start_time)

                    if not os.path.exists(save_path + 'encoder.pth'):
                        shutil.copyfile(f'{caco2_pretrained_dir}cv{cv_fold}/encoder.pth', save_path + 'encoder.pth')

                # Inference on the outer test set using the 5-cv ensemble
                preds_per_cv = []
                sub_df = merge_df[merge_df[target_col].notna()]
                test_sub_df = sub_df[sub_df['COMPID'].isin(test_id)]
                if len(test_sub_df) == 0:
                    continue
                test_dataset = ADMETDataset(smiles_to_graph, test_sub_df, target_col)
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=len(test_dataset),
                    shuffle=False,
                    collate_fn=admet_collate_fn,
                )

                for cv_fold in range(5):
                    save_path = f'{target_model_output_path}cv{cv_fold}/'
                    if not (os.path.exists(f'{save_path}encoder.pth') and os.path.exists(f'{save_path}decoder.pth')):
                        continue
                    encoder = GraphEncoderGCN(
                        in_feats=atom_featurizer.feat_size('h'),
                        hidden_feats=params['gcn_hidden_feats'],
                        activation=[nn.LeakyReLU(), nn.LeakyReLU()],
                        dropout=params['gcn_dropout'],
                    ).to(device)
                    decoder = DNN(
                        input_dim=2 * params['gcn_hidden_feats'][-1],
                        hidden_dims=params['dnn_hidden_dims'],
                        output_dim=params['dnn_output_dim'],
                        dropout=params['dnn_dropout'],
                    ).to(device)
                    encoder.load_state_dict(torch.load(f'{save_path}encoder.pth', map_location=device, weights_only=True))
                    decoder.load_state_dict(torch.load(f'{save_path}decoder.pth', map_location=device, weights_only=True))
                    encoder.eval(); decoder.eval()

                    compids_cv = []
                    labels_cv = []
                    preds_cv = []
                    with torch.no_grad():
                        for compids, graphs, labels in test_loader:
                            graphs = graphs.to(device)
                            gf = encoder(graphs, graphs.ndata['h'].float())
                            p = decoder(gf).cpu().numpy().ravel()
                            compids_cv.extend(list(compids))
                            labels_cv.extend(labels.numpy().ravel().tolist())
                            preds_cv.extend(p.tolist())
                    cv_pred_df = pd.DataFrame({
                        'COMPID': compids_cv, 'LABEL': labels_cv, 'PRED': preds_cv, 'FOLD_INNER': cv_fold,
                    })
                    preds_per_cv.append(cv_pred_df)

                if not preds_per_cv:
                    continue
                preds_all = pd.concat(preds_per_cv, ignore_index=True)
                preds_all.to_csv(f'{target_model_output_path}test_pred_df.csv', index=False)
                pred_group_df = preds_all.groupby('COMPID').agg(
                    LABEL=('LABEL', 'first'),
                    pred_mean=('PRED', 'mean'),
                    pred_std=('PRED', 'std'),
                ).reset_index()
                pred_group_df = pred_group_df.rename(columns={
                    'pred_mean': f'{target_col}_pred_mean',
                    'pred_std': f'{target_col}_pred_std',
                })
                pred_group_df.to_csv(f'{target_model_output_path}test_pred_group_df.csv', index=False)

                obs = pred_group_df['LABEL'].values
                pred = pred_group_df[f'{target_col}_pred_mean'].values
                r2 = r2_score(obs, pred)
                rmse = root_mean_squared_error(obs, pred)
                mae = mean_absolute_error(obs, pred)
                dfs.append(pd.DataFrame({
                    'Target': [target_col], 'Fold': [fold+1],
                    'RMSE': [rmse], 'R2': [r2], 'MAE': [mae],
                }))

        score_df = pd.concat(dfs, ignore_index=True)
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
        score_df.to_csv(f'{model_path}/{split_pattern}/Table04_CV_TL_caco2_finetune_metrics.csv', index=False)
