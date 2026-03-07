import os
import time
import datetime
import logging
import yaml
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dgllife.utils import SMILESToBigraph
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

# DGL-Life does not support MPS yet
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

from utils_for_admet_model.datasets import ADMETDataset, admet_collate_fn
from utils_for_admet_model.models import GraphEncoderGCN, DNN
from utils_for_admet_model.dataloader_loop import train_loop, eval_loop, infer_loop
from utils_for_admet_model.utils import fix_seed, EarlyStopping, MultiTaskLoss


def run_train(merge_df, target_col, train_id, val_id, config, id_col='COMPID'):
    fix_seed(42)

    save_path = config['save_path']
    os.makedirs(save_path, exist_ok=True)
    study_params = config['study_params']

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        filename=save_path + 'training.log',
        filemode='w'
    )
    logger = logging.getLogger()

    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    smiles_to_graph = SMILESToBigraph(
        node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
    )

    # make dataset
    train_dataset = ADMETDataset(
        smiles_to_graph,
        merge_df[merge_df[id_col].isin(train_id)],
        target_col,
    )
    val_dataset = ADMETDataset(
        smiles_to_graph,
        merge_df[merge_df[id_col].isin(val_id)],
        target_col,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=study_params['batch_size'],
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

    # define model
    encoder = GraphEncoderGCN(
        in_feats=atom_featurizer.feat_size('h'),
        hidden_feats=study_params['gcn_hidden_feats'],
        activation=[nn.LeakyReLU(), nn.LeakyReLU()],
        dropout=study_params['gcn_dropout'],
    ).to(device)

    decoder = DNN(
        input_dim=2 * study_params['gcn_hidden_feats'][-1],
        hidden_dims=study_params['dnn_hidden_dims'],
        output_dim=study_params['dnn_output_dim'],
        dropout=study_params['dnn_dropout'],
    ).to(device)

    optimizer = optim.AdamW(params=[
        {
            'params': encoder.parameters(),
            'lr': study_params['gcn_lr'],
            'initial_lr': study_params['gcn_lr'],
        },
        {
            'params': decoder.parameters(),
            'lr': study_params['dnn_lr'],
            'initial_lr': study_params['dnn_lr'],
        }
    ], weight_decay=1e-3)

    if study_params['dnn_output_dim'] == 1:
        loss_func = nn.MSELoss().to(device)
    elif study_params['dnn_output_dim'] > 1:
        loss_func = MultiTaskLoss(weighted=study_params['multi_task_loss_weighted']).to(device)

    patience = study_params['earlystopping_patience']
    earlystopping = EarlyStopping(
        patience=patience, verbose=False, save_model=True,
        save_path=[
            save_path + 'encoder.pth',
            save_path + 'decoder.pth'
        ]
    )
    if study_params['scheduler']:
        warmup_epochs = study_params['warmup_epochs']
        effective_epochs = study_params['effective_epochs']
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=effective_epochs - warmup_epochs,
            eta_min=study_params['eta_min']
        )
    else:
        scheduler = None
        warmup_epochs = None

    # start training
    start_time = time.time()
    max_epoch = 10000
    for epoch in range(max_epoch):
        train_loss = train_loop(
            encoder, decoder, train_loader, loss_func, optimizer, device,
            scheduler, epoch, warmup_epochs
        )
        val_loss, _ = eval_loop(
            encoder, decoder, val_loader, loss_func, device
        )
        earlystopping(val_loss, [encoder, decoder])

        lr_values = [group['lr'] for group in optimizer.param_groups]
        logger.info(
            f'Epoch [{epoch}/{max_epoch}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
            f'best score: {-earlystopping.best_score:.6f}, '
            f'encoder_lr: {lr_values[0]:.8f}, decoder_lr: {lr_values[1]:.8f}'
        )

        if earlystopping.early_stop:
            break

    end_time = time.time()
    td = datetime.timedelta(seconds=end_time - start_time)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return epoch, td, -earlystopping.best_score


def run_eval(merge_df, target_col, test_id, config, id_col='COMPID'):
    save_path = config['save_path']
    study_params = config['study_params']

    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    smiles_to_graph = SMILESToBigraph(
        node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
    )

    # make dataset
    test_dataset = ADMETDataset(
        smiles_to_graph,
        merge_df[merge_df[id_col].isin(test_id)],
        target_col,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        collate_fn=admet_collate_fn,
    )

    # define model
    encoder = GraphEncoderGCN(
        in_feats=atom_featurizer.feat_size('h'),
        hidden_feats=study_params['gcn_hidden_feats'],
        activation=[nn.LeakyReLU(), nn.LeakyReLU()],
        dropout=study_params['gcn_dropout'],
    ).to(device)

    decoder = DNN(
        input_dim=2 * study_params['gcn_hidden_feats'][-1],
        hidden_dims=study_params['dnn_hidden_dims'],
        output_dim=study_params['dnn_output_dim'],
        dropout=study_params['dnn_dropout'],
    ).to(device)

    # load best model
    encoder.load_state_dict(torch.load(
        save_path + 'encoder.pth', weights_only=True
    ))
    decoder.load_state_dict(torch.load(
        save_path + 'decoder.pth', weights_only=True
    ))

    if study_params['dnn_output_dim'] == 1:
        loss_func = nn.MSELoss().to(device)
    elif study_params['dnn_output_dim'] > 1:
        loss_func = MultiTaskLoss(weighted=study_params['multi_task_loss_weighted']).to(device)

    test_loss, outputs = eval_loop(
        encoder, decoder, test_loader, loss_func, device
    )
    return test_loss, outputs


def run_infer(merge_df, config, suffix=''):
    save_path = config['save_path']
    study_params = config['study_params']

    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    smiles_to_graph = SMILESToBigraph(
        node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
    )

    infer_dataset = ADMETDataset(
        smiles_to_graph=smiles_to_graph,
        merge_df=merge_df,
        target_col=None,  # Inference does not require target_col
    )
    infer_loader = DataLoader(
        dataset=infer_dataset,
        batch_size=len(infer_dataset),
        shuffle=False,
        collate_fn=admet_collate_fn,
    )

    # define model
    encoder = GraphEncoderGCN(
        in_feats=atom_featurizer.feat_size('h'),
        hidden_feats=study_params['gcn_hidden_feats'],
        activation=[nn.LeakyReLU(), nn.LeakyReLU()],
        dropout=study_params['gcn_dropout'],
    ).to(device)

    decoder = DNN(
        input_dim=2 * study_params['gcn_hidden_feats'][-1],
        hidden_dims=study_params['dnn_hidden_dims'],
        output_dim=study_params['dnn_output_dim'],
        dropout=study_params['dnn_dropout'],
    ).to(device)

    # load best model
    encoder.load_state_dict(torch.load(
        save_path + 'encoder.pth', weights_only=True
    ))
    decoder.load_state_dict(torch.load(
        save_path + 'decoder.pth', weights_only=True
    ))

    outputs = infer_loop(
        encoder, decoder, infer_loader, device
    )
    return outputs


def execute_mt_cv_train(model_output_path, merge_df, target_cols, trainval_id, test_id, study_params):

    os.makedirs(model_output_path)

    pd.Series(trainval_id, name='COMPID').to_csv(f'{model_output_path}trainval_id.csv', index=False)
    pd.Series(test_id, name='COMPID').to_csv(f'{model_output_path}test_id.csv', index=False)

    with open(f'{model_output_path}study_params.yaml', 'w') as f:
        yaml.dump(study_params, f, allow_unicode=True)

    train_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_id)):
        train_id = trainval_id[train_idx]
        val_id = trainval_id[val_idx]
        os.makedirs(f'{model_output_path}cv{fold}/', exist_ok=True)
        pd.Series(train_id, name='COMPID').to_csv(f'{model_output_path}cv{fold}/train_id.csv', index=False)
        pd.Series(val_id, name='COMPID').to_csv(f'{model_output_path}cv{fold}/val_id.csv', index=False)

        config = {
            'save_path': f'{model_output_path}cv{fold}/',
            'study_params': study_params
        }
        epoch, td, best_score = run_train(merge_df, target_cols, train_id, val_id, config)        
        train_results.append({'fold': fold, 'epoch': epoch, 'time': td.total_seconds() / 60, 'best_score': best_score})

    # Test set evaluation
    with open(f'{model_output_path}study_params.yaml', 'r') as f:
        study_params = yaml.safe_load(f)
    target_cols = study_params['target_col']
    pred_df = pd.DataFrame()
    test_results = []
    for fold in range(5):
        config = {
            'save_path': f'{model_output_path}cv{fold}/',
            'study_params': study_params
        }
        test_loss, outputs = run_eval(merge_df, target_cols, test_id, config)
        test_results.append({'fold': fold, 'test_loss': test_loss})

        tmp_pred_df = pd.DataFrame(outputs['preds'], columns=target_cols)
        tmp_pred_df.insert(loc=0, column='COMPID', value=outputs['compids'])
        tmp_pred_df.insert(loc=0, column='FOLD', value=fold)
        pred_df = pd.concat([pred_df, tmp_pred_df], ignore_index=True)

    pred_df.to_csv(f'{model_output_path}test_pred_df.csv', index=False)

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
    pred_group_df = pred_group_df.loc[test_id].reset_index()
    pred_group_df.to_csv(f'{model_output_path}test_pred_group_df.csv', index=False)

    return train_results, test_results


def execute_st_cv_train(model_output_path, merge_df, target_col, trainval_id, test_id, study_params):

    os.makedirs(model_output_path)

    pd.Series(trainval_id, name='COMPID').to_csv(f'{model_output_path}trainval_id.csv', index=False)
    pd.Series(test_id, name='COMPID').to_csv(f'{model_output_path}test_id.csv', index=False)

    with open(f'{model_output_path}study_params.yaml', 'w') as f:
        yaml.dump(study_params, f, allow_unicode=True)

    train_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_id)):
        train_id = trainval_id[train_idx]
        val_id = trainval_id[val_idx]
        os.makedirs(f'{model_output_path}cv{fold}/', exist_ok=True)
        pd.Series(train_id, name='COMPID').to_csv(f'{model_output_path}cv{fold}/train_id.csv', index=False)
        pd.Series(val_id, name='COMPID').to_csv(f'{model_output_path}cv{fold}/val_id.csv', index=False)

        config = {
            'save_path': f'{model_output_path}cv{fold}/',
            'study_params': study_params
        }
        epoch, td, best_score = run_train(merge_df[merge_df[target_col].notna()], target_col, train_id, val_id, config)        
        train_results.append({'fold': fold, 'epoch': epoch, 'time': td.total_seconds() / 60, 'best_score': best_score})

    # Test set evaluation
    with open(f'{model_output_path}study_params.yaml', 'r') as f:
        study_params = yaml.safe_load(f)
    target_col = study_params['target_col']
    pred_df = pd.DataFrame()
    test_results = []
    for fold in range(5):
        config = {
            'save_path': f'{model_output_path}cv{fold}/',
            'study_params': study_params
        }
        test_loss, outputs = run_eval(merge_df[merge_df[target_col].notna()], target_col, test_id, config)
        test_results.append({'fold': fold, 'test_loss': test_loss})

        tmp_pred_df = pd.DataFrame(outputs['preds'], columns=[target_col])
        tmp_pred_df.insert(loc=0, column='COMPID', value=outputs['compids'])
        tmp_pred_df.insert(loc=0, column='FOLD', value=fold)
        pred_df = pd.concat([pred_df, tmp_pred_df], ignore_index=True)

    pred_df.to_csv(f'{model_output_path}test_pred_df.csv')

    # Calculate mean and SD of CV predictions for test set
    pred_group_df = pred_df.groupby('COMPID').agg({target_col: ['mean', 'std']})
    pred_group_df.columns = [f"{col[0]}_pred_{col[1]}" for col in pred_group_df.columns]
    pred_group_df.to_csv(f'{model_output_path}test_pred_group_df.csv')

    return train_results, test_results