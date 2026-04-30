import os
import json
from collections import Counter
import pandas as pd
import numpy as np
import yaml
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import rdMolDraw2D
from captum.attr import IntegratedGradients
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
    out_dir = 'results_from_manuscript/interpretability/'
    ig_dir = f'{out_dir}ig_case_studies/'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ig_dir, exist_ok=True)

    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']

    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)


    # Load 5-cv MT-GCN ensemble
    with open(f'{ad_dir}/study_params.yaml') as f:
        study_params = yaml.safe_load(f)
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    smiles_to_graph = SMILESToBigraph(node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)

    encoders = []
    decoders = []
    for cv in range(5):
        enc = GraphEncoderGCN(
            in_feats=atom_featurizer.feat_size('h'),
            hidden_feats=study_params['gcn_hidden_feats'],
            activation=[nn.LeakyReLU(), nn.LeakyReLU()],
            dropout=study_params['gcn_dropout'],
        ).to(device)
        dec = DNN(
            input_dim=2 * study_params['gcn_hidden_feats'][-1],
            hidden_dims=study_params['dnn_hidden_dims'],
            output_dim=study_params['dnn_output_dim'],
            dropout=study_params['dnn_dropout'],
        ).to(device)
        enc.load_state_dict(torch.load(f'{ad_dir}/cv{cv}/encoder.pth', map_location=device, weights_only=True))
        dec.load_state_dict(torch.load(f'{ad_dir}/cv{cv}/decoder.pth', map_location=device, weights_only=True))
        enc.eval(); dec.eval()
        encoders.append(enc); decoders.append(dec)


    # Cross-assay prediction correlation on the 10% holdout
    holdout_pred = pd.read_csv(f'{ad_dir}/test_pred_group_df.csv')
    pred_cols = [f'{a}_pred_mean' for a in target_cols]
    corr_mat = holdout_pred[pred_cols].corr(method='pearson').values
    pd.DataFrame(corr_mat, index=target_cols, columns=target_cols).to_csv(
        f'{out_dir}Table_I1_cross_assay_pred_correlation.csv'
    )

    n_assays = len(target_cols)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_assays)); ax.set_yticks(range(n_assays))
    ax.set_xticklabels(target_cols, rotation=45, ha='right'); ax.set_yticklabels(target_cols)
    for i in range(n_assays):
        for j in range(n_assays):
            color = 'white' if abs(corr_mat[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr_mat[i, j]:.2f}', ha='center', va='center', color=color, fontsize=9)
    ax.set_title('Cross-assay correlation of MT-GCN predictions\n(holdout test set)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(f'{out_dir}Fig_I1_cross_assay_pred_correlation.png', dpi=300)
    plt.close()


    # Readout dims vs physicochemical descriptors (on holdout)
    holdout_merge = holdout_pred.merge(merge_df[['COMPID', 'SMILES']], on='COMPID', how='left')
    smiles_list = holdout_merge['SMILES'].tolist()
    compids = holdout_merge['COMPID'].tolist()

    readout_dim = 2 * study_params['gcn_hidden_feats'][-1]
    readouts = np.zeros((len(smiles_list), readout_dim))
    with torch.no_grad():
        for k, smi in enumerate(smiles_list):
            g = smiles_to_graph(smi)
            if g is None:
                readouts[k, :] = np.nan
                continue
            g = g.to(device)
            fs = []
            for enc in encoders:
                f = enc(g, g.ndata['h'].float()).cpu().numpy().ravel()
                fs.append(f)
            readouts[k, :] = np.mean(np.stack(fs, axis=0), axis=0)

    prop_rows = []
    for smi in smiles_list:
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
    physchem = pd.DataFrame(prop_rows)
    descriptor_cols = ['logP', 'MW', 'TPSA', 'HBD', 'HBA']
    pearson = np.zeros((readout_dim, len(descriptor_cols)))
    for j, col in enumerate(descriptor_cols):
        xj = physchem[col].values
        mask = ~np.isnan(xj) & ~np.any(np.isnan(readouts), axis=1)
        for k in range(readout_dim):
            yk = readouts[mask, k]
            xj_m = xj[mask]
            if np.std(yk) < 1e-12 or np.std(xj_m) < 1e-12:
                pearson[k, j] = 0.0
            else:
                pearson[k, j] = float(np.corrcoef(yk, xj_m)[0, 1])
    corr_df = pd.DataFrame(pearson, columns=descriptor_cols)
    corr_df['readout_dim'] = np.arange(readout_dim)
    corr_df = corr_df[['readout_dim'] + descriptor_cols]
    corr_df.to_csv(f'{out_dir}Table_I2_readout_vs_physchem_correlation.csv', index=False)

    top_rows = []
    for col in descriptor_cols:
        j = descriptor_cols.index(col)
        abs_r = np.abs(pearson[:, j])
        top_idx = np.argsort(-abs_r)[:10]
        for rank, idx in enumerate(top_idx):
            top_rows.append({
                'descriptor': col,
                'rank': rank + 1,
                'readout_dim': int(idx),
                'pearson_r': float(pearson[idx, j]),
            })
    pd.DataFrame(top_rows).to_csv(f'{out_dir}Table_I2b_readout_vs_physchem_top_dims.csv', index=False)

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(pearson, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(descriptor_cols)))
    ax.set_xticklabels(descriptor_cols, rotation=45, ha='right')
    ax.set_ylabel('MT-GCN readout dimension (0..511)')
    ax.set_title('Pearson r between MT-GCN readout dims and physchem descriptors')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(f'{out_dir}Fig_I2_readout_vs_physchem_heatmap.png', dpi=200)
    plt.close()


    # RRCK failure-mode analysis (scaffold split CV predictions)
    fold_rows = []
    split_pattern = 'scaffold_split_cv'
    for fold in range(1, 11):
        fold_dir = f'{model_path}/{split_pattern}/fold_{fold}'
        if not os.path.isdir(fold_dir):
            continue
        for assay in target_cols:
            st_path = f'{fold_dir}/single_task_cv/{assay}/test_pred_group_df.csv'
            mt_path = f'{fold_dir}/multi_task_cv/test_pred_group_df.csv'
            if not (os.path.exists(st_path) and os.path.exists(mt_path)):
                continue
            st = pd.read_csv(st_path)[['COMPID', f'{assay}_pred_mean']].rename(columns={f'{assay}_pred_mean': 'y_st'})
            mt_all = pd.read_csv(mt_path)
            mt = mt_all[['COMPID', f'{assay}_pred_mean']].rename(columns={f'{assay}_pred_mean': 'y_mt'})
            true_df = merge_df[['COMPID', assay]].rename(columns={assay: 'y_true'}).dropna(subset=['y_true'])
            merged = true_df.merge(st, on='COMPID', how='inner').merge(mt, on='COMPID', how='inner')
            merged['Target'] = assay; merged['Fold'] = fold
            merged['err_st'] = merged['y_st'] - merged['y_true']
            merged['err_mt'] = merged['y_mt'] - merged['y_true']
            fold_rows.append(merged[['COMPID', 'Fold', 'Target', 'y_true', 'y_st', 'y_mt', 'err_st', 'err_mt']])
    fold_preds = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()

    if not fold_preds.empty:
        sub = fold_preds[fold_preds['Target'] == 'RRCK'].copy()
        sub['abs_err_st'] = sub['err_st'].abs()
        sub['abs_err_mt'] = sub['err_mt'].abs()
        smi_lookup = merge_df.set_index('COMPID')['SMILES'].to_dict()
        sub['SMILES'] = sub['COMPID'].map(smi_lookup)

        scaffolds_map = []
        for smi in sub['SMILES'].values:
            if not isinstance(smi, str):
                scaffolds_map.append('')
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaffolds_map.append('')
                continue
            scaffolds_map.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        sub['SCAFFOLD'] = scaffolds_map

        top_n = 30
        worst_st = sub.sort_values('abs_err_st', ascending=False).head(top_n).copy()
        worst_st[['COMPID', 'Fold', 'y_true', 'y_st', 'y_mt', 'err_st', 'err_mt', 'SMILES', 'SCAFFOLD']].to_csv(
            f'{out_dir}Table_I3_RRCK_top_ST_errors.csv', index=False,
        )

        top_sc = Counter(worst_st['SCAFFOLD'].tolist())
        all_sc = Counter(sub['SCAFFOLD'].tolist())
        total_top = sum(top_sc.values()) or 1
        total_all = sum(all_sc.values()) or 1
        enr_rows = []
        for sc, c_top in top_sc.most_common():
            c_all = all_sc.get(sc, 0)
            frac_top = c_top / total_top
            frac_all = c_all / total_all
            enr_rows.append({
                'scaffold': sc if sc else '(no scaffold)',
                'count_in_top_errors': c_top,
                'count_overall': c_all,
                'frac_in_top_errors': frac_top,
                'frac_overall': frac_all,
                'log2_enrichment': float(np.log2((frac_top + 1e-6) / (frac_all + 1e-6))),
            })
        pd.DataFrame(enr_rows).to_csv(f'{out_dir}Table_I3b_RRCK_top_error_scaffold_enrichment.csv', index=False)

        summary = {
            'n_top': top_n,
            'n_MT_better_than_ST': int((worst_st['abs_err_mt'] < worst_st['abs_err_st']).sum()),
            'n_MT_halves_ST_error': int((worst_st['abs_err_mt'] < 0.5 * worst_st['abs_err_st']).sum()),
            'mean_abs_err_st_in_top': float(worst_st['abs_err_st'].mean()),
            'mean_abs_err_mt_in_top': float(worst_st['abs_err_mt'].mean()),
        }
        with open(f'{out_dir}Table_I3c_RRCK_top_error_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


    # Case-study selection (random_split_cv with RF_tuned / ST-GCN / MT-GCN available)
    case_rows = []
    split_pattern = 'random_split_cv'
    for fold in range(1, 11):
        fold_dir = f'{model_path}/{split_pattern}/fold_{fold}'
        if not os.path.isdir(fold_dir):
            continue
        for assay in target_cols:
            rf_path = f'{fold_dir}/RandomForest_tuned/pred_df_rf_model_{assay}.csv'
            st_path = f'{fold_dir}/single_task_cv/{assay}/test_pred_group_df.csv'
            mt_path = f'{fold_dir}/multi_task_cv/test_pred_group_df.csv'
            if not (os.path.exists(rf_path) and os.path.exists(st_path) and os.path.exists(mt_path)):
                continue
            rf = pd.read_csv(rf_path).rename(columns={'LABEL': 'y_true', 'PRED': 'y_rf_tuned'})
            st = pd.read_csv(st_path)[['COMPID', f'{assay}_pred_mean']].rename(columns={f'{assay}_pred_mean': 'y_st'})
            mt = pd.read_csv(mt_path)[['COMPID', f'{assay}_pred_mean']].rename(columns={f'{assay}_pred_mean': 'y_mt'})
            merged = rf.merge(st, on='COMPID').merge(mt, on='COMPID')
            merged['Target'] = assay; merged['Fold'] = fold
            merged['abs_err_mt'] = (merged['y_mt'] - merged['y_true']).abs()
            merged['abs_err_st'] = (merged['y_st'] - merged['y_true']).abs()
            merged['abs_err_rf'] = (merged['y_rf_tuned'] - merged['y_true']).abs()
            case_rows.append(merged)
    case_preds = pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()

    if not case_preds.empty:
        selected_rows = []
        for assay in target_cols:
            s = case_preds[case_preds['Target'] == assay]
            cand = s[(s['abs_err_mt'] < 0.3) & (s['abs_err_st'] > 0.5) & (s['abs_err_rf'] > 0.5)].copy()
            cand['advantage'] = (cand['abs_err_st'] - cand['abs_err_mt']) + (cand['abs_err_rf'] - cand['abs_err_mt'])
            cand = cand.sort_values('advantage', ascending=False).head(2)
            selected_rows.append(cand)
        case_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
        if not case_df.empty:
            case_df = case_df.merge(merge_df[['COMPID', 'SMILES']], on='COMPID', how='left')
            case_df.to_csv(f'{out_dir}Table_I4_case_studies.csv', index=False)


    # Integrated Gradients atom attributions on case-study compounds
    def ig_forward(atom_feats, graph, encoder, decoder, assay_idx):
        af = atom_feats.squeeze(0)
        gf = encoder(graph, af)
        pred = decoder(gf)
        return pred[:, assay_idx]

    ig_rows = []
    if not case_preds.empty and not case_df.empty:
        for _, row in case_df.iterrows():
            smi = row['SMILES']
            target_col = row['Target']
            assay_idx = target_cols.index(target_col)
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                continue
            graph = smiles_to_graph(smi).to(device)
            atom_feats_input = graph.ndata['h'].float().unsqueeze(0)
            atom_feats_input.requires_grad_(True)
            baseline = torch.zeros_like(atom_feats_input)

            ensemble_attr = torch.zeros_like(atom_feats_input)
            ensemble_preds = []
            for enc, dec in zip(encoders, decoders):
                enc.eval(); dec.eval()
                ig = IntegratedGradients(ig_forward)
                attr = ig.attribute(
                    atom_feats_input,
                    baselines=baseline,
                    additional_forward_args=(graph, enc, dec, assay_idx),
                    n_steps=50,
                    internal_batch_size=1,
                )
                ensemble_attr += attr.detach()
                with torch.no_grad():
                    gf = enc(graph, graph.ndata['h'].float())
                    ensemble_preds.append(float(dec(gf)[0, assay_idx].cpu().item()))
            ensemble_attr /= len(encoders)
            atom_scores = ensemble_attr.squeeze(0).sum(dim=1).cpu().numpy()

            max_abs = float(np.max(np.abs(atom_scores))) if len(atom_scores) else 1.0
            if max_abs < 1e-12:
                max_abs = 1e-12
            norm_scores = atom_scores / max_abs
            atom_colors = {}
            for k, s in enumerate(norm_scores):
                if s >= 0:
                    atom_colors[k] = (1.0, 1.0 - s, 1.0 - s)
                else:
                    atom_colors[k] = (1.0 + s, 1.0 + s, 1.0)

            drawer = rdMolDraw2D.MolDraw2DCairo(600, 450)
            drawer.drawOptions().addAtomIndices = False
            drawer.drawOptions().baseFontSize = 0.7
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer, mol,
                highlightAtoms=list(range(mol.GetNumAtoms())),
                highlightAtomColors=atom_colors,
            )
            drawer.FinishDrawing()
            fname = f'ig_{target_col}_{row["COMPID"]}.png'
            with open(f'{ig_dir}{fname}', 'wb') as f:
                f.write(drawer.GetDrawingText())

            pd.DataFrame({
                'atom_idx': np.arange(len(atom_scores)),
                'atom_symbol': [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())],
                'ig_score_signed': atom_scores,
                'ig_score_normalised': norm_scores,
            }).to_csv(f'{ig_dir}{fname.replace(".png", "_atom_scores.csv")}', index=False)

            ig_rows.append({
                'COMPID': row['COMPID'],
                'Target': target_col,
                'SMILES': smi,
                'y_true': row['y_true'],
                'y_mt': row['y_mt'],
                'y_mt_ensemble_here': float(np.mean(ensemble_preds)),
                'y_st': row['y_st'],
                'y_rf_tuned': row['y_rf_tuned'],
                'ig_png': fname,
                'max_abs_atom_score': max_abs,
                'top_atom_idx': int(np.argmax(np.abs(atom_scores))),
                'top_atom_symbol': mol.GetAtomWithIdx(int(np.argmax(np.abs(atom_scores)))).GetSymbol(),
                'top_atom_signed_score': float(atom_scores[int(np.argmax(np.abs(atom_scores)))]),
            })

    if ig_rows:
        pd.DataFrame(ig_rows).to_csv(f'{out_dir}Table_I4b_case_studies_with_IG.csv', index=False)
