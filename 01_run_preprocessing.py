import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt


def _list_unique(x):
    if len(x) > 1:
        tmp = pd.Series.unique(x)
        tmp_str = [str(i) for i in tmp if pd.notnull(i)]
        return '_'.join(tmp_str)
    else:
        return x
        

if __name__ == "__main__":

    # Settings
    input_path = 'training_data/'
    model_path = 'trained_model/'
    os.makedirs(model_path, exist_ok=True)

    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv', dtype={'COMPID': str})
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)

    # Generate Morgan fingerprints
    morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
    fp_list = []
    for smi in merge_df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(1024, dtype=np.int8)
        if mol is not None:
            fp = morgan_gen.GetFingerprint(mol)  # ExplicitBitVect
            DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr)
    fp_df = pd.DataFrame(fp_list, columns=[f'fp_{i}' for i in range(1024)])
    fp_df.insert(0, 'COMPID', merge_df['COMPID'].values)
    fp_df.to_csv(f'{input_path}/fp_df.csv', index=False)


    # Hold-out split
    split_pattern = 'random_split'
    os.makedirs(f'{model_path}/{split_pattern}', exist_ok=True)

    trainval_id, test_id = train_test_split(
        merge_df['COMPID'].unique(), test_size=0.1, random_state=42
    )
    pd.Series(trainval_id, name='COMPID').to_csv(f'{model_path}/{split_pattern}/trainval_id.csv', index=False)
    pd.Series(test_id, name='COMPID').to_csv(f'{model_path}/{split_pattern}/test_id.csv', index=False)

    cell_types = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    bins = np.arange(-3, 4.1, 0.5)
    fig, axes = plt.subplots(1, len(cell_types), figsize=(18, 6), sharey=True, sharex=True)
    for i, ct in enumerate(cell_types):
        num_train = merge_df[merge_df['COMPID'].isin(trainval_id)][ct].dropna().shape[0]
        axes[i].hist(
            merge_df[merge_df['COMPID'].isin(trainval_id)][ct].dropna(), 
            bins=bins, alpha=0.5, label=f'Train: {num_train}'
        )
        num_test = merge_df[merge_df['COMPID'].isin(test_id)][ct].dropna().shape[0]
        axes[i].hist(
            merge_df[merge_df['COMPID'].isin(test_id)][ct].dropna(),
            bins=bins, alpha=0.5, label=f'Test: {num_test}'
        )
        num_all = merge_df[ct].dropna().shape[0]
        axes[i].set_title(f'{ct}: {num_all}')
        axes[i].set_xlabel('Papp')
        if i == 0:
            axes[i].set_ylabel('Count')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'{model_path}/{split_pattern}/FigXXX_papp_distribution.png')
    plt.close()


    # Cross-validation split
    split_pattern = 'random_split_cv'
    Kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (trainval_idx, test_idx) in enumerate(Kf.split(merge_df['COMPID'].unique())):
        fold_dir = f'{model_path}/{split_pattern}/fold_{fold+1}'
        os.makedirs(fold_dir, exist_ok=True)
        trainval_id = merge_df['COMPID'].unique()[trainval_idx]
        test_id = merge_df['COMPID'].unique()[test_idx]
        pd.Series(trainval_id, name='COMPID').to_csv(f'{fold_dir}/trainval_id.csv', index=False)
        pd.Series(test_id, name='COMPID').to_csv(f'{fold_dir}/test_id.csv', index=False)
        cell_types = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
        bins = np.arange(-3, 4.1, 0.5)
        fig, axes = plt.subplots(1, len(cell_types), figsize=(18, 6), sharey=True, sharex=True)
        for i, ct in enumerate(cell_types):
            num_train = merge_df[merge_df['COMPID'].isin(trainval_id)][ct].dropna().shape[0]
            axes[i].hist(
                merge_df[merge_df['COMPID'].isin(trainval_id)][ct].dropna(), 
                bins=bins, alpha=0.5, label=f'Train: {num_train}'
            )
            num_test = merge_df[merge_df['COMPID'].isin(test_id)][ct].dropna().shape[0]
            axes[i].hist(
                merge_df[merge_df['COMPID'].isin(test_id)][ct].dropna(),
                bins=bins, alpha=0.5, label=f'Test: {num_test}'
            )
            num_all = merge_df[ct].dropna().shape[0]
            axes[i].set_title(f'{ct}: {num_all}')
            axes[i].set_xlabel('Papp')
            if i == 0:
                axes[i].set_ylabel('Count')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(f'{fold_dir}/data_distribution.png')
        plt.close()


    # Murcko scaffolds
    scaffolds = []
    for compid, smi in zip(merge_df['COMPID'].values, merge_df['SMILES'].values):
        scaffold = ''
        if isinstance(smi, str):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append({'COMPID': compid, 'SCAFFOLD': scaffold})
    scaffold_df = pd.DataFrame(scaffolds)
    scaffold_df.to_csv(f'{input_path}/murcko_scaffolds.csv', index=False)


    # Scaffold hold-out split
    split_pattern = 'scaffold_split'
    os.makedirs(f'{model_path}/{split_pattern}', exist_ok=True)

    scaffold_groups = {}
    for compid, scaffold in zip(scaffold_df['COMPID'].values, scaffold_df['SCAFFOLD'].values):
        scaffold_groups.setdefault(scaffold, []).append(compid)

    rng = np.random.RandomState(42)
    scaffold_keys = list(scaffold_groups.keys())
    rng.shuffle(scaffold_keys)

    n_total = len(scaffold_df)
    n_target_test = int(round(n_total * 0.1))
    test_id = []
    trainval_id = []
    for scaffold in scaffold_keys:
        members = scaffold_groups[scaffold]
        if len(test_id) + len(members) <= n_target_test:
            test_id.extend(members)
        else:
            trainval_id.extend(members)
    trainval_id = np.array(trainval_id)
    test_id = np.array(test_id)
    pd.Series(trainval_id, name='COMPID').to_csv(f'{model_path}/{split_pattern}/trainval_id.csv', index=False)
    pd.Series(test_id, name='COMPID').to_csv(f'{model_path}/{split_pattern}/test_id.csv', index=False)

    cell_types = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    bins = np.arange(-3, 4.1, 0.5)
    fig, axes = plt.subplots(1, len(cell_types), figsize=(18, 6), sharey=True, sharex=True)
    for i, ct in enumerate(cell_types):
        num_train = merge_df[merge_df['COMPID'].isin(trainval_id)][ct].dropna().shape[0]
        axes[i].hist(
            merge_df[merge_df['COMPID'].isin(trainval_id)][ct].dropna(),
            bins=bins, alpha=0.5, label=f'Train: {num_train}'
        )
        num_test = merge_df[merge_df['COMPID'].isin(test_id)][ct].dropna().shape[0]
        axes[i].hist(
            merge_df[merge_df['COMPID'].isin(test_id)][ct].dropna(),
            bins=bins, alpha=0.5, label=f'Test: {num_test}'
        )
        num_all = merge_df[ct].dropna().shape[0]
        axes[i].set_title(f'{ct}: {num_all}')
        axes[i].set_xlabel('Papp')
        if i == 0:
            axes[i].set_ylabel('Count')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'{model_path}/{split_pattern}/FigXXX_papp_distribution.png')
    plt.close()


    # Scaffold cross-validation split (greedy largest-scaffold-first into smallest fold)
    split_pattern = 'scaffold_split_cv'
    rng = np.random.RandomState(42)
    scaffold_keys = list(scaffold_groups.keys())
    rng.shuffle(scaffold_keys)
    scaffold_keys.sort(key=lambda s: (-len(scaffold_groups[s]), s))

    n_splits = 10
    fold_members = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for scaffold in scaffold_keys:
        target_fold = int(np.argmin(fold_sizes))
        fold_members[target_fold].extend(scaffold_groups[scaffold])
        fold_sizes[target_fold] += len(scaffold_groups[scaffold])

    for fold in range(n_splits):
        fold_dir = f'{model_path}/{split_pattern}/fold_{fold+1}'
        os.makedirs(fold_dir, exist_ok=True)
        test_id = np.array(fold_members[fold])
        trainval_id = np.array(
            [cid for k in range(n_splits) if k != fold for cid in fold_members[k]]
        )
        pd.Series(trainval_id, name='COMPID').to_csv(f'{fold_dir}/trainval_id.csv', index=False)
        pd.Series(test_id, name='COMPID').to_csv(f'{fold_dir}/test_id.csv', index=False)
        cell_types = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
        bins = np.arange(-3, 4.1, 0.5)
        fig, axes = plt.subplots(1, len(cell_types), figsize=(18, 6), sharey=True, sharex=True)
        for i, ct in enumerate(cell_types):
            num_train = merge_df[merge_df['COMPID'].isin(trainval_id)][ct].dropna().shape[0]
            axes[i].hist(
                merge_df[merge_df['COMPID'].isin(trainval_id)][ct].dropna(),
                bins=bins, alpha=0.5, label=f'Train: {num_train}'
            )
            num_test = merge_df[merge_df['COMPID'].isin(test_id)][ct].dropna().shape[0]
            axes[i].hist(
                merge_df[merge_df['COMPID'].isin(test_id)][ct].dropna(),
                bins=bins, alpha=0.5, label=f'Test: {num_test}'
            )
            num_all = merge_df[ct].dropna().shape[0]
            axes[i].set_title(f'{ct}: {num_all}')
            axes[i].set_xlabel('Papp')
            if i == 0:
                axes[i].set_ylabel('Count')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(f'{fold_dir}/data_distribution.png')
        plt.close()