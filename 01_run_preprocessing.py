import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
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