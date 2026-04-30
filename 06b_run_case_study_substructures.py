import os
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# Cross-assay substructure analysis on the case-study compounds selected
# in 06_run_interpretability.py. Produces four tables:
#   I5  : whole-molecule top-3 cross-assay nearest neighbours
#   I5b : per-bit cross-assay prevalence and enrichment flag
#   I5c : per-case top-5 enriched bits with fold changes and fragment SMILES
#   I5d : overlap between enriched-substructure atoms and top-IG atoms
if __name__ == "__main__":

    # Settings
    input_path = 'training_data/'
    model_path = 'trained_model/'
    out_dir = 'results_from_manuscript/interpretability/'
    ig_dir = f'{out_dir}ig_case_studies/'

    target_cols = ['Caco-2', 'LLC-PK1', 'MDCK', 'PAMPA', 'RRCK']
    n_bits = 1024
    morgan_radius = 2

    merge_df = pd.read_csv(f'{input_path}/SupportingInformation_PappValues.csv')
    merge_df.rename(columns={'ChEMBL_ID': 'COMPID'}, inplace=True)
    smi_lookup = dict(zip(merge_df['COMPID'].values, merge_df['SMILES'].values))

    trainval_id = pd.read_csv(f'{model_path}/random_split/trainval_id.csv', dtype={'COMPID': str})['COMPID'].values
    case_df = pd.read_csv(f'{out_dir}Table_I4_case_studies.csv')


    # Per-assay trainval slices
    trainval_by_assay = {}
    fp_by_assay = {}
    morgan_gen = GetMorganGenerator(radius=morgan_radius, fpSize=n_bits)
    for assay in target_cols:
        sub = merge_df[(merge_df['COMPID'].isin(trainval_id)) & (merge_df[assay].notna())].copy()
        trainval_by_assay[assay] = sub
        fps = []
        for smi in sub['SMILES'].values:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                fps.append(None)
            else:
                fps.append(morgan_gen.GetFingerprint(mol))
        fp_by_assay[assay] = fps


    # ------------------------------------------------------------------
    # Table I5: whole-molecule top-3 nearest neighbours per other assay
    # ------------------------------------------------------------------
    rows = []
    for _, case in case_df.iterrows():
        case_compid = case['COMPID']
        case_assay = case['Target']
        case_smi = smi_lookup.get(case_compid)
        if not isinstance(case_smi, str):
            continue
        case_mol = Chem.MolFromSmiles(case_smi)
        if case_mol is None:
            continue
        case_fp = morgan_gen.GetFingerprint(case_mol)
        try:
            case_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=case_mol, includeChirality=False)
        except Exception:
            case_scaffold = ''
        for other_assay in target_cols:
            if other_assay == case_assay:
                continue
            other_sub = trainval_by_assay[other_assay]
            other_fps = fp_by_assay[other_assay]
            sims = []
            for i, fp in enumerate(other_fps):
                if fp is None:
                    sims.append(-1.0)
                else:
                    sims.append(DataStructs.TanimotoSimilarity(case_fp, fp))
            sims = np.array(sims)
            order = np.argsort(-sims)[:3]
            for rank, idx in enumerate(order, start=1):
                neighbour = other_sub.iloc[int(idx)]
                neighbour_smi = neighbour['SMILES']
                neighbour_scaffold = ''
                if isinstance(neighbour_smi, str):
                    nm = Chem.MolFromSmiles(neighbour_smi)
                    if nm is not None:
                        try:
                            neighbour_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=nm, includeChirality=False)
                        except Exception:
                            neighbour_scaffold = ''
                rows.append({
                    'case_compid': case_compid,
                    'case_target_assay': case_assay,
                    'case_y_true': case['y_true'],
                    'neighbour_assay': other_assay,
                    'neighbour_rank': rank,
                    'neighbour_compid': neighbour['COMPID'],
                    'neighbour_y_true': neighbour[other_assay],
                    'tanimoto': float(sims[idx]),
                    'shared_scaffold': bool(case_scaffold) and (case_scaffold == neighbour_scaffold),
                    'neighbour_scaffold': neighbour_scaffold,
                })
    pd.DataFrame(rows).to_csv(f'{out_dir}Table_I5_case_study_cross_assay_neighbours.csv', index=False)


    # ------------------------------------------------------------------
    # Per-assay bit prevalence (fraction of trainval compounds carrying a bit)
    # ------------------------------------------------------------------
    bit_prev_by_assay = {}
    for assay in target_cols:
        fps = fp_by_assay[assay]
        counts = np.zeros(n_bits, dtype=np.int64)
        n_total = 0
        for fp in fps:
            if fp is None:
                continue
            arr = np.zeros(n_bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            counts += arr
            n_total += 1
        bit_prev_by_assay[assay] = counts / max(n_total, 1)


    # ------------------------------------------------------------------
    # Table I5b: per-bit cross-assay prevalence + enriched flag
    # Table I5c: per-case top-5 enriched bits with fold-change and fragment SMILES
    # ------------------------------------------------------------------
    enrich_thresh_ratio = 2.0
    rare_own_thresh = 0.01
    high_other_thresh = 0.05
    min_other_prev_floor = 0.02

    i5b_rows = []
    i5c_rows = []
    case_enriched_atoms = {}

    for _, case in case_df.iterrows():
        case_compid = case['COMPID']
        case_assay = case['Target']
        case_smi = smi_lookup.get(case_compid)
        if not isinstance(case_smi, str):
            continue
        case_mol = Chem.MolFromSmiles(case_smi)
        if case_mol is None:
            continue
        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(case_mol, radius=morgan_radius, nBits=n_bits, bitInfo=bit_info)
        on_bits = sorted(bit_info.keys())

        # I5b rows
        bit_records = []
        for bit in on_bits:
            own_prev = float(bit_prev_by_assay[case_assay][bit])
            other_prevs = [float(bit_prev_by_assay[a][bit]) for a in target_cols if a != case_assay]
            mean_other = float(np.mean(other_prevs))
            max_other = float(np.max(other_prevs))
            cond_a = (own_prev > 0) and (mean_other >= enrich_thresh_ratio * own_prev)
            cond_b = (own_prev < rare_own_thresh) and (max_other >= high_other_thresh)
            enriched = (cond_a or cond_b) and (max_other >= min_other_prev_floor)
            bit_records.append({
                'case_compid': case_compid,
                'case_target': case_assay,
                'bit': bit,
                'own_prev': own_prev,
                'mean_other_prev': mean_other,
                'max_other_prev': max_other,
                'enriched': bool(enriched),
            })
        i5b_rows.extend(bit_records)

        enriched_records = [r for r in bit_records if r['enriched']]
        n_enriched = len(enriched_records)

        # Top-5 enriched bits sorted by fold-change (max_other / max(own, 0.001))
        for r in enriched_records:
            r['fold_change'] = r['max_other_prev'] / max(r['own_prev'], 0.001)
        enriched_records.sort(key=lambda r: -r['fold_change'])
        top5 = enriched_records[:5]

        top_bits = []
        top_folds = []
        top_other_assays = []
        top_smiles = []
        atoms_in_enriched = set()
        for r in top5:
            bit = r['bit']
            atom_idx, radius = bit_info[bit][0]
            other_prevs_named = {a: float(bit_prev_by_assay[a][bit]) for a in target_cols if a != case_assay}
            top_other = max(other_prevs_named, key=lambda k: other_prevs_named[k])
            env = Chem.FindAtomEnvironmentOfRadiusN(case_mol, radius, atom_idx)
            atom_set = set()
            for bidx in env:
                b = case_mol.GetBondWithIdx(bidx)
                atom_set.add(b.GetBeginAtomIdx())
                atom_set.add(b.GetEndAtomIdx())
            if not atom_set:
                atom_set = {atom_idx}
            try:
                frag = Chem.MolFragmentToSmiles(case_mol, atomsToUse=sorted(atom_set), bondsToUse=list(env))
            except Exception:
                frag = ''
            top_bits.append(str(bit))
            top_folds.append(f'{r["fold_change"]:.1f}')
            top_other_assays.append(top_other)
            top_smiles.append(frag)
        # Atoms across all enriched bits (not only top-5) for IG overlap
        for r in enriched_records:
            atom_idx, radius = bit_info[r['bit']][0]
            env = Chem.FindAtomEnvironmentOfRadiusN(case_mol, radius, atom_idx)
            for bidx in env:
                b = case_mol.GetBondWithIdx(bidx)
                atoms_in_enriched.add(b.GetBeginAtomIdx())
                atoms_in_enriched.add(b.GetEndAtomIdx())
            if not env:
                atoms_in_enriched.add(atom_idx)
        case_enriched_atoms[case_compid] = atoms_in_enriched

        i5c_rows.append({
            'case_compid': case_compid,
            'target_assay': case_assay,
            'y_true': case['y_true'],
            'advantage': case['advantage'],
            'n_on_bits': len(on_bits),
            'n_enriched_bits': n_enriched,
            'top_enriched_bits': '|'.join(top_bits),
            'top_enriched_fold_changes': '|'.join(top_folds),
            'top_enriched_other_assays': '|'.join(top_other_assays),
            'top_enriched_fragment_smiles': '|'.join(top_smiles),
        })

    pd.DataFrame(i5b_rows).to_csv(f'{out_dir}Table_I5b_case_study_fragment_sharing.csv', index=False)
    pd.DataFrame(i5c_rows).to_csv(f'{out_dir}Table_I5c_case_study_substructure_summary.csv', index=False)


    # ------------------------------------------------------------------
    # Table I5d: overlap between enriched-fragment atoms and top-K |IG| atoms
    # K = max(3, N_heavy / 4)
    # ------------------------------------------------------------------
    rows_d = []
    for _, case in case_df.iterrows():
        case_compid = case['COMPID']
        case_assay = case['Target']
        atom_score_path = f'{ig_dir}ig_{case_assay}_{case_compid}_atom_scores.csv'
        if not os.path.exists(atom_score_path):
            continue
        atom_scores = pd.read_csv(atom_score_path)
        n_heavy = len(atom_scores)
        k = max(3, n_heavy // 4)
        top_ig_atoms = set(
            atom_scores.reindex(atom_scores['ig_score_signed'].abs().sort_values(ascending=False).index)
            .head(k)['atom_idx'].astype(int).values
        )
        enriched_atoms = case_enriched_atoms.get(case_compid, set())
        if len(enriched_atoms) == 0:
            overlap = 0.0
        else:
            overlap = len(enriched_atoms & top_ig_atoms) / len(enriched_atoms)
        rows_d.append({
            'compid': case_compid,
            'target': case_assay,
            'n_enriched_atoms': len(enriched_atoms),
            'n_top_ig_atoms': len(top_ig_atoms),
            'overlap': overlap,
        })
    pd.DataFrame(rows_d).to_csv(f'{out_dir}Table_I5d_ig_overlap_with_enriched.csv', index=False)
