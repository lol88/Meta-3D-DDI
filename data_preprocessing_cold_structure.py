
# %%
import os
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, cut_tree
from matplotlib import pyplot as plt
from collections import Counter

# %%
root = '/data0/linan/data/SA-Graphformer/data/preprocessed/drugbank/'
data_path = os.path.join(root, 'pair_pos_neg_triplets.csv')
df = pd.read_csv(data_path, index_col=False)
new_dict = defaultdict(list)
for i, (drug1_id, drug2_id, Y, neg_sample) in df.iterrows():
    neg_id, Ntype = neg_sample.split('$')
    new_dict['pos_pair_h'].append(drug1_id)
    new_dict['pos_pair_t'].append(drug2_id)
    
    if Ntype == 'h':
        new_dict['neg_pair_h'].append(neg_id)
        new_dict['neg_pair_t'].append(drug2_id)
    else:
        new_dict['neg_pair_h'].append(drug1_id)
        new_dict['neg_pair_t'].append(neg_id)  

    new_dict['Y'].append(Y)

new_df = pd.DataFrame(new_dict)
# print(new_df)
# %%
smiles_root = '/data0/linan/data/SA-Graphformer/data/drugbank.tab'
smiles_df = pd.read_csv(smiles_root, delimiter='\t')
smiles_dict = {}
for i, row in smiles_df.iterrows():
    smiles_dict[row['ID1']] = row['X1']
    smiles_dict[row['ID2']] = row['X2']

# %%
def search_index(df, keyvals, key):
    if len(keyvals) < 2:
        return pd.DataFrame(), pd.DataFrame()

    smiles = []
    idx_id_map = {}
    for i, kv in enumerate(keyvals):
        # df_single = df[df[key] == kv]
        # df_list.append(df_single)
        smi = smiles_dict[kv]
        smiles.append(smi)
        idx_id_map[i] = kv

    vec_list = []
    for smi in smiles:
        m1 = Chem.MolFromSmiles(smi)
        fp4 = list(AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=256))
        vec_list.append(fp4)

    Z = linkage(vec_list, 'average', metric='jaccard')
    cluster_num = round(len(vec_list) * 0.5)
    cluster = cut_tree(Z, cluster_num).ravel()
    # plt.hist(cluster)
    stat_dict = {k: v for k, v in sorted(Counter(cluster).items(), key=lambda item: item[1], reverse=True)}
    data_dict = defaultdict(list)
    for i in stat_dict.keys():
        pos = np.nonzero(cluster==i)[0]
        if len(data_dict['train']) < round(len(smiles) * 0.75):
            data_dict[f'train'] += list(pos)
        else:
            data_dict[f'test'] += list(pos)

    df_list = []
    for idx in data_dict['train']:
        kv = idx_id_map[idx]
        df_single = df[df[key] == kv]
        df_list.append(df_single)
    if len(df_list) == 0:
        return pd.DataFrame(), pd.DataFrame()
    else:
        df_train = pd.concat(df_list)

    df_list = []
    for idx in data_dict['test']:
        kv = idx_id_map[idx]
        df_single = df[df[key] == kv]
        df_list.append(df_single)
    if len(df_list) == 0:
        return pd.DataFrame(), pd.DataFrame()
    else:
        df_test = pd.concat(df_list)



    # print(len(df_train))
    # print(len(df_test))

    return df_train, df_test

count = 0
df_train_list = []
df_test_list = []
for y in new_df['Y'].unique():
    df_sub = new_df[new_df['Y'] == y]
    unique_pos_pair_t_id = df_sub['pos_pair_t'].unique()
    df_train, df_test = search_index(df_sub, keyvals=unique_pos_pair_t_id, key='pos_pair_t')
    df_train_list.append(df_train)
    df_test_list.append(df_test)

df_train_merge = pd.concat(df_train_list)
df_test_merge = pd.concat(df_test_list)

print(count)

print(len(df_train_merge))
print(len(df_test_merge))

# %%
df_train_merge.to_csv(os.path.join(root, 'train_new_old_structure.csv'), index=False)
df_test_merge.to_csv(os.path.join(root, 'test_new_old_structure.csv'), index=False)


# %%
