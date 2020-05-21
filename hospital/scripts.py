import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

pd.option_context('display.max_rows', None, 'display.max_columns', None)

data_type = 'test'
data_set = 1


def drop_data(data, columns):
    df = data.copy()[columns]
    if data_type == 'train':
        if data_set == 1:
            # df.drop(df.index[10000:], axis=0, inplace=True)
            pass
        elif data_set == 2:
            df.drop(df.index[40000:], axis=0, inplace=True)
            df.drop(df.index[:20000], axis=0, inplace=True)
        elif data_set == 3:
            df.drop(df.index[40000:], axis=0, inplace=True)
            df.drop(df.index[:20000], axis=0, inplace=True)
        elif data_set == 3:
            df.drop(df.index[40000:], axis=0, inplace=True)
            df.drop(df.index[:20000], axis=0, inplace=True)
    return df


def make_data_types_file(data, columns):
    print("Make data types file")
    types = pd.read_csv(f'hospital/WiDS Datathon 2020 Dictionary.csv')
    types = types[['Variable Name', 'Data Type', 'Example']]
    types.set_index('Variable Name', inplace=True)
    types = types.transpose()
    types = types[columns]
    types = types.transpose()
    i = 0
    lengths = []
    has_negs = []
    for column in columns:
        vals = data[column].dropna().values
        if data[column].dtype == object:
            iis = list(range(1, len(set(vals)) + 1))
            key_getter = dict(zip(set(vals), iis))
            key_getter[np.nan] = np.nan
            data[column] = [float(key_getter[l]) for l in data[column].values]
            i += 1
        lengths.append(len(set(vals)) + 1)
        try:
            has_negs.append(any(pd.to_numeric(vals) < 0))
        except ValueError:
            has_negs.append(False)

    types['dim'] = lengths
    types['nclass'] = lengths
    types['neg'] = has_negs
    types.columns = ['type', 'ex', 'dim', 'nclass', 'neg']
    old = types
    new_types_data = [
        ['real' if row[4] else 'pos', 1, ''] if (row[0] == 'numeric' or (row[0] == 'string' and get_float(row[1])))
        else
        (['cat', 2, 2] if row[0] == 'binary'
         else
         (['cat', row[2], row[2]]))
        for row in types.values
    ]

    types_df = pd.DataFrame(new_types_data, columns=['type', 'dim', 'nclass'])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(old.shape)
    #     print(types_df.shape)
    #     print(pd.concat([old, types_df], axis=1))

    types_df.to_csv(f'hospital/data_types.csv', index=False)


def generate_missing_file(df):
    if data_type == 'test':
        df['hospital_death'] = np.nan
    x, y = sp.coo_matrix(df.isnull()).nonzero()
    if data_type == 'test':
        f = open(f"hospital/Missing100_1.csv", "w")
    else:
        f = open(f"hospital/Missing33_1.csv", "w")
    for x, y in list(zip(x, y)):
        f.write(f"{x + 1},{y + 1}\n")
    f.close()


def get_float(string: str):
    try:
        np.float64(string)
        print(string)
        return True
    except ValueError:
        return False


def change_types(df):
    i = 0
    lengths = []
    for column in df.columns:
        vals = df[column].dropna().values
        if df[column].dtype == object:
            iis = list(range(1, len(set(vals)) + 1))
            key_getter = dict(zip(set(vals), iis))
            key_getter[np.nan] = np.nan
            df[column] = [key_getter[l] for l in df[column].values]
            i += 1
        lengths.append(len(set(vals)) + 1)

    df = df.fillna(0)
    df.to_csv(f'hospital/data_{data_type}.csv', header=False, index=False)


def generate_files():
    if data_type == 'train':
        data = pd.read_csv(f'hospital/org_train_data.csv')
    else:
        data = pd.read_csv(f'hospital/org_test_data.csv')

    identifier = list(data.columns[:3])  # 1
    demographic = list(data.columns[3:18])  # 2
    apache_covariate = list(data.columns[18:46])  # 3
    vitals = list(data.columns[46:98])  # 4
    labs = list(data.columns[98:158])  # 5
    labs2 = list(data.columns[158:174])  # 6
    apache_prediction = list(data.columns[174:176])  # 7
    apache_comorbidity = list(data.columns[176:184])  # 8
    apache_grouping = list(data.columns[184:186])  # 9

    # Change for different sets of variables
    columns = demographic + apache_comorbidity + vitals

    m = np.sum(data[columns].isnull().sum().tolist())
    s = data[columns].shape
    t = s[0] * s[1]
    print(m / t)
    if data_type == 'train':
        make_data_types_file(data, columns)
        # return 0

    print(f"Make {data_type} files")
    df = drop_data(data, columns)
    generate_missing_file(df)
    change_types(df)


if __name__ == '__main__':
    data_type = sys.argv[1]
    if len(sys.argv) > 2:
        data_set = int(sys.argv[2])
    generate_files()
