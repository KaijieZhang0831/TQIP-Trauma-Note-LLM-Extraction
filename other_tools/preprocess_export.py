import pandas as pd
import numpy as np
import pdb
df = pd.read_csv('cases.csv', dtype={
    'Medical Record Number': str,
    'Alias MRN': str, 
    'CSN': str
})
grouped_df = df.groupby('REGISTRYNUM').agg(list).reset_index()

exclude_values = [np.nan, '*NA', '*BL']

# Filter rows that do not have NaN, '*NA', or '*BL' in the list
filtered_df = grouped_df[grouped_df['Complications'].apply(lambda x: any(val not in exclude_values for val in x))]
filtered_df['comp'] = filtered_df['Complications'].apply(
    lambda x: '-'.join([str(val) for val in x if val not in exclude_values])
)
filtered_df['mrn'] = filtered_df['Medical Record Number'].apply(
    lambda x: '-'.join([str(val) for val in x if val not in exclude_values])
)
filtered_df['alias_mrn'] = filtered_df['Alias MRN'].apply(
    lambda x: '-'.join([str(val) for val in x if val not in exclude_values])
)
filtered_df['csn'] = filtered_df['CSN'].apply(
    lambda x: '-'.join([str(val) for val in x if val not in exclude_values])
)
filtered_df = filtered_df[['mrn', 'alias_mrn', 'csn', 'comp']]
pdb.set_trace()
# filtered_df.to_csv('ground_truth.csv', values=False)