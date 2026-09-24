#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file_path = "Restaurant.csv"

# read data into dataframe
df = pd.read_csv(file_path)

# drop unnamed columns
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

print(df.head())


# In[3]:


# standardise column names: strip spaces, lowercase, replace space/hypgen with underscores
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(' ', '_')
              .str.replace('-', '_'))
print(df.info())


# In[4]:


# % of rows where a value is missing
missing_rows_pct = df.isna().any(axis=1).mean() * 100
print(f"Percentage of rows with missing values: {missing_rows_pct:.2f}%")

# columns with missing values
missing_cols = df.columns[df.isna().any()].tolist()
print(f"Columns with missing value: {missing_cols}")


# In[5]:


# handling typos in values

import difflib

VALID_GENDER = ['male', 'female']
VALID_SMOKER = ['yes', 'no']
# per the assignment's data dictionary, Day only ever takes these 4 values
VALID_DAY = ['thursday', 'friday', 'saturday', 'sunday']
VALID_TIME = ['lunch', 'dinner']

def auto_correct(val, valid_list, cutoff=0.2):
  if pd.isna(val):
    return np.nan

  clean_val = str(val).strip().lower()

  if clean_val in valid_list:
    return clean_val

  # handle repeated character or single letter abbreviations (eg 'd', 'ddd', 'l', 'sss')
  unique_chars = set(clean_val)
  if len(unique_chars) == 1:
    char = list(unique_chars)[0]
    # find all valid targets starting with this letter
    prefix_matches = [v for v in valid_list if v.startswith(char)]

    # return only if unambiguous
    if len(prefix_matches) == 1:
      return prefix_matches[0]

    else:
      return np.nan   # return nan for 's' or 't' collisions so imputation handles it

  # fallback to difflib fuzzy matching for normal typos
  matches = difflib.get_close_matches(str(clean_val), valid_list, n=1, cutoff=cutoff)
  return matches[0] if matches else np.nan

df['gender'] = df['gender'].apply(auto_correct, valid_list=VALID_GENDER)

df['smoker'] = df['smoker'].apply(auto_correct, valid_list=VALID_SMOKER, cutoff=0.1)

# for day column, ambiguous values such as 's', 'ss', 'sss' will be replaced by np.nan for later imputation
df['day'] = df['day'].apply(auto_correct, valid_list=VALID_DAY, cutoff=0.3)

# for time column, ambiguous values such as 'LD' will be replaced by np.nan for later imputation
df['time'] = df['time'].apply(auto_correct, valid_list=VALID_TIME, cutoff=0.3)


# In[6]:


# handling typos such as ` and $, plus comma/hyphen used as a decimal point
# (hyphen is only treated as a decimal point when it sits BETWEEN two digits,
# e.g. '25-89' -> '25.89', so a genuine negative value like '-7.78' is untouched)
# note: not gated on dtype == 'object' -- pandas >=2.x/3.x can give these columns
# dtype 'str' (StringDtype) straight out of read_csv, so that check would silently
# skip this whole cleanup; .astype(str) + these replaces are idempotent on already-clean values

for col in ['amount', 'tip', 'partysize']:
  clean_s = df[col].astype(str).str.replace(r'[`$]', '', regex=True).str.strip()
  clean_s = clean_s.str.replace(",", ".", regex=False)
  clean_s = clean_s.str.replace(r'(\d)-(\d)', r'\1.\2', regex=True)
  df[col] = clean_s


# In[7]:


# converting column dtypes

df['tip'] = pd.to_numeric(df['tip'], errors='coerce')
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['partysize'] = pd.to_numeric(df['partysize'], errors='coerce')
df['gender'] = df['gender'].astype('category')
df['smoker'] = df['smoker'].map({"yes":True, "no":False})
df['day'] = df['day'].astype('category')
df['time'] = df['time'].astype('category')


# In[8]:


# drop cases where tip > amount

# number of rows before dropping
num_rows = len(df)

# filter out rows where tip > amount
df = df[~(df['tip'] > df['amount'])].copy()

# print verification
dropped_rows = num_rows - len(df)
print(f"Dropped {dropped_rows} row(s) where tip > amount.")
print(f"Remaining dataset rows: {len(df)}")


# In[9]:


# contextual replacement of invalid amount, tip, partysize, or time using KNN Imputation

from sklearn.impute import KNNImputer

cols = ['amount', 'tip', 'partysize', 'time', 'smoker']

# flag invalid values as nan
df.loc[df['amount'] <= 0, 'amount'] = np.nan
df.loc[df['tip'] < 0, 'tip'] = np.nan
df.loc[df['partysize'] < 1, 'partysize'] = np.nan

# mapping time and smoker for imputation
df['time'] = df['time'].map({'lunch':0, 'dinner':1})
df['smoker'] = df['smoker'].map({True:1, False:0})

# impute missing values

# intialise and impute
imputer = KNNImputer(n_neighbors=5)
df[cols] = imputer.fit_transform(df[cols])

# decode time and smoker back
df['time'] = np.where(df['time']>0.5, 'dinner', 'lunch')
df['smoker'] = np.where(df['smoker']>0.5, True, False)


# format and round values
df['partysize'] = df['partysize'].round().astype('Int64')
df[['amount', 'tip']] = df[['amount', 'tip']].round(2).astype('Float64')


# In[10]:


# contextual imputation for day (stratified mode by time)
# flag which rows had day imputed, so downstream tests that involve 'day'
# (HT5, HT6) can run on complete cases only, avoiding circularity from
# imputing day using time and then testing day against time
df['day_imputed'] = df['day'].isna()

df['day'] = df.groupby('time')['day'].transform(
    lambda x: x.fillna(x.mode()[0])
)


# In[11]:


# % of rows where a value is missing
missing_rows_pct = df.isna().any(axis=1).mean() * 100
print(f"Percentage of rows with missing values: {missing_rows_pct:.2f}%")

# columns with missing values
missing_cols = df.columns[df.isna().any()].tolist()
print(f"Columns with missing value: {missing_cols}")


# In[12]:


# cap tip and partysize at their espective 99th percentiles

upper_bound_tip = df['tip'].quantile(0.99).round(3)
upper_bound_partysize = int(np.round(df['partysize'].quantile(0.99)))

df['tip'] = df['tip'].clip(upper=upper_bound_tip)

df['partysize'] = df['partysize'].clip(upper=upper_bound_partysize)

df['partysize'] = df['partysize'].round().astype('Int64')

df['tip'] = df['tip'].round(2).astype('Float64')

print(f"Capped 'tip' at 99th percentile: {upper_bound_tip}")
print(f"Capped 'partysize' at 99th percentile: {upper_bound_partysize}")


# In[13]:


output_path = "Restaurant_Cleaned.csv"
df.to_csv(output_path, index=False)

print(f"Dataset successfully saved to: {output_path}")


# In[ ]:




