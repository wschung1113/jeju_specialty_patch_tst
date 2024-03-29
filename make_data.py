import pandas as pd


df = pd.read_csv('/mnt/c/Users/wschu/OneDrive/Documents/data/jeju_specialty/open/train.csv')[['ID','timestamp', 'price(원/kg)']]
df['item_id'] = df.ID.str[0:6]
df.drop(columns="ID", inplace=True)
df.rename(columns={"price(원/kg)": "price"}, inplace=True)

pivot_df = df.pivot_table(index='timestamp', columns='item_id', values='price', aggfunc='first')
pivot_df.reset_index(inplace=True)

pivot_df.to_csv("./dataset/jeju_specialty.csv", index=False)