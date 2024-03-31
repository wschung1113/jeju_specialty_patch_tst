import pandas as pd


def get_jeju_data():
    df = pd.read_csv('/mnt/c/Users/wschu/OneDrive/Documents/data/jeju_specialty/open/train.csv')[['ID','timestamp', 'price(원/kg)']]
    
    return df

    
def pivot_jeju_data(df):
    df['item_id'] = df.ID.str[0:6]
    df.drop(columns="ID", inplace=True)
    df.rename(columns={"price(원/kg)": "price"}, inplace=True)

    pivot_df = df.pivot_table(index='timestamp', columns='item_id', values='price', aggfunc='first')
    pivot_df.reset_index(inplace=True)
    
    return pivot_df


if __name__ == '__main__':
    df = get_jeju_data()
    pivot_df = pivot_jeju_data(df)

    pivot_df.to_csv("./dataset/jeju_specialty.csv", index=False)