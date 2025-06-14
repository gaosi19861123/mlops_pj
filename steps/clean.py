import numpy as np
from sklearn.impute import SimpleImputer

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        
    def clean_data(self, data):
        # データのコピーを作成
        data = data.copy()
        
        # ターゲット変数の欠損値を確認
        if 'Switch' in data.columns:
            print("ターゲット変数の欠損値の数:", data['Switch'].isnull().sum())
            # ターゲット変数に欠損値がある行を削除
            data = data.dropna(subset=['Switch'])
        
        # 不要な列を削除
        columns_to_drop = ['id', 'SalesChannelID', 'VehicleAge', 'DaysSinceCreated']
        data = data.drop(columns=columns_to_drop)
        
        # AnnualPremiumの処理
        data.loc[:, 'AnnualPremium'] = data['AnnualPremium'].str.replace('£', '').str.replace(',', '').astype(float)
            
        # カテゴリカル変数の欠損値処理
        for col in ['Gender', 'RegionID']:
            data.loc[:, col] = self.imputer.fit_transform(data[[col]]).flatten()
             
        # 数値変数の欠損値処理
        data.loc[:, 'Age'] = data['Age'].fillna(data['Age'].median())
        data.loc[:, 'HasDrivingLicense'] = data['HasDrivingLicense'].fillna(1)
        data.loc[:, 'PastAccident'] = data['PastAccident'].fillna("Unknown")
        
        # 外れ値の処理
        Q1 = data['AnnualPremium'].quantile(0.25)
        Q3 = data['AnnualPremium'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        data = data[data['AnnualPremium'] <= upper_bound]
        
        return data