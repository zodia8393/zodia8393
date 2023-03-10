import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(train_df, test_df):
    # Fill missing values with mean
    imputer = SimpleImputer(strategy="mean")
    train_df_filled = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    test_df_filled = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

    # Standardize the data
    scaler = StandardScaler()
    train_df_scaled = scaler.fit_transform(train_df_filled.drop(columns=["row_id", "microbusiness_density"]))
    test_df_scaled = scaler.transform(test_df_filled.drop(columns=["row_id"]))

    # Reduce dimensionality using PCA
    pca = PCA(n_components=10)
    train_df_pca = pca.fit_transform(train_df_scaled)
    test_df_pca = pca.transform(test_df_scaled)

    return train_df_pca, test_df_pca
