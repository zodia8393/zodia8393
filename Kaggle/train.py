import argparse
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from preprocessing import load_data, handle_missing_values, encode_categorical_features, scale_features

def train(args):
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    train_data = handle_missing_values(train_data, args.missing_value_cols, args.impute_strategy)
    test_data = handle_missing_values(test_data, args.missing_value_cols, args.impute_strategy)
    train_data = encode_categorical_features(train_data, args.cat_cols, args.encoder_type)
    test_data = encode_categorical_features(test_data, args.cat_cols, args.encoder_type)
    train_data = scale_features(train_data, args.scaler_type, args.scaler_cols)
    test_data = scale_features(test_data, args.scaler_type, args.scaler_cols)
    X_train = train_data.drop([args.target_col], axis=1)
    y_train = train_data[args.target_col]
    X_test = test_data.drop([args.target_col], axis=1)
    xgb_model = xgb.XGBRegressor(**args.xgb_params)
    xgb_model.fit(X_train, y_train)
    lgb_model = lgb.LGBMRegressor(**args.lgb_params)
    lgb_model.fit(X_train, y_train)
    rf_model = RandomForestRegressor(**args.rf_params)
    rf_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    lgb_preds = lgb_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    preds = (args.xgb_weight * xgb_preds) + (args.lgb_weight * lgb_preds) + (args.rf_weight * rf_preds)
    return preds

def write_submission_file(submission_file_path, preds):
    submission_df = pd.read_csv(submission_file_path)
    submission_df['microbusiness_density'] = preds
    submission_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--missing_value_cols', type=str, nargs='+', default=[])
    parser.add_argument('--impute_strategy', type=str, default='mean')
    parser.add_argument('--cat_cols', type=str, nargs='+', default=[])
    parser.add_argument('--encoder_type', type=str, default='onehot')
    parser.add_argument('--scaler_type', type=str, default='standard')
    parser.add_argument('--scaler_cols', type=str, nargs='+', default=[])
    parser.add_argument('--target_col', type=str, required=True)
    parser.add_argument('--xgb_params', type=dict, default={'objective': 'reg:squarederror', 'random_state': 42})
    parser.add_argument('--lgb_params', type=dict, default={'objective': 'regression', 'random_state': 42})
    parser.add_argument('--rf_params', type=dict, default={'random_state': 42})
    parser.add_argument('--xgb_weight', type=float, default=0.5)
    parser.add_argument('--lgb_weight', type=float, default=0.25)
    parser.add_argument('--rf_weight', type=float, default=0.25)
    parser.add_argument('--submission_file_path', type=str, required=True
    args = parser.parse_args()
    preds = train(args)
    write_submission_file(args.submission_file_path, preds)

    
