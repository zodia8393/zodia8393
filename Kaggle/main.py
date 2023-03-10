import pandas as pd
from sklearn.preprocessing import StandardScaler
from lstm_model import create_lstm_model
from cnn_model import create_cnn_model
from transformer_model import create_transformer_model
from ensemble_model import create_ensemble_model

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Preprocess data
train_df = train_df.dropna()
train_df["first_day_of_month"] = pd.to_datetime(train_df["first_day_of_month"])
train_df = train_df.set_index("first_day_of_month")
train_df = train_df[["microbusiness_density"]]
scaler = StandardScaler()
train_df_scaled = scaler.fit_transform(train_df)

# Split data into training and validation sets
train_size = int(len(train_df_scaled) * 0.7)
train_data, val_data = train_df_scaled[:train_size], train_df_scaled[train_size:]

# Create models
lstm_model = create_lstm_model(input_shape=(train_data.shape[1], 1))
cnn_model = create_cnn_model(input_shape=(train_data.shape[1], 1))
transformer_model = create_transformer_model(input_shape=(train_data.shape[1],))
ensemble_model = create_ensemble_model(lstm_model, cnn_model, input_shape=(train_data.shape[1], 1))

# Train models
lstm_model.fit(train_data.reshape(-1, train_data.shape[1], 1), epochs=50, batch_size=64, validation_data=(val_data.reshape(-1, val_data.shape[1], 1),))
cnn_model.fit(train_data.reshape(-1, train_data.shape[1], 1), epochs=50, batch_size=64, validation_data=(val_data.reshape(-1, val_data.shape[1], 1),))
transformer_model.fit(train_data, epochs=50, batch_size=64, validation_data=(val_data,))
ensemble_model.fit([train_data.reshape(-1, train_data.shape[1], 1), train_data.reshape(-1, train_data.shape[1], 1)], epochs=50, batch_size=64, validation_data=([val_data.reshape(-1, val_data.shape[1], 1), val_data.reshape(-1, val_data.shape[1], 1)],))

# Make predictions on the test set using the ensemble model
test_df["first_day_of_month"] = pd.to_datetime(test_df["first_day_of_month"])
test_df = test_df.set_index("first_day_of_month")
test_df = test_df.drop("cfips", axis=1)
test_df_scaled = scaler.transform(test_df)
y_pred = ensemble_model.predict([test_df_scaled.reshape(-1, test_df_scaled.shape[1], 1), test_df_scaled.reshape(-1, test_df_scaled.shape[1], 1)])

# Create submission file
submission_df = pd.DataFrame({"row_id": test_df.index.strftime("%Y-%m-%d") + "_" + test_df.index.month.astype(str) + "-01"})
submission_df["microbusiness_density"] = y_pred
submission_df.to_csv("submission.csv", index=False)
