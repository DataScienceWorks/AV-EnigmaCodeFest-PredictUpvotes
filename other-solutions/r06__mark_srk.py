import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, model_selection, linear_model
import lightgbm as lgb

if __name__ == "__main__":
	train_df = pd.read_csv("../input/train_NIR5Yl1.csv")
	test_df = pd.read_csv("../input/test_8i3B3FC.csv")
	print(train_df.shape, test_df.shape)

	train_id = train_df["ID"].values
	test_id = test_df["ID"].values
	train_upvotes = (train_df["Upvotes"].values).copy()
	train_y = (train_df["Upvotes"].values) #/ np.maximum(train_df["Views"].astype('float'),1.)

	train_df["mult_repu_views"] = train_df["Views"] * (train_df["Reputation"]/1000000.)
	test_df["mult_repu_views"] = test_df["Views"] * (test_df["Reputation"]/1000000.)

	model = linear_model.Ridge(fit_intercept=False)
	model.fit(train_df[["mult_repu_views"]], train_y)
	
	pred_train = model.predict(train_df[["mult_repu_views"]])
	pred_test_full = model.predict(test_df[["mult_repu_views"]])
	print("CV score is : ", np.sqrt(metrics.mean_squared_error(train_upvotes, pred_train)))
	print(model.coef_)
	print(model.intercept_)

	sub_df = pd.DataFrame({"ID":test_id})
	sub_df["Upvotes"] = pred_test_full
	sub_df.to_csv("test_ridge_preds.csv", index=False)

	sub_df = pd.DataFrame({"ID":train_id})
	sub_df["Upvotes"] = pred_train
	sub_df.to_csv("train_ridge_preds.csv", index=False)
