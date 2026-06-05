from models.decision_tree.variants.simple_decision_tree import DecisionTree
from utils.data_loader import load_classification_data_split_transformed_for_DT
import numpy as np


X_train ,X_test , y_train , y_test = load_classification_data_split_transformed_for_DT("datasets/classification/raw/Social_Network_Ads.csv", 0  , 2 ,change_y= False, )
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
model = DecisionTree()

model.fit(X_train, y_train)
model.predict_batch(X_test)

