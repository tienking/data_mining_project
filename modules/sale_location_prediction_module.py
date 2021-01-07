import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

class SaleLocationPrediction(object):
    def __init__(self):
        pass
    
    def get_dataset(self, url):
        self.df = pd.read_csv(url, index_col="Name")
    
    def clean(self):
        self.df.drop(columns=['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Global_Sales', 'Year_of_Release'], inplace=True)
        self.label = self.df[['NA_Sales' ,'EU_Sales', 'JP_Sales', 'Other_Sales']].idxmax(axis=1)
        self.df.drop(columns=['NA_Sales' ,'EU_Sales', 'JP_Sales', 'Other_Sales'], inplace=True)
        self.input_data = pd.get_dummies(self.df, dtype='float64', dummy_na=True, columns=list(self.df))
        self.column_names = list(self.input_data)
    
    def build_DecisionTree(self, p):
        X_train, X_test, y_train, y_test = train_test_split(self.input_data, self.label, shuffle=True, test_size=p, random_state=47)
        self.dt_clf = tree.DecisionTreeClassifier()
        self.dt_clf.fit(X_train, y_train)
        y_pred = self.dt_clf.predict(X_test)
        print("Accuracy: {0}".format(accuracy_score(y_test, y_pred)))
        dot_data = tree.export_graphviz(self.dt_clf,
                                feature_names=list(self.input_data), 
                                class_names=['NA_Sales','EU_Sales','JP_Sales','Other_Sales'], 
                                filled=True, 
                                rounded=True, 
                                special_characters=True)
        
    def build_BNB(self, p):
        X_train, X_test, y_train, y_test = train_test_split(self.input_data, self.label, shuffle=True, test_size=p, random_state=47)
        self.nb_clf = BernoulliNB()
        self.nb_clf.fit(X_train, y_train)
        y_pred = self.nb_clf.predict(X_test)
        print("Accuracy: {0}".format(accuracy_score(y_test, y_pred)))
        
    def predict(self, pre_url, model_name):
        pre_df = pd.read_csv(pre_url, index_col='Name')
        name_index = pre_df.index
        pre_df = pd.get_dummies(pre_df, dtype='float64', dummy_na=True, columns=list(pre_df))
        pre_input = pd.DataFrame(data=pre_df, columns=self.column_names)
        pre_input = pre_input.fillna(0)
        model_dir = {"DT":self.dt_clf, "BNB":self.nb_clf}
        model = model_dir[model_name]
        result = model.predict(pre_input)
        result_df = pd.DataFrame(index=name_index, columns=["Sale Location"])
        result_df["Sale Location"] = result
        print(result_df)
        result_df.to_csv("./result/" + model_name + "_sale_location_prediction_label.csv")