import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class WeatherPrediction(object):
    def __init__(self):
        pass
      
    def get_dataset(self, url):
        self.df = pd.read_csv(url, index_col='Date')
        self.df.head(10)
        
    def clean(self):
        self.df.dropna(subset = ['RainTomorrow'], inplace=True)
        self.label = self.df[['RainTomorrow']].copy()
        input_data = self.df.iloc[:, :-1]
        numeric_label_dir = {"No":0, "Yes":1}
        self.label.replace(numeric_label_dir, inplace=True)
        num_data = pd.get_dummies(input_data, dtype='float64', dummy_na=True)
        num_data['Label'] = self.label.RainTomorrow.copy()
        filled_data = num_data.groupby(["Label"]).transform(lambda x: x.fillna(x.mean()))
        minmax_scaler = preprocessing.MinMaxScaler()
        scaled_arr = minmax_scaler.fit_transform(filled_data)
        self.cleaned_data = pd.DataFrame(scaled_arr)
        print("Data cleaned")
        self.column_names = list(self.cleaned_data)
            
    def build_KNN(self, k, percent):
        X_train, X_test, y_train, y_test = train_test_split(self.cleaned_data, 
                                                            np.array(self.label).ravel(), 
                                                            shuffle=True, 
                                                            test_size=percent, 
                                                            random_state=42)
        self.knn_cf = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=2)
        self.knn_cf.fit(X_train, y_train)
        #y_pred = self.knn_cf.predict(X_test)
        #acc = accuracy_score(y_test, y_pred)
        #print("Accuracy: {0}".format(acc))
    
    def predict(self, pre_url):
        pre_df = pd.read_csv(pre_url, index_col='Date')
        date_index = pre_df.index
        pre_df = pd.get_dummies(pre_df, dtype='float64')
        pre_df = pre_df.fillna(0.0)
        pre_df = pre_df.apply(lambda x: x.astype(np.float64))
        minmax_scaler = preprocessing.MinMaxScaler()
        scaled_arr = minmax_scaler.fit_transform(pre_df)
        pre_df = pd.DataFrame(scaled_arr)
        pre_input = pd.DataFrame(data=pre_df, columns=self.column_names)
        pre_input = pre_input.fillna(0)
        result = self.knn_cf.predict(pre_input)
        result_df = pd.DataFrame(index=date_index, columns=["RainTomorrow"])
        result_df['RainTomorrow'] = result
        result_dir = {1:"Yes", 0:"No"}
        result_df['RainTomorrow'].replace(result_dir, inplace=True)
        print(result_df)
        result_df.to_csv("./result/weather_prediction_label.csv")