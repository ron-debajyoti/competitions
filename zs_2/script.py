import pandas as pd, numpy as np, copy
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, preprocessing
from sklearn.linear_model import SGDClassifier
import pickle,os

import matplotlib.pyplot as plt
import json

# setting the module specific env variables
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
scaler = preprocessing.MaxAbsScaler()


def Forest(data, features):
    x_train, x_test, y_train, y_test = train_test_split(data[features['Specs'].tolist()], data['is_goal'],
                                                        random_state=42, test_size=0.20)
    x_train = preprocessing.scale(x_train)
    x_train = scaler.fit_transform(x_train)
    x_test = preprocessing.scale(x_test)
    x_test = scaler.fit_transform(x_test)

    parameters = {'bootstrap': True,
                  'min_samples_leaf': 10,
                  'n_estimators': 50,
                  'min_samples_split': 30,
                  'max_features': 'sqrt',
                  'max_depth': 50}
    model = RandomForestClassifier(**parameters)
    model.fit(x_train, y_train)
    print(metrics.accuracy_score(y_test, model.predict(x_test)))
    return model


def SVM_model(data, top_ten_features):
    x_train, x_test, y_train, y_test = train_test_split(data[top_ten_features['Specs'].tolist()], data['is_goal'],
                                                        random_state=42, test_size=0.20)

    text_clf_svm = Pipeline([('clf', SGDClassifier(loss='log', penalty='l2', alpha=1.5e-4, max_iter=2000, tol=1e-5,
                                                   random_state=4)), ])
    text_clf_svm = text_clf_svm.fit(x_train, y_train)
    parameters = {'clf__alpha': (1e-3, 1e-4)}
    gs_clf_svm = GridSearchCV(text_clf_svm, parameters, n_jobs=-1)
    gs_clf_svm = gs_clf_svm.fit(x_train, y_train)
    print(gs_clf_svm.best_params_)

    predicted = gs_clf_svm.predict(x_test)
    print(np.mean(predicted == y_test))


def feature_selection(data):
    ''' using SelectKBest to get the best variables dependent on the target variable '''
    columns = data.columns
    target_column = ['is_goal']
    filler_column = ['team_name', 'team_id', 'match_id']
    X = data[[x for x in columns if x not in (filler_column and target_column)]]
    y = data[target_column]
    bestfeatures = SelectKBest(score_func=f_classif, k=5)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_10 = featureScores.nlargest(10, 'Score')
    top_10 = top_10.reset_index(drop=True)
    print(top_10)
    return top_10


def pre_processing():
    dataframe = pd.read_csv('./data.csv')
    dataset = copy.deepcopy(dataframe)

    ''' cleaning the data '''
    dataset = dataset.drop(
        ['remaining_min.1', 'power_of_shot.1', 'knockout_match.1', 'remaining_sec.1', 'distance_of_shot.1'],
        axis=1)

    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            dataset[column] = dataset[column].fillna('0')
        elif column == 'is_goal':
            pass
        else:
            dataset[column] = dataset[column].fillna(0)

        if dataset[column].dtype == type(object) or column == ('location_x' or 'location_y'):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])

    if not os.path.isfile('./test_dataset.pkl'):
        test_dataset = pd.DataFrame(columns=dataset.columns)
        ''' Separating the testing and the training data '''
        for index, row in dataset.iterrows():
            if pd.isna(row['is_goal']):
                test_dataset = test_dataset.append(row)
                dataset.loc[index]['shot_id_number']= index+1
                print("$%#$%%%%%"+ str(index) + "      " +  str(dataset.loc[index]['shot_id_number']))
        pickle_out = open("test_dataset.pkl","wb")
        pickle.dump(test_dataset,pickle_out)
        pickle_out.close()
    else:
        file = open("test_dataset.pkl","rb")
        test_dataset = pickle.load(file)
        file.close()

    dataset = dataset.dropna()
    print(dataset[dataset.isna().any(axis=1)])
    top_ten_features = feature_selection(dataset)
    return top_ten_features,dataset,test_dataset



def prediction(model,test_data,features):
    predicted_output = model.predict_proba(scaler.fit_transform(preprocessing.scale(
            test_data[features['Specs'].tolist()])))[:,0]

    output = pd.DataFrame({'shot_id_number':test_data['shot_id_number'],
                           'is_goal':predicted_output.tolist()})
    print(output)
    output.to_csv('submission.csv',sep=',')



if __name__ == '__main__':
    top_ten_features,dataset,test_dataset = pre_processing()
    model = Forest(dataset, top_ten_features)
    prediction(model, test_dataset, top_ten_features)
