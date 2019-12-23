#!/usr/bin/env python
# coding: utf-8

def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics.scorer import make_scorer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    #get_ipython().run_line_magic('matplotlib', 'inline')
    import seaborn as sns
    import warnings
    from sklearn.metrics import classification_report,confusion_matrix
    warnings.filterwarnings("ignore", category=FutureWarning)
    from sklearn.model_selection import GridSearchCV

    #https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization#

    names = ["Signal1", "Signal2", "Signal3", "Signal4", "Signal5", "Signal6", "Signal7", "Room"]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00422/wifi_localization.txt"
    WIFI_data=pd.read_csv(url,delimiter='\t',encoding="latin-1",names=names)
    print(WIFI_data.head())

    #WIFI_data=pd.read_csv("wifi_localization.csv",encoding="latin-1")
    #WIFI_data.head()

    print(WIFI_data.count())
    print(len(WIFI_data))
    print(WIFI_data.describe())
    print(WIFI_data.isnull().sum())
    print(WIFI_data["Room"].unique())
    print(WIFI_data.dtypes)

    cols = WIFI_data.columns.tolist()
    cols.insert(0, cols.pop(cols.index("Room")))
    WIFI_data2 = WIFI_data[cols]
    WIFI_data2.head()

    corr_matrix = WIFI_data2.corr()
    sns.set(font_scale=1.2)
    plt.subplots(figsize=(8,8))
    heatmap = sns.heatmap(corr_matrix, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=corr_matrix.columns.values, xticklabels=corr_matrix.columns.values)
    plt.show()

    # room distribution
    fig = plt.figure(figsize=(10,1))
    plt.style.use('seaborn-ticks')
    sns.countplot(y="Room", data=WIFI_data2)

    X=WIFI_data2.iloc[:,1:]
    y=WIFI_data2.iloc[:,0]

    # we reduce features PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.99)
    X_reduced = pca.fit_transform(X)

    # train test splitting
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=5)

    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # test set should be scaled with the scaler trained on the training set.
    X_test = scaler.transform(X_test)

    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)

    variance_exp_cumsum = pca.explained_variance_ratio_.cumsum()
    fig, axes = plt.subplots(1,1,figsize=(15,7), dpi=100)
    plt.plot(variance_exp_cumsum, color='firebrick')
    plt.title('Variance Explained %', fontsize=22)
    plt.xlabel('# of PCs', fontsize=16)
    plt.show()

    print('Explained Variance Ratio')
    for i in range(7):
        print('PC{}: {}'.format(i+1,pca.explained_variance_ratio_[i]))

    # try ro find best k value
    scores = []
    for i in range(1,8):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train);
        scores.append(model.score(X_test,y_test))

    plt.plot(range(1,8), scores)
    plt.xticks(np.arange(1,8,1))
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.show()

    acc = max(scores)*100
    print("Maximum KNN Score is {:.2f}%".format(acc))
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.score(X_test,y_test))

    print("KNN confusion_matrix:\n",confusion_matrix(y_test,y_pred))

    from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
    import seaborn as sns

    neighbors = [1,2,3,4,5,6,7]
    cms = []
    acc = []

    for i in neighbors:
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train);
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print("k:",i, " mea:",mae)

    # Random forest
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=3, random_state=0,n_estimators=100)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Random Forest accuracy:", accuracy_score(y_test, y_pred))

    print("Random Forest confusion_matrix:\n",confusion_matrix(y_test,y_pred))

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [3,4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }

    rfc = RandomForestClassifier(random_state=0)
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y_train)

    print("Random Forest Best Parameters w/ grid search:", CV_rfc.best_params_)

    # random forest using best params from GridSearch
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=7, random_state=0,n_estimators=100 ,max_features='auto', criterion='entropy')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Random Forest w/ grid search:", accuracy_score(y_test, y_pred))

    print("Random Forest w/ grid search confusion_matrix:\n",confusion_matrix(y_test,y_pred))

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver="newton-cg")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred))

    print("Logitic Regression confusion_matrix:\n",confusion_matrix(y_test,y_pred))

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Naive Bayes accuracy:",accuracy_score(y_test, y_pred))

    print("Naive Bayes confusion_matrix:\n",confusion_matrix(y_test,y_pred))

    from sklearn.svm import SVC
    clf = SVC(gamma='auto')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("SVC accuracy:",accuracy_score(y_test, y_pred))

    print("SVC confusion_matrix:\n",confusion_matrix(y_test,y_pred))

    # Fit a decision tree
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Decision Tree accuracy:", accuracy_score(y_test, y_pred))

    print("Decision Tree confusion_matrix:\n", confusion_matrix(y_test,y_pred))

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1000)

    mlp.fit(X_train,y_train)

    y_pred = mlp.predict(X_test)
    print("Decision Tree accuracy:", accuracy_score(y_test, y_pred))

    conf_mx = confusion_matrix(y_test,y_pred)
    print("Decision Tree confusion_matrix:\n", conf_mx)

    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(n_estimators=100)

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Gradient Boosting accuracy:", accuracy_score(y_test, y_pred))
    print("Gradient Boosting confusion_matrix:\n", confusion_matrix(y_test,y_pred))

    ### cross validation
    from sklearn.model_selection import KFold #for K-fold cross validation
    from sklearn.model_selection import cross_val_score #score evaluation
    from sklearn.model_selection import cross_val_predict #prediction
    kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
    xyz=[]
    accuracy=[]
    std=[]

    classifiers=['KNN','Random Forest','Logistic Regression','Naive Bayes','SVC','Decision Tree','MLPClassifier','GradientBoostingClassifier']

    models=[KNeighborsClassifier(n_neighbors=5), RandomForestClassifier(max_depth=7, random_state=0,n_estimators=100 ,max_features='auto', criterion='entropy'),
            LogisticRegression(solver="newton-cg"),GaussianNB(),SVC(gamma='auto'),
            DecisionTreeClassifier(criterion='gini', max_depth=3),MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1000),GradientBoostingClassifier(n_estimators=100)]

    for i in models:
        model = i
        cv_result = cross_val_score(model,X_train, y_train, cv = kfold,scoring = "accuracy")
        cv_result=cv_result
        xyz.append(cv_result.mean())
        std.append(cv_result.std())
        accuracy.append(cv_result)
    models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)
    print(models_dataframe)

if __name__ == "__main__": main()
