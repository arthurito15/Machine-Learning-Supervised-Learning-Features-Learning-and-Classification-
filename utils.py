import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import random
import pickle
import shap
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, make_scorer, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# Q2 fonction d'evaluation
def train_test(clf, X, y, X_test, y_test, label=None):
    # On partage les données en deux ensembles : un pour l'apprentissage
    # et un pour les tests
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=1)
    clf_fitted = deepcopy(clf).fit(X, y)
    pred = clf_fitted.predict(X_test)
    clf_confusion = confusion_matrix(pred, y_test)
    # ConfusionMatrixDisplay(clf_confusion).plot()
    # print(clf_confusion)
    
    fpr, tpr, threshold = roc_curve(pred, y_test)
    plt.plot(fpr, tpr, label=label)
    plt.legend()
    plt.title("ROC")

    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(y_test, pred)

    # On prefere de la securite pour attribuer un credit, il faut regarder 
    # l'accuracy et la précision 
    precis = precision_score(y_test, pred)
    
    return { "clf": clf_fitted,
             "score": (accuracy + precis) / 2,
             "confusion_matrix": clf_confusion,
             "fpr": fpr,
             "tpr": tpr,
             "auc": auc,
             "X_train": X,
             "X_test": X_test,
             "y_train": y,
             "y_test": y_test }


# Q2 fonction d’apprentissage et de test 
def run_classifiers_train_test(clfs, train_test, X, y):
    # On partage les données en deux ensembles : un pour l'apprentissage
    # et un pour les tests

    plt.clf()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=1)
    results = [(clf[0], train_test(clf[1], X_train, y_train, X_test, y_test, label=clf[0])) for clf in clfs.items()]
    results_sorted = sorted(results, key=lambda x: x[1]["score"])
    for clf, results in results_sorted:
        print(f"Score for {clf} is:\t{results['score']:.3f}")

    return results_sorted


# Q3 normalisation avec StandardScaler
def normalize_X(X):
    X_norm = pd.DataFrame(data=StandardScaler().fit_transform(X), columns=X.columns)
    return X_norm


# Q3 fonction d'evaluation
def norm_train_test(clf, X, y, X_test, y_test, label=None):
    X_norm = normalize_X(X)
    X_test_norm = normalize_X(X_test)
    return train_test(clf, X_norm, y, X_test_norm, y_test,label=label)


# Q4 Creation de 3 nouvelles variables à partir des données normalisées
def normalize_xt_X(X):
    X_norm = normalize_X(X)
    X_n_xt = X.copy(deep=True)
    X_n_xt[['pca1', 'pca2', 'pca3']] = PCA(n_components=3).fit_transform(X_norm)
    return X_n_xt


# Q4 fonction d'evaluation
def xt_train_test(clf, X, y, X_test, y_test, label=None):
    X_n_xt = normalize_xt_X(X)
    X_test_n_xt = normalize_xt_X(X_test)
    return train_test(clf, X_n_xt, y, X_test_n_xt, y_test, label=label)

# Q5
def variable_importance(clf, X, y, X_test, y_test):
    RFC= RandomForestClassifier(n_estimators=1000,random_state=1)
    RFC.fit(X, y)
    importances=RFC.feature_importances_
    std = np.std([tree.feature_importances_ for tree in RFC.estimators_],axis=0)
    sorted_idx = np.argsort(importances)[::-1]
    sorted_cols = X.columns[sorted_idx]
    #print(sorted_cols)
    padding = np.arange(X.size/len(X)) + 0.5

    plt.clf()
    plt.barh(padding, importances[sorted_idx],xerr=std[sorted_idx], align='center')
    plt.yticks(padding, sorted_cols)
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()

    return sorted_cols

# Q5
def optimal_variable_selection(clf, X, y, X_test, y_test, sorted_cols):
    scores=np.zeros(X.shape[1])
    clf = deepcopy(clf)
    for f in np.arange(0, X.shape[1]):
        X1_f = X[sorted_cols[:f+1]]
        X2_f = X_test[sorted_cols[:f+1]]
        clf.fit(X1_f, y)
        YKNN = clf.predict(X2_f)
        scores[f] = np.round(accuracy_score(y_test, YKNN), 3)

    plt.clf()
    plt.plot(scores)
    plt.xlabel("Nombre de Variables")
    plt.ylabel("Accuracy")
    plt.title("Evolution de l'accuracy en fonction des variables")
    plt.show()

    return np.argmax(scores)

# Q5
def shap_explain(X, y, X_test, y_test):
    RFC= RandomForestClassifier(n_estimators=1000,random_state=1)
    RFC.fit(X, y)
    
    explainer = shap.TreeExplainer(RFC)
    shap_values = explainer.shap_values(X_test)
    
    # Importance globale
    plt.clf()
    shap.summary_plot(shap_values, X_test)
    

# Q5 Selection des meilleurs variables à partir des données normalisées avec nouvelles variables
def get_best_vars(clf, X, y, X_test, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=1)
    X_n_xt = normalize_xt_X(X)
    X_test_n_xt = normalize_xt_X(X_test)

    sorted_cols = variable_importance(clf, X_n_xt, y, X_test_n_xt, y_test)
    best_vars_num = optimal_variable_selection(clf, X_n_xt, y, X_test_n_xt, y_test, sorted_cols)

    print(f"Les {best_vars_num} premieres variables donnent le meilleur score: {sorted_cols[:best_vars_num]}")
    
    return sorted_cols, best_vars_num


# Q6 Selection des meilleurs parametres à partir des données normalisées avec nouvelles variables
def get_best_params(clf, X, y):
    X_n_xt = normalize_xt_X(X)
    def scoring(*args, **kwargs):
        return (accuracy_score(*args, **kwargs)+precision_score(*args, **kwargs))/2

    GSCV = GridSearchCV(clf, {'hidden_layer_sizes' : [[x+20,x] for x in range(20,30)]} , scoring = make_scorer(scoring)).fit(X_n_xt, y)
    return GSCV.best_params_


# Q7 Pipeline
def Q7_pipeline(X, y, name='final_pipeline_q7.pkl'):
    X_n_xt = normalize_xt_X(X)
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=3), MLPClassifier(hidden_layer_sizes=[47, 27], random_state=1, max_iter=200)).fit(X_n_xt, y)

    # Sauvegarder l'objet dans un fichier binaire
    with open(name, 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline

# Q8 Orchestration        
def pipeline_generation_train_test_split(clfs, X, y, name='final_pipeline_q8.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=1)
    
    results = [(clf[0], train_test(clf[1], X_train, y_train, X_test, y_test, label=clf[0])) for clf in clfs.items()]
    results_sorted = sorted(results, key=lambda x: x[1]["score"])

    best_clf = clfs[results_sorted[-1][0]] # on isole le classifier avec le score maximal
    best_clf = clfs["MLP"]
    best_vars, best_vars_num = get_best_vars(best_clf, X_train, y_train, X_test, y_test)
    best_params = get_best_params(best_clf, X_train, y_train)

    #pipeline = make_pipeline(StandardScaler(), PCA(n_components=3), best_clf.set_params(**best_params)).fit(X_train[best_vars[:best_vars_num]], y_train)
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=3), best_clf.set_params(**best_params)).fit(X_train, y_train)

    # Sauvegarder l'objet dans un fichier binaire
    with open(name, 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline
    

# Q10 Comparaison de plusieurs algorithmes d’apprentissage 
def run_classifiers_cv(clfs, X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for i in clfs:
        clf = clfs[i]
        cv_acc = cross_val_score(clf, X, y, cv=kf)
        print(f"Accuracy for {i} is:\t{np.mean(cv_acc):.3f} +/- {np.std(cv_acc):.3f}")

# Q10 Pipeline
def pipeline_generation_cv(clfs, X, y):
    X_norm = normalize_X(X)
    X_norm_xt= normalize_xt_X(X)

    print("Donnees brutes")
    run_classifiers_cv(clfs, X, y)
    print("Donnees normalisees")
    run_classifiers_cv(clfs, X_norm, y)
    print("Donnees normalisees + nouvelles variables")
    run_classifiers_cv(clfs, X_norm_xt, y)
    
