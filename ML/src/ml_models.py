from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier

models_dict = {
    "KNearestNeighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights="distance"),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "SupportVectorMachine": SVC(kernel="poly", degree=5),
    "LogisticRegression": LogisticRegression(solver="saga", n_jobs=-1),
    "ArtificalNeuralNetwork": MLPClassifier(hidden_layer_sizes=30, max_iter=2000, solver="lbfgs"),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "ExtraTree": ExtraTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="error", n_jobs=-1, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=128, n_jobs=-1, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=128, learning_rate=1.0, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=128, n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=128, learning_rate=1.0, random_state=42),
}