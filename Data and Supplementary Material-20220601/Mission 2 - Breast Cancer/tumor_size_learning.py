import numpy as np

from preproccess import load_and_clean_train_set
from sklearn.model_selection import cross_val_score

TRAIN_X_PATH = 'train.feats.csv'
TRAIN_Y_PATH = 'train.labels.1.csv'
from xgboost import XGBRegressor

if __name__ == '__main__':
    X, y = load_and_clean_train_set(TRAIN_X_PATH, TRAIN_Y_PATH, 'אבחנה-Tumor size')
    score = np.mean(np.absolute(
        cross_val_score(XGBRegressor(objective='reg:squarederror'), X, y, cv=5, scoring='neg_mean_squared_error')))
    print(score)

    # scores = []
    # for max_depth in range(1,30,1):
    #     forest = DecisionTreeClassifier(max_depth = max_depth)
    #     multi_target_forest = MultiOutputClassifier(forest)
    #     scores_macro = np.mean(cross_val_score(multi_target_forest, X, y, cv=5,scoring='f1_macro'))
    #     scores_micro = np.mean(cross_val_score(multi_target_forest, X, y, cv=5,scoring='f1_micro'))
    #     score_av = np.mean([scores_macro, scores_micro])
    #     print(f'{max_depth} : {score_av}')
    #     scores.append(score_av)
    # plt.plot(range(1,30,1),scores)
    # plt.show()

    # tree = DecisionTreeClassifier()
    # multi_target_forest = MultiOutputClassifier(tree)
    # multi_target_forest.fit(X, y)
    # for est in multi_target_forest.estimators_:
    #     plot_tree(est)
    #     plt.show()
    # scores_macro = np.mean(cross_val_score(multi_target_forest, X, y, cv=5,scoring='f1_macro'))
    # scores_micro = np.mean(cross_val_score(multi_target_forest, X, y, cv=5,scoring='f1_micro'))
    # score_av = np.mean([scores_macro, scores_micro])
    # print(f'{score_av}')

    # for c in range(5,100,5):
    #     knn = KNeighborsClassifier(n_neighbors=c)
    #     multi_target_forest = MultiOutputClassifier(knn)
    #     scores_macro = np.mean(cross_val_score(multi_target_forest, X, y, cv=5, scoring='f1_macro'))
    #     scores_micro = np.mean(cross_val_score(multi_target_forest, X, y, cv=5, scoring='f1_micro'))
    #     score_av = np.mean([scores_macro, scores_micro])
    #     print(f'{c} : {score_av}')
    #     scores.append(score_av)
    # plt.plot(range(5,100,5), scores)

    # clf = SVC(kernel='linear', C=1, random_state=42)
    # multi_target_forest = MultiOutputClassifier(svm, n_jobs=2)
    # multi_target_forest.fit(X, Y)
    # print(multi_target_forest.predict(x_test))
