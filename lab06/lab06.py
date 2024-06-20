# %%
from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)

# %% [markdown]
# # Task 1

# %%
from sklearn.model_selection import train_test_split

X_breast_cancer = data_breast_cancer.data
y_breast_cancer = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.2)

# %% [markdown]
# # Task 2 3 4

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pickle

X_breast = X_train[["mean texture", "mean symmetry"]]
X_breast_test = X_test[["mean texture", "mean symmetry"]]

vot_clf = VotingClassifier(
    estimators=[
        ('tree_clf', DecisionTreeClassifier()),
        ('l_reg', LogisticRegression()),
        ('kneigh_clf', KNeighborsClassifier())
    ],
    voting='hard'
)

acc_vote = []
classifiers = []
vot_clf.fit(X_breast, y_train)
for name, clf in vot_clf.named_estimators_.items():
    acc_vote.append((accuracy_score(y_train, clf.predict(X_breast)), accuracy_score(y_test, clf.predict(X_breast_test))))
    classifiers.append(clf)

acc_vote.append((accuracy_score(y_train, vot_clf.predict(X_breast)), accuracy_score(y_test, vot_clf.predict(X_breast_test))))
classifiers.append(vot_clf)

vot_clf = VotingClassifier(
    estimators=[
        ('tree_clf', DecisionTreeClassifier()),
        ('l_reg', LogisticRegression()),
        ('kneigh_clf', KNeighborsClassifier())
    ],
    voting='soft'
)

vot_clf.fit(X_breast, y_train)
acc_vote.append((accuracy_score(y_train, vot_clf.predict(X_breast)), accuracy_score(y_test, vot_clf.predict(X_breast_test))))
classifiers.append(vot_clf)

with open('acc_vote.pkl', 'wb') as acc_file:
    pickle.dump(acc_vote, acc_file)

with open('vote.pkl', 'wb') as vote_file:
    pickle.dump(classifiers, vote_file)

# %% [markdown]
# # Task 5 6

# %%
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

task5_classifiers = []
task5_acc = []

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, n_jobs=-1)
bag_clf.fit(X_breast, y_train)
task5_classifiers.append(bag_clf)

bag_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples = 0.5, n_jobs=-1)
bag_clf_half.fit(X_breast, y_train)
task5_classifiers.append(bag_clf_half)

pasting_clf = BaggingClassifier(DecisionTreeClassifier(), bootstrap=False, n_estimators=30, n_jobs=-1)
pasting_clf.fit(X_breast, y_train)
task5_classifiers.append(pasting_clf)

pasting_clf_half = BaggingClassifier(DecisionTreeClassifier(), bootstrap=False, n_estimators=30, max_samples = 0.5, n_jobs=-1)
pasting_clf_half.fit(X_breast, y_train)
task5_classifiers.append(pasting_clf_half)

rnd_clf = RandomForestClassifier(n_estimators=30, n_jobs=-1)
rnd_clf.fit(X_breast, y_train)
task5_classifiers.append(rnd_clf)

ada_clf = AdaBoostClassifier(n_estimators=30)
ada_clf.fit(X_breast, y_train)
task5_classifiers.append(ada_clf)

grad_clf = GradientBoostingClassifier(n_estimators=30)
grad_clf.fit(X_breast, y_train)
task5_classifiers.append(grad_clf)

for clf in task5_classifiers:
    train_acc = accuracy_score(y_train, clf.predict(X_breast))
    test_acc = accuracy_score(y_test, clf.predict(X_breast_test))
    task5_acc.append((train_acc, test_acc))

with open('acc_bag.pkl', 'wb') as acc_file:
    pickle.dump(task5_acc, acc_file)

with open('bag.pkl', 'wb') as bag_file:
    pickle.dump(task5_classifiers, bag_file)

# %% [markdown]
# # Task 7 8

# %%
bag2_clf = BaggingClassifier(n_estimators=30, max_features=2, max_samples=0.5)
bag2_clf.fit(X_train, y_train)

clf_lst = [bag2_clf]
clf_acc = [accuracy_score(y_train, bag2_clf.predict(X_train)), accuracy_score(y_test, bag2_clf.predict(X_test))]

with open('acc_fea.pkl', 'wb') as acc_fea:
    pickle.dump(clf_acc, acc_fea)

with open('fea.pkl', 'wb') as fea_file:
    pickle.dump(clf_lst, fea_file)

# %% [markdown]
# # Task 9

# %%
import pandas as pd

results = pd.DataFrame(columns=["Train Accuracy", "Test Accuracy", "Features"])

for est, feat in zip(bag2_clf.estimators_, bag2_clf.estimators_features_):
    train_acc = accuracy_score(y_train, est.predict(X_train.iloc[:, feat]))
    test_acc = accuracy_score(y_test, est.predict(X_test.iloc[:, feat]))
    results.loc[len(results)] = [train_acc, test_acc, feat]

results = results.sort_values(by=["Test Accuracy", "Train Accuracy"], ascending=False, ignore_index=True)
results.to_pickle('acc_fea_rank.pkl')

results.head(30)


