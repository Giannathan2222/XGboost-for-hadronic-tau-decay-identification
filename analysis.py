import uproot
import numpy as np
import awkward as ak
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

#########################################################
# TRAINING THE HADRONIC TAU DECAY IDENTIFYING ALGORITHM # 
#########################################################



#load in multiple files to get a roughly equal balance of fake and true taus



#loading the root file(s) containing mostly true taus

filepath1 = filename
file1 = uproot.open(filepath1)

filepath2 = filename
file2 = uproot.open(filepath2)

#etc...

#loading the root file(s) containing mostly fake taus

filepath3 = filename3
file3 = uproot.open(filepath3)

filepath4 = filename4
file4 = uproot.open(filepath4)

#etc...
#discovering all possible tau variables

branches = file1["mini"].keys()

tau_vars = [b for b in branches if b.startswith("tau_") 
            and "truth" not in b.lower()]

features = tau_vars + ["tau_truthMatched"] #include truth label which will be compared to on y axis

#collecting our files

process_names = [
    "ggH",
    "Ztautau",
    "Wmunu",
    "Wtaunu",
] #are some examples of particle interactions but these should refer to the files that you upload

Files = [
    filepath1,
    filepath2,
    filepath3,
    filepath4,
   #etc..etc..(have as many as you need)
]

arrays = []
proc_labels = []

#merge files into awkward arrays and extract only selected features

for f, pname in zip(Files, process_names):
    arr = uproot.open(f)["mini"].arrays(features, library="ak")
    arrays.append(arr)
    # one label per tau candidate
    n = len(ak.flatten(arr["tau_pt"]))
    proc_labels.extend([pname] * n)

merged = ak.concatenate(arrays, axis=0)
proc_labels = np.array(proc_labels)

#make sure arrays are jagged so can be flattened
valid_tau_vars = [
    var for var in tau_vars
    if isinstance(merged[var].layout, ak.contents.ListOffsetArray)
]

#build feature matrix X 

X = np.column_stack([
    ak.to_numpy(ak.flatten(merged[var]))
    for var in valid_tau_vars
])

#build label vector Y 

y = ak.to_numpy(ak.flatten(merged["tau_truthMatched"])).astype(int) 
#y = correcttau(1)(true) vs jet structure(0)(false) 

print('X shape:', X.shape)
print('y shape:', y.shape)


#Clean the data (remove empty or infinite values)
mask = np.isfinite(X).all(axis=1)
X = X[mask]
y = y[mask]

#########################################################################################################

#double check for my own sanity
print(np.unique(y))
vals, counts = np.unique(y, return_counts=True)
print(list(zip(vals, counts))) #checks how unbalanced dataset is (real taus vs fakes)

print("Tau variables used (" + str(len(tau_vars)) + "):") #lets see what variables we are using
for v in tau_vars:
    print(str(v))

#########################################################################################################

#Building our test model

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist"
}

#training our model

evals = [(dtrain, "train"), (dtest, "test")]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=evals,
    early_stopping_rounds=20
)

#compute accuracy 


#########################################################
#cross products to check that xgboost is consistent
#build masks for other particle interactions
#YOU WILL WANT TO EXPERIMENT WITH DIFFERENT "PROCESS NAMES" to see how results vary and which environment the xgboost peforms best in!
#by doing so you can effectively "stress test" your algorithm to see if even under extreme conditions (ie: training data containing almost no taus vs test data containing almost entirely taus)
#>your algorithm will still be accurate
train_processes = ["process_name1", "process_name2"] #etc
test_processes  = ["process_name3", "process_name4"] #etc

train_mask = np.isin(proc_labels, train_processes)
test_mask  = np.isin(proc_labels, test_processes)

X_train_cross = X[train_mask]
y_train_cross = y[train_mask]

X_test_cross = X[test_mask]
y_test_cross = y[test_mask]

dtrain_cross = xgb.DMatrix(X_train_cross, label=y_train_cross)
dtest_cross  = xgb.DMatrix(X_test_cross,  label=y_test_cross)

model_cross = xgb.train(params, dtrain_cross, num_boost_round=300)

y_pred_cross = model_cross.predict(dtest_cross)
auc_cross = roc_auc_score(y_test_cross, y_pred_cross)

print("Cross-process AUC:", auc_cross)
#########################################################

#for training
y_pred_train = model.predict(dtrain)

train_auc = roc_auc_score(y_train, y_pred_train)
print("Train AUC:", train_auc)

#for test
y_pred = model.predict(dtest)

test_auc = roc_auc_score(y_test, y_pred)
print("test AUC:", test_auc)

#plot ROC curve

fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

#which variables matter the most?

xgb.plot_importance(model, max_num_features=10)
plt.show()
