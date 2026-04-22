import uproot
import glob
import numpy as np
import awkward as ak
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

#########################################################
# TRAINING THE HADRONIC TAU DECAY IDENTIFYING ALGORITHM # 
#########################################################

arrays = []
proc_labels = []


#load in every file to get a combination of fake and true taus

directory = r"C:\Users\Gianluigi\Documents\1lep1tau\1lep1tau\MC"
all_files = glob.glob(directory + "/*.root")

#discovering all possible tau variables

#only include variables that dont indirectly reveal tau identity
valid_tau_vars = ["tau_pt", "tau_eta", "tau_phi", "tau_E", "tau_charge", "tau_nTracks"]
features = valid_tau_vars + ["tau_truthMatched",
         "lep_pt",
         "lep_eta",
         "lep_phi"] #include truth label which will be compared to on y axis


#removing files not containing tau candidates
def file_has_taus(filepath):
    try:
        arr = uproot.open(filepath)["mini"].arrays(["tau_pt"], library="ak")
        return len(ak.flatten(arr["tau_pt"])) > 0
    except:
        return False
    
process_names = [
    "ggH",
    "Ztautau",
    "Wmunu",
    "Wtaunu",
    "VBFH",
    "Zee",
    "WqqZll",
    "singleTop",
    "WplvWmqq",
    "ttbar"
]

#merge files into awkward arrays and extract only selected features

for f in all_files:
    if not file_has_taus(f):
        print(f"Skipping (no taus): {f}")
        continue

    print(f"Loading: {f}")
    arr = uproot.open(f)["mini"].arrays(features, library="ak")
    arrays.append(arr)

    # process label from filename: mc_XXXXXX.PROCNAME....
    fname = f.split("\\")[-1]          # Windows path → take last part
    pname = fname.split(".")[1].split("_")[0]      # e.g. "Ztautau_PTV0_70_CVetoBVeto"
    n = len(ak.flatten(arr["tau_pt"]))
    proc_labels.extend([pname] * n)

############################
#creating our own variables#
############################

merged = ak.concatenate(arrays, axis=0)
proc_labels = np.array(proc_labels)


tau_pt     = ak.to_numpy(ak.flatten(merged["tau_pt"]))
tau_eta    = ak.to_numpy(ak.flatten(merged["tau_eta"]))
tau_phi    = ak.to_numpy(ak.flatten(merged["tau_phi"]))
tau_E      = ak.to_numpy(ak.flatten(merged["tau_E"]))
tau_charge = ak.to_numpy(ak.flatten(merged["tau_charge"]))
tau_nTrk   = ak.to_numpy(ak.flatten(merged["tau_nTracks"]))

# avoid division-by-zero issues
tau_pt = np.where(tau_pt == 0, 1e-6, tau_pt)
tau_E  = np.where(tau_E  == 0, 1e-6, tau_E)

# 1) log(pt)
log_tau_pt = np.log(tau_pt)

# 2) |eta|
abs_tau_eta = np.abs(tau_eta)

# 3) pt / E
pt_over_E = tau_pt / tau_E

# 5 normalised track count
nTrk_over_pt = tau_nTrk / tau_pt

# 6) approximate tau mass from (E, pt, eta, phi)
# treat tau as a massless 4-vector and reconstruct invariant mass of "cluster"
# here we just use m^2 = E^2 - p^2 with p ≈ pt * cosh(eta)
p = tau_pt * np.cosh(tau_eta)
m2 = tau_E**2 - p**2
tau_mass_approx = np.sqrt(np.clip(m2, 0, None))

####################################
# EVENT-LEVEL & ISOLATION FEATURES #
####################################

# number of taus in each event
n_taus_event = ak.to_numpy(ak.num(merged["tau_pt"]))
n_taus_event = np.repeat(n_taus_event, ak.num(merged["tau_pt"]))

# tau index inside the event (0 = leading tau)
tau_index = ak.to_numpy(ak.flatten(ak.local_index(merged["tau_pt"])))

# sum of tau pt in the event
sum_tau_pt = ak.sum(merged["tau_pt"], axis=1)
sum_tau_pt = ak.to_numpy(sum_tau_pt)
sum_tau_pt = np.repeat(sum_tau_pt, ak.num(merged["tau_pt"]))

# tau pt fraction of total tau pt in event
tau_pt_fraction = tau_pt / sum_tau_pt

# number of leptons in the event (isolation proxy)
n_leps_event = ak.to_numpy(ak.num(merged["lep_pt"]))
n_leps_event = np.repeat(n_leps_event, ak.num(merged["tau_pt"]))


#build feature matrix X 

X = np.column_stack([
    tau_pt,
    tau_eta,
    tau_phi,
    tau_E,
    tau_charge,
    tau_nTrk,
    log_tau_pt,
    abs_tau_eta,
    pt_over_E,
    nTrk_over_pt,
    tau_mass_approx,
    n_taus_event,
    tau_index,
    tau_pt_fraction,
    n_leps_event,
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
    "eta": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist"
}

#training our model

evals = [(dtrain, "train"), (dtest, "test")]

model = xgb.train(
    params,
    dtrain,
    num_boost_round= 250,
    evals=evals,
    early_stopping_rounds=30
)

#compute accuracy 


#########################################################
#cross products to check that xgboost is consistent
#build maks for other particle interactions
#YOU WILL WANT TO EXPERIMENT WITH DIFFERENT "PROCESS NAMES" to see how results vary and which environment the xgboost peforms best in!
train_mask = np.array(
    [("Wtaunu" in p) or ("Wmunu" in p) for p in proc_labels]
)

test_mask = np.array(
    [("Ztautau" in p) or ("VBFH" in p) for p in proc_labels]
)
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

xgb.plot_importance(model, max_num_features=20)
plt.show()
