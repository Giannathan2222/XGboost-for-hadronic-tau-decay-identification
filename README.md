# Hadronic tau detector
#### Distinguishes Hadronic tau decays from QCD jets and other backround noise in high energy physics event data, using XGBoost (native Python API, backed by C++).

#### Hadronic tau decays produce narrow, low-multiplicity jets that can be easily confused with QCD background jets. Efficient tau identification is essential in many LHC analyses, including Higgs boson decay channels.

# Explaining the model
### The model reads data from the root files of the CERN Monte Carlo simulation data
#### From the dataset the model finds all the possible tau variables (excluding tau_truth label which we will compare to afterwards).
## These include:
 - tau_pt = transverse momentum of the tau
 
 - tau_eta = pseudorapidity (detector location in comparison)
 
 - tau_nTracks = number of tracks associated with the tau
 
 - tau_phi = azimuthal angle
 
 - tau_charge = charge of the tau
 
 - tau_E = energy of the tau
#### These variables are used to train the model to make a prediction against the tau_truth label. 
#### The model then generates an ROC curve plotting the true positive rate against the false positive rate and generating an AUC curve to give us an idea about its accuracy on raw detector data.
#### The model also generates a feature importance graph. Each feature is given as a number which correlates to the order of which each feature is printed and ranked in terms of importance, allowing you to remove variables you believe are redundant.
#### I have reached AUC scores of train: 0.99552 and test: 0.99467 using these hyperparameters however results may depend on which files you use to train the data from (and of course if you so choose to change the hyperparams).
## Note:
#### You can load as many files as you need, it will make a difference if your training data has a variety of particle interactions as there are different types of Hadronic tau decays. So, dont train the data from only one type 
#### (the more variety the better)
#### Also ensure that between your files containing mostly true taus vs fake taus, there is a roughly 50:50 split. 

# Cross-Processing feature
## Not important for actually identifying true hadronic taus however can be usefel to effectively stress test your algorithm
### How it can be used
#### Runs parralel to the actual model. You can customise the environment that the algorithm has to train from, then test on.
#### You can specify from which 'processes' the algorithm has to find hadronic taus (if there are any).
#### Maybe a good idea to not just use the exact same training environment as your actual code as that would defeat the purpose.
## Go extreme. Push the limits. 
#### For example, i used 'Wtaunu' and 'Wmunu' as training data (containing hardly any taus ~560) and 'Ztautau' and 'VBFH' as testing data (containing almost entirely taus ~ 300,000) and got an AUC of ~0.85 suggesting that this machine learning is versatile under even more extreme environments.
#### However XG-Boost wont work if data contains only taus or no taus at all so bear that in mind

## Extra information
#### Requires https://opendata.cern.ch/record/15002 from CERN open data
#### Make sure you change the placeholder 'filepath' to the actual filepaths of your root files
# Good luck!
