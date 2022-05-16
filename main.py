# -*- coding: utf-8 -*-
"""
@author: Anna Mizera
Main script:
    - Data processing
    - Model training and testing
    - Results validation
    - Printing and plotting the results (uplifts, cumulative uplifts, AUUC, Gini scores)
"""



#########################################################################
# 0 Directory, import functions
#########################################################################
import os 

# Set directory
directory = "..."
os.chdir(directory)

# Read in help functions saved in FUNCTIONS.py file
    
#########################################################################
# I Import the libraries
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from causalml.inference.meta import BaseTClassifier
from causalml.metrics import plot, plot_qini, auuc_score, qini_score
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier, uplift_tree_string, uplift_tree_plot

from xgboost import XGBClassifier

from IPython.display import Image
import graphviz
#########################################################################
# II Data preprocessing
#########################################################################
plt.style.use("seaborn")
# random_state=42

# 1. Importing the pilot dataset
pilot = pd.read_csv('pilot.csv')

# 2. Check if any missing data occurs (none found)
pilot.isna().any()

# 3. Division of dataset into three subsets: X, Y and treatment
pilot_X = pilot.iloc[:, :-2]
pilot_y = pilot.iloc[:, -1]
pilot_t =  pilot.iloc[: ,-2]

# 4. Encoding categorical_data: adding dummy variables
pilot_X = pd.get_dummies(pilot_X, columns=['V2', "V19"])
columns_pilot_X = list(pilot_X.columns) 

# 5. training set and test set na X, Y, t
x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = train_test_split(
        pilot_X, pilot_t.values, pilot_y.values, test_size=0.2, random_state=42
    )

# 6. feature scaling_train
# omit the added dummy variables V2 and V19
number = pilot['V2'].value_counts().count() + pilot['V19'].value_counts().count() # count how many dummies were created
col_no_dummies = columns_pilot_X[:-number] 
del(col_no_dummies[-2])
sc = StandardScaler()
x_train[col_no_dummies] = sc.fit_transform(x_train[col_no_dummies])  
x_test[col_no_dummies] = sc.transform(x_test[col_no_dummies])


# 7. Visualizing data
cols = list(pilot.columns)
for i in cols:
    sns_plot = sns.displot(data=pilot, x=i, bins=20)
    plt.savefig('PLOTS\\{}.png'.format(i), dpi=100)

# Profit gained for each customer buying the product
prof_per_customer = 100

#########################################################################
# III T-Learner models
#########################################################################
# T-Learner XG boost
# Training
xgb_tlearner = BaseTClassifier(learner=XGBClassifier(random_state=42, n_estimators = 300,
                               max_depth = 5, learning_rate = 0.1))
xgb_tlearner.fit(X=x_train, y=outcome_train, treatment=treatment_train)

# Predictions
t_pred = xgb_tlearner.predict(X=x_test)  # returns Predictions of treatment effects
t_pred_xgb = t_pred
# Aggregating everything on a dataframe
valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'T-Learner': t_pred.reshape(-1), 
                  })

# Validation measures
print('AUUC:\n',round(auuc_score(valid_t), 5))
print('QINI:\n',qini_score(valid_t))
## Plotting the 3 types of up # to ja zrobilam !

# Validation plots lift curve. 
plot_qini = plot(valid_t, kind='qini', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot_lift = plot(valid_t, kind='lift', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot_gain = plot(valid_t ,kind='gain', outcome_col='y', treatment_col='w',figsize=(10, 3.3))

# Additional functions and plots
to_target(t_pred)
plot_t_xgboost = plot_my_uplift(t_pred, to_target(t_pred))
plot_qini_xgboost = plot_my_qini(t_pred, to_target(t_pred))

#########################################################################
# III.1 T-Learner: Logisitc regression
#########################################################################
# Training
log_learner = BaseTClassifier(learner = LogisticRegression(random_state = 42))
log_learner.fit(X=x_train, y=outcome_train, treatment=treatment_train)

# Predictions
t_pred = log_learner.predict(X=x_test)
t_pred_log = t_pred

# Validation of predictions
# type(t_pred)
# pd.unique(t_pred == uplift)
# uplift = pd.DataFrame(t_pred[0])# uplift
# pd.DataFrame(t_pred[1]) # uplift control
# pd.DataFrame(t_pred[2]) # upfit treat
# pd.DataFrame(t_pred[2])-pd.DataFrame(t_pred[1]) == pd.DataFrame(t_pred[0])

## Aggregating everything on a dataframe
valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'T-Learner': t_pred.reshape(-1), 
                  })

# Validation measures
print('AUUC:\n',round(auuc_score(valid_t), 5))
print('QINI:\n', qini_score(valid_t))

## Plotting the 3 types of uplift curve. 
plot(valid_t,kind='qini', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot(valid_t,kind='lift', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot(valid_t,kind='gain', outcome_col='y', treatment_col='w',figsize=(10, 3.3))

# Extra functions and plots
to_target(t_pred)
plot_t_logistic = plot_my_uplift(t_pred, to_target(t_pred))
plot_qini_logistic = plot_my_qini(t_pred, to_target(t_pred))

#########################################################################
# III.2 T-Learner: K-Nearest Neighbours
#########################################################################
# Training
knn_learner = BaseTClassifier(learner = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))
knn_learner.fit(X= x_train, y= outcome_train, treatment= treatment_train)

## Predictions
t_pred = knn_learner.predict(X=x_test)
t_pred_knn = t_pred

## Aggregating everything on a dataframe
valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'T-Learner': t_pred.reshape(-1), 
                  })

## Plotting the 3 types of uplift curve. 
plot(valid_t,kind='qini', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot(valid_t,kind='lift', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot(valid_t,kind='gain', outcome_col='y', treatment_col='w',figsize=(10, 3.3))

# Extra functions and plots
to_target(t_pred)
plot_t_knn = plot_my_uplift(t_pred, 250)
plot_qini_knn = plot_my_qini(t_pred, 250)


#########################################################################
# III.3 T-Learner: Decision Tree
#########################################################################
# Training
tree_learner = BaseTClassifier(learner = DecisionTreeClassifier(criterion = 'gini', random_state = 42, max_depth=5, min_samples_leaf=200))
tree_learner.fit(X=x_train, y=outcome_train, treatment=treatment_train)

## Predictions
t_pred = tree_learner.predict(X=x_test)
t_pred_dec_tree = t_pred

## Aggregating everything on a dataframe
valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'T-Learner': t_pred.reshape(-1), 
                  })

## Plotting the 3 types of uplift curve. 
plot(valid_t,kind='qini', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot(valid_t,kind='lift', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot(valid_t,kind='gain', outcome_col='y', treatment_col='w',figsize=(10, 3.3))

# Extra functions and plots
to_target(t_pred)
plot_t_tree = plot_my_uplift(t_pred, to_target(t_pred))
plot_qini_tree = plot_my_qini(t_pred, to_target(t_pred))


################  Additional check - how the decision tree model works
# x_train["treatment_train"] = treatment_train
# x_train["outcome_train"] = outcome_train

# x_control = x_train[x_train["treatment_train"] == 0]
# x_treatment = x_train[x_train["treatment_train"] == 1]

# del x_control["treatment_train"]
# del x_treatment["treatment_train"]

# classifier_con  = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
# classifier_treat  = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)

# x_con = classifier_con.fit(x_control.iloc[:, :-1],x_control.iloc[:, -1])
# x_treat = classifier_treat.fit(x_treatment.iloc[:, :-1],x_treatment.iloc[:, -1])

# y_pred_con = x_con.predict(x_test)
# y_pred_treat = x_treat.predict(x_test)
# wynik_1 = pd.DataFrame(y_pred_treat-y_pred_con)
# wynik_2 = pd.DataFrame(t_pred_dec_tree)

# np.unique(wynik_1 == wynik_2)
# tree.plot_tree(x_con);

#########################################################################
# III.4 T-Learner: Random Forest
#########################################################################
# Training 
random_forest = BaseTClassifier(learner = RandomForestClassifier(n_estimators = 400, criterion = 'entropy', random_state = 42, max_depth=14, min_samples_leaf=80 ))
random_forest.fit(X=x_train, y=outcome_train, treatment=treatment_train)

# Predictions
t_pred = random_forest.predict(X=x_test)
t_pred_random_forest = t_pred

# Aggregating everything on a dataframe
valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'T-Learner': t_pred.reshape(-1), 
                  })

# Extra functions and plots
to_target(t_pred)
plot_forest = plot_my_uplift(t_pred, to_target(t_pred))
plot_qini_forest = plot_my_qini(t_pred, to_target(t_pred))

#########################################################################
# IV.1 Direct Uplift: Trees
#########################################################################
# Training 
uplift_tree = UpliftTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_treatment=50,
                                    n_reg=100, evaluationFunction='ED', control_name="control")
uplift_tree.fit(x_train.values,
                 np.where(treatment_train < 1, "control", "treatment"),
                 y=outcome_train)

# Visualizing decision tree 
features_names = list(x_train.columns)
graph = uplift_tree_plot(uplift_tree.fitted_uplift_tree, features_names)
Image(graph.create_png())

# Predictions
t_pred = uplift_tree.predict(X=x_test.values)[1]
valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'Uplift-Tree': t_pred, 
                   })


print('AUUC:\n',auuc_score(valid_t))
## Plotting the 3 types of uplift curve. 
plot(valid_t,kind='qini', outcome_col='y', treatment_col='w',figsize=(10, 3.3))
plot(valid_t,kind='lift', outcome_col='y', treatment_col='w',figsize=(10, 3.3))


# Predictions - different format of prediction output to use my functions below (to get uplifts, use full_output=True)
tree_pred_true = uplift_tree.predict(X=x_test.values, full_output=True)  
hidden_uplifts = np.array(tree_pred_true[2])[:, 0] 

to_target(hidden_uplifts)
plot_t_tree = plot_my_uplift(hidden_uplifts, to_target(hidden_uplifts))
plot_qini_tree = plot_my_qini(hidden_uplifts, to_target(hidden_uplifts))

#########################################################################
# IV.2 Direct Uplift: Random Forest
#########################################################################
# Training 
uplift_forrest = UpliftRandomForestClassifier(n_estimators=400, max_depth=14, min_samples_leaf=80, min_samples_treatment=50,
                                    n_reg=10, evaluationFunction='KL', control_name="control")
uplift_forrest.fit(x_train.values,
                 np.where(treatment_train<1, "control", "treatment"),
                 y=outcome_train)

# Predictions
forrest_pred = uplift_forrest.predict(X=x_test.values, full_output = False)

## Aggregating everything on a dataframe
valid_forrest = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'Uplift-Forrest': forrest_pred.reshape(-1) # if in predict full_output = False 
                   })

## Plotting the 3 types of uplift curve
print('AUUC:\n',auuc_score(valid_forrest))
plot(valid_forrest,kind='qini', outcome_col='y', treatment_col='w',figsize=(10, 3.3))

# Predictions - different format of prediction output to use my functions below, full_output = True
forrest_pred_true = uplift_forrest.predict(X=x_test.values, full_output = True)
delta_uplifts = np.array(forrest_pred_true["delta_treatment"])

to_target(delta_uplifts)
plot_my_uplift(delta_uplifts, to_target(delta_uplifts))
plot_my_qini(delta_uplifts, to_target(delta_uplifts))

#########################################################################
# V Cross validation
#########################################################################
# Qini score
# Validation performed by randomly assigning records to training and testing sets 20 times
# Calculate each time Qini score, Auuc score, averages across thecalculations
no_of_vals = 20
 
# XBGoost
out_xgb_qini = []   
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_xgb_qini.append(cross_val_T_learner_all(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                           XGBClassifier(random_state=i, n_estimators = 300, max_depth = 5, learning_rate = 0.1)))
 
# Logisitc Regression
out_log_qini = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_log_qini.append(cross_val_T_learner_all(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                           LogisticRegression(random_state = i)))
       
# KNN
out_knn_qini = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_knn_qini.append(cross_val_T_learner_all(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                           KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)  ))

# Decision Tree
out_tree_qini = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_tree_qini.append(cross_val_T_learner_all(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                            DecisionTreeClassifier(criterion = 'gini', random_state = i, max_depth=5, min_samples_leaf=200) ))

# Random forest
out_forest_qini = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_forest_qini.append(cross_val_T_learner_all(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                              RandomForestClassifier(n_estimators = 400, criterion = 'entropy', random_state = i, max_depth=14,min_samples_leaf=80) ))

# Uplift decision tree
out_uplift_tree_qini = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_uplift_tree_qini.append(cross_val_uplift_tree(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test))
    
# Uplift random forest
out_uplift_forest_qini = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_uplift_forest_qini.append(cross_val_uplift_random_forest(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test))


models =["XGBoost", "Logistic regression", "Knn", "Decision trees", "Random forest", "Uplift decision trees", "Uplift random forest"]
frames = [pd.DataFrame(out_xgb_qini)["T-Learner"],
          pd.DataFrame(out_log_qini)["T-Learner"], 
          pd.DataFrame(out_knn_qini)["T-Learner"], 
          pd.DataFrame(out_tree_qini)["T-Learner"], 
          pd.DataFrame(out_forest_qini)["T-Learner"], 
          pd.DataFrame(out_uplift_tree_qini)["Uplift-Tree"], 
          pd.DataFrame(out_uplift_forest_qini)["Uplift-Forrest"]]  
                                  
cross_val_results = pd.concat(frames, axis = 1)
cross_val_results.columns= models
# Add row with column averages
cross_val_results.loc['mean'] = cross_val_results.mean()
# Add a column showing the model with highest score in the run
cross_val_results = pd.concat([cross_val_results, cross_val_results.idxmax(axis=1)], axis = 1)
cross_val_results



# Table with AUUC Score
out_xgb_auuc = [] 

# XGBoost   
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_xgb_auuc.append(cross_val_T_learner_all_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                           XGBClassifier(random_state=i, n_estimators = 300, max_depth = 5, learning_rate = 0.1)))
    
# Logistic regression
out_log_auuc = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_log_auuc.append(cross_val_T_learner_all_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                           LogisticRegression(random_state = i)))
# Knn
out_knn_auuc = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_knn_auuc.append(cross_val_T_learner_all_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                           KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)  ))
# Decision trees
out_tree_auuc = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_tree_auuc.append(cross_val_T_learner_all_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                            DecisionTreeClassifier(criterion = 'gini', random_state = i, max_depth=5, min_samples_leaf=200) ))
# Random forest
out_forest_auuc = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_forest_auuc.append(cross_val_T_learner_all_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, 
                                              RandomForestClassifier(n_estimators = 400, criterion = 'entropy', random_state = i, max_depth=14,min_samples_leaf=80) ))
# Uplift decision trees
out_uplift_tree_auuc = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_uplift_tree_auuc.append(cross_val_uplift_tree_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test))
    
# Uplift random forest
out_uplift_forest_auuc = []
for i in range(no_of_vals):
    random_data(i)
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = random_data(i)
    out_uplift_forest_auuc.append(cross_val_uplift_random_forest_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test))


models =["XGBoost", "Logistic regression", "Knn", "Decision trees", "Random forest", "Uplift decision trees", "Uplift random forest"]
frames = [pd.DataFrame(out_xgb_auuc)["T-Learner"],
          pd.DataFrame(out_log_auuc)["T-Learner"], 
          pd.DataFrame(out_knn_auuc)["T-Learner"], 
          pd.DataFrame(out_tree_auuc)["T-Learner"], 
          pd.DataFrame(out_forest_auuc)["T-Learner"], 
          pd.DataFrame(out_uplift_tree_auuc)["Uplift-Tree"], 
          pd.DataFrame(out_uplift_forest_auuc)["Uplift-Forrest"]]        
                            
cross_val_results = pd.concat(frames, axis = 1)
cross_val_results.columns= models
# Add row with column averages
cross_val_results.loc['mean'] = cross_val_results.mean()
# Add a column showing the model with highest score in the run
cross_val_results = pd.concat([cross_val_results, cross_val_results.idxmax(axis=1)], axis = 1)
cross_val_results



#########################################################################
# VI Summary Plots
#########################################################################

# Gini curve scaled by 100 and random state = 42
t_pred_all = [t_pred_xgb, t_pred_log, t_pred_knn, t_pred_dec_tree, t_pred_random_forest, hidden_uplifts, delta_uplifts]
labels = ["xgb", "log", "knn", "tree", "forest", "up_tree", "up_forest"]
plots = []
colours = ["#06470c", "#fe01b1", "#a0025c", "#dbb40c", "#069af3", "#fdff52", "#0504aa"]
fig, ax = plt.subplots()
fig.patch.set_facecolor('#c1c6fc')
ax.set_facecolor("#c1c6fc")

# Start iterating through each model
i = 0
for t_pred in t_pred_all:   
    t_pred_trans_plot = pd.DataFrame(t_pred, columns= ["UPLIFT"]).sort_values("UPLIFT", ascending = False,  ignore_index=True) * prof_per_customer
    plocik = ax.plot(t_pred_trans_plot.index, t_pred_trans_plot.cumsum()["UPLIFT"], color = colours[i], label = labels[i] )
    i= i + 1
ax.set_xlabel("sorted_customers")
ax.set_ylabel("uplift_per_customer")
ax.set_title("UPLIFT PLOT")
ax.legend()
plt.show()

#########################################################################
# VII Customer Base
# Data preprocessing and predictions for Uplift Random Forest Model
#########################################################################

# Data preprcoessing
# The format and type of the columns should be in line with the input of the function
# Format Customer Base data to be in line with pilot format
Customer_Base = pd.read_csv("CustomerBase.csv")
# Check if any NAs occur
# Customer_Base.isna().any() 
number = Customer_Base['V2'].value_counts().count() + Customer_Base['V19'].value_counts().count() 
# Customer_Base['V19'].value_counts().count() # different number of categories than in pilot

# Encoding categorical_data
Customer_Base = pd.get_dummies(Customer_Base, columns=['V2', "V19"]) # all of the data is in here, it just needs to be scaled
columns_Customer_Base = list(Customer_Base.columns) 

# feature scaling_
# How many dummies are created #18
columns_Customer_Base = columns_Customer_Base[:-number] 
del(columns_Customer_Base[-2])  
sc = StandardScaler()
Customer_Base_old = Customer_Base
Customer_Base[columns_Customer_Base] = sc.fit_transform(Customer_Base[columns_Customer_Base])  

# Final data to make projections for selected model
# Disregard the columns that do not have a lot of data - leave only columns that are in training data
columns_allowed = list(x_test.columns)

# Data to be used to make predictions
Customer_Base_final = Customer_Base[columns_allowed] 

# Prediction on Customer Base for Direct Uplift Random Forest 
forrest_pred_customer_base = uplift_forrest.predict(X=Customer_Base_final.values, full_output = True)
delta_uplifts_customer_base = np.array(forrest_pred_customer_base["delta_treatment"])


to_target(delta_uplifts_customer_base)
decisions = who_to_call(delta_uplifts_customer_base)
decisions = pd.DataFrame(decisions, columns = ["Customer_base"])


plot_my_uplift(delta_uplifts_customer_base, to_target(delta_uplifts_customer_base))
plot_my_qini(delta_uplifts_customer_base, to_target(delta_uplifts_customer_base))

# saving decisions to csv
decisions.to_csv(path_or_buf="decisions.csv", index=False)








