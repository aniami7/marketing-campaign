# -*- coding: utf-8 -*-

"""
@author: Anna Mizera
Help functions 

"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def random_data(choose_random_state): # 
    """    
    
    Function randomly assigning records to training and testing sets
    
    """
    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test = train_test_split(
        pilot_X, pilot_t.values, pilot_y.values, test_size=0.2, random_state=choose_random_state
        )
    number = pilot['V2'].value_counts().count() + pilot['V19'].value_counts().count() # How many dummies were created
    col_no_dummies = columns_pilot_X[:-number] 
    del(col_no_dummies[-2])
    sc = StandardScaler()
    x_train[col_no_dummies] = sc.fit_transform(x_train[col_no_dummies])  
    x_test[col_no_dummies] = sc.transform(x_test[col_no_dummies])
    
    return x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test


def to_target(t_pred, prof_per_customer = 100, cost_per_customer = 5):
    t_pred_trans = pd.DataFrame(t_pred, columns= ["UPLIFT"]).sort_values("UPLIFT", ascending = False)
    t_pred_trans["Profit"] = t_pred_trans["UPLIFT"] * prof_per_customer - cost_per_customer
    chosen_clients = t_pred_trans[t_pred_trans["Profit"] > 0] 
    print("Call have positive impact on " + str(len(chosen_clients)) + " people" )
    # Check if the constraint has not been breached
    no_of_clients_treated = min(len(t_pred_trans[t_pred_trans["Profit"] > 0]), len(t_pred_trans)/4 ) 
    print("We should target " + str(no_of_clients_treated) + " people.")
    return no_of_clients_treated


def who_to_call(t_pred, prof_per_customer = 100, cost_per_customer = 5): 
    """
    
    Arguments: uplifts from numpy.ndarray format from the model
    
    """
    t_pred_trans = pd.DataFrame(t_pred, columns= ["UPLIFT"]).sort_values("UPLIFT", ascending = False)
    t_pred_trans["Profit"] = t_pred_trans["UPLIFT"] * prof_per_customer - cost_per_customer
    chosen_clients = t_pred_trans[t_pred_trans["Profit"] > 0] 
    # Check if the constraint has not been breached
    no_of_clients_treated = min(len(t_pred_trans[t_pred_trans["Profit"] > 0]), len(t_pred_trans)/4 )
    # Indices of customers chosen
    final_list = chosen_clients[:int(no_of_clients_treated)]
    index_of_clients = list(final_list.index)
    out = []
    for i in range(0,len(t_pred_trans)): 
        if i in index_of_clients:
            out.append("1")
        else:
            out.append("0")
    return out


def plot_my_uplift(t_pred, no_of_clients_treated, prof_per_customer = 100):
    t_pred_trans_plot = pd.DataFrame(t_pred, columns= ["UPLIFT"]).sort_values("UPLIFT", ascending = False,  ignore_index=True) * prof_per_customer
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#c1c6fc')
    ax.set_facecolor("#c1c6fc")
    ax.bar(t_pred_trans_plot.index, t_pred_trans_plot["UPLIFT"], color ="#ff81c0")
    ax.axhline(y= 5, linewidth=2, color='r')
    ax.axvline(x = no_of_clients_treated, linewidth=2, color='b', linestyle  = '--' )
    ax.set_xlabel("sorted_customers")
    ax.set_ylabel("uplift_per_customer")
    ax.set_title("UPLIFT PLOT")
    plt.show()
    

def plot_my_qini(t_pred, no_of_clients_treated, prof_per_customer = 100):
    t_pred_trans_plot = pd.DataFrame(t_pred, columns= ["UPLIFT"]).sort_values("UPLIFT", ascending = False,  ignore_index=True) * prof_per_customer
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#c1c6fc')
    ax.set_facecolor("#c1c6fc")
    ax.plot(t_pred_trans_plot.index, t_pred_trans_plot.cumsum()["UPLIFT"], color ="#ff81c0", linewidth = 4)
    ax.axvline(x = no_of_clients_treated, linewidth=2, color='b', linestyle  = '--' )
    ax.set_xlabel("sorted_customers")
    ax.set_ylabel("qini")
    ax.set_title("QINI PLOT")
    plt.show()

###################### Finish functions showing who should be called + uplift plots
def cross_val_T_learner_all(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, classic_ML_model):
    learner = BaseTClassifier(learner = classic_ML_model)
    learner.fit(X=x_train, y=outcome_train, treatment=treatment_train)
    t_pred = learner.predict(X=x_test) 
    # Aggregating data in a dataframe
    valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'T-Learner': t_pred.reshape(-1), 
                  })
    # print('AUUC:\n', round(auuc_score(valid_t), 5)  )  
    score = round(qini_score(valid_t), 5)
    return score

def cross_val_uplift_tree(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test):
    uplift_tree = UpliftTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_treatment=50,
                                    n_reg=100, evaluationFunction='ED', control_name="control")
    uplift_tree.fit(x_train.values,
                 np.where(treatment_train < 1, "control", "treatment"),
                 y=outcome_train)
    t_pred = uplift_tree.predict(X=x_test.values)[1]
    valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'Uplift-Tree': t_pred, 
                   })
    score = round(qini_score(valid_t), 5)
    return score

def cross_val_uplift_random_forest(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test):
    uplift_forrest = UpliftRandomForestClassifier(n_estimators=400, max_depth=14, min_samples_leaf=80, min_samples_treatment=50,
                                    n_reg=10, evaluationFunction='KL', control_name="control")
    uplift_forrest.fit(x_train.values,
                 np.where(treatment_train<1, "control", "treatment"),
                 y=outcome_train)
    forrest_pred = uplift_forrest.predict(X=x_test.values, full_output = False)
    valid_forrest = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'Uplift-Forrest': forrest_pred.reshape(-1)# gdy w predict full_output = False, 
                   })
    score = round(qini_score(valid_forrest), 5)
    return score
    
##########################
def cross_val_T_learner_all_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, classic_ML_model):
    learner = BaseTClassifier(learner = classic_ML_model)
    learner.fit(X=x_train, y=outcome_train, treatment=treatment_train)
    t_pred = learner.predict(X=x_test) 
    # Aggregating data in a dataframe
    valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'T-Learner': t_pred.reshape(-1), 
                  })
    # print('AUUC:\n', round(auuc_score(valid_t), 5)  )  
    score = round(auuc_score(valid_t), 5)
    return score

# The following ones remain separate
def cross_val_uplift_tree_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test):
    uplift_tree = UpliftTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_treatment=50,
                                    n_reg=100, evaluationFunction='ED', control_name="control")
    uplift_tree.fit(x_train.values,
                 np.where(treatment_train < 1, "control", "treatment"),
                 y=outcome_train)
    t_pred = uplift_tree.predict(X=x_test.values)[1]
    valid_t = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'Uplift-Tree': t_pred, 
                   })
    score = round(auuc_score(valid_t), 5)
    return score

def cross_val_uplift_random_forest_auuc(x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test):
    uplift_forrest = UpliftRandomForestClassifier(n_estimators=400, max_depth=14, min_samples_leaf=80, min_samples_treatment=50,
                                    n_reg=10, evaluationFunction='KL', control_name="control")
    uplift_forrest.fit(x_train.values,
                 np.where(treatment_train<1, "control", "treatment"),
                 y=outcome_train)
    forrest_pred = uplift_forrest.predict(X=x_test.values, full_output = False)
    valid_forrest = pd.DataFrame({'y': outcome_test,
                   'w': treatment_test,
                   'Uplift-Forrest': forrest_pred.reshape(-1)# if in predict full_output == False
                   })
    score = round(auuc_score(valid_forrest), 5)
    return score
    









