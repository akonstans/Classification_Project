import pandas as pd
import seaborn as sns
import new_lib as nl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import acquire
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def chi2(parameter, df):
    chi2, p, degf, expected = chi2, p, degf, expected = stats.chi2_contingency(pd.crosstab(df.churn, df[parameter]))
    if p <= .05:
        print(f'Comparing relationship between churn and {parameter}')
        print(chi2, p)
        print('------------------')
        print('\n')
        
# Using a chi2 test to determine the main drivers behind the churn at Telco 
# Does not include payments yet, have to drop na and make a seperate test below

def t_test(parameter, df):
    churn = df[df.churn == 'Yes'][parameter]
    not_churned = df[df.churn == 'No'][parameter]
    t, p = stats.ttest_ind(churn, not_churned, equal_var = False)
    print(f'Comparing relationship between churn and {parameter}')
    print(t, p)
    print('------------------')
    print('\n')
# T test for the charges because they we are comparing a catagorical variable to a continuous one
# The code will print the results in a similar format to the chi2 test above

def data_split(df, target):
    train, validate, test, X_train, y_train, X_val, y_val, X_test, y_test= nl.train_vailidate_test_split(df, target)
    
    return train, validate, test, X_train, y_train, X_val, y_val, X_test, y_test
# Creating splits for the data using custom libraray function

def contract_plot(df, x, hue):
    sns.countplot(x = df[x], hue = df[hue])
# Creating a count plot for contract_type hued by churn to visualize churn by each contract type

def payment_plot(df, y, hue):
    sns.countplot(y = df[y], hue = df[hue])
#Count Plot that visualizes payment_type and the count of churn within each type of payment

def dependents_plot(df, x, hue):
    sns.countplot(x = df[x], hue = df[hue])
# Countplot for dependents

def hist_plot(df, target, parameter, color1, color2, alpha1, alpha2, edgecolor, label1, label2, xlabel, ylabel, title):
    yes_churn = df[df[target] == 'Yes'][parameter]
    no_churn = df[df[target] =='No'][parameter]
    plt.hist(x = yes_churn, color = color1, alpha = alpha1, edgecolor = edgecolor, label = label1)
    plt.hist(x = no_churn, color = color2, alpha = alpha2, edgecolor = edgecolor, label = label2)
    plt.legend(loc = 'upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
# Creating two histograms to overlay on each other using monthly charges
# Seperated by churn status
# Originally tried to do in seaborn but cann only do using matplot

def make_binary(df, parameter):
    df[parameter] = np.where(df[parameter] == 'Yes', 1, 0)

def drop_cols(df, drop1, drop2, ax):
    df = df.drop([drop1, drop2], axis = ax)
    return df

def baseline(df, target):
    baseline = len(df[df[target] == 'No'])/ len(df)
    return baseline

def dec_tree(x, y, depth):
    train_tree = DecisionTreeClassifier(max_depth= depth, random_state=77)
    train_tree.fit(x, y)
    plt.figure(figsize=(13, 7))
    plot_tree(train_tree, feature_names=x.columns, class_names=train_tree.classes_, rounded=True)

    return train_tree

def tree_score(x,y,depth):
    train_tree = DecisionTreeClassifier(max_depth= depth, random_state=77)
    train_tree.fit(x, y)
    train_tree.score(x,y)
    return train_tree.score(x,y)

def tree_matrix(x, y, depth):
    train_tree = DecisionTreeClassifier(max_depth= depth, random_state=77)
    train_tree.fit(x, y)
    pred = train_tree.predict(x)
    labels = sorted(y.unique())
    df = pd.DataFrame(confusion_matrix(y, pred), index=labels, columns=labels)
    return df

def tree_report(x, y, depth):
    train_tree = DecisionTreeClassifier(max_depth= depth, random_state=77)
    train_tree.fit(x, y)
    pred = train_tree.predict(x)
    print(classification_report(y, pred))

def ran_score(x, y, depth):
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=depth, 
                            random_state=77)
    rf.fit(x, y)
    pred = rf.predict(x)
    rf.score(x, y)
    return rf.score(x, y)

def ran_matrix(x, y, depth):
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=depth, 
                            random_state=77)
    rf.fit(x, y)
    pred = rf.predict(x)
    labels = sorted(y.unique())
    df = pd.DataFrame(confusion_matrix(y, pred), index=labels, columns=labels)
    return df

def ran_report(x, y, depth):
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=depth, 
                            random_state=77)
    rf.fit(x, y)
    pred = rf.predict(x)
    print(classification_report(y, pred))

def log_reg_score(x, y, c):
    logit = LogisticRegression(C= c, random_state=77, intercept_scaling=1, solver='lbfgs')
    logit.fit(x, y)
    pred = logit.predict(x)
    logit.score(x, y)
    return logit.score(x, y)

def log_matrix(x, y, c):
    logit = LogisticRegression(C= c, random_state=77, intercept_scaling=1, solver='lbfgs')
    logit.fit(x, y)
    pred = logit.predict(x)
    labels = sorted(y.unique())
    df = pd.DataFrame(confusion_matrix(y, pred), index=labels, columns=labels)
    return df

def log_report(x, y, c):
    logit = LogisticRegression(C= c, random_state=77, intercept_scaling=1, solver='lbfgs')
    logit.fit(x, y)
    pred = logit.predict(x)
    print(classification_report(y, pred))
