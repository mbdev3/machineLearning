import pandas as pd # data analysis and manipulation
import numpy as np #  arrays manipulation
import matplotlib.pyplot as plt # to create visualizations
from termcolor import colored as cl # Color formatting for output in terminal
import seaborn as sns # data vizualization based on matplotlib

from imblearn.over_sampling import SMOTE #Oversampling for Imbalanced Classification (duplicate examples from the minority class)

import warnings # ignore warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets

## scaling
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.preprocessing import MinMaxScaler # data scaling
from sklearn.model_selection import train_test_split # data split

## Algorithms
from sklearn.tree import DecisionTreeClassifier # decision tree classifier
from sklearn.neighbors import KNeighborsClassifier # k-nearest neighbors
from sklearn.linear_model import LogisticRegression # Logistic regression 
from sklearn.ensemble import RandomForestClassifier # random forest classifier
from xgboost import XGBClassifier # XGBoost algorithm
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes
from sklearn.neural_network import MLPClassifier # Multi-layer Perceptron classifier
from sklearn.svm import SVC # C-Support Vector Classification

from sklearn import metrics

import time

from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

##

import multiprocessing 

total_time = time.time()


df = pd.read_csv(r'archive/creditcard.csv') #importing data
df = df.loc[1:50000] # limit data to first 10.000 cols



cases = len(df) # get length of the dataset to conclude num of cases
nonfraud_count = len(df[df.Class == 0]) # count of non fraud cases
fraud_count = len(df[df.Class == 1]) # count of fraud cases
fraud_percentage = round(fraud_count/nonfraud_count*100, 2) # the percentage of fraud cases

print(cl('CASE COUNT : ', attrs = ['bold'],color='red'))
print(cl('--------------------------------------------', attrs = ['bold'],color='red'))
print('Total number of cases are :',end=" ")
print(cl('{}'.format(cases), attrs = ['bold'],color='green'))
print('Non-fraud cases are :',end=" ")
print(cl('{}'.format(nonfraud_count), attrs = ['bold'],color='green'))
print('Fraud cases are :',end=" ")
print(cl('{}'.format(fraud_count), attrs = ['bold'],color='green'))
print('Percentage of fraud cases is :',end=" ")
print(cl('{}%'.format(fraud_percentage), attrs = ['bold'],color='green'))
print(cl('--------------------------------------------', attrs = ['bold'],color='red'))


nonfraud_cases = df[df.Class == 0]  # get non fraud cases only
fraud_cases = df[df.Class == 1] # #get fraud cases only

print(cl('CASE AMOUNT STATISTICS', attrs = ['bold'],color="red"))
print(cl('--------------------------------------------', attrs = ['bold'],color="red"))
print(cl('>>> NON-FRAUD CASE AMOUNT STATS <<<', attrs = ['bold']))
print((nonfraud_cases.Amount.describe()))
print(cl('--------------------------------------------', attrs = ['bold'],color="red"))
print(cl('>>> FRAUD CASE AMOUNT STATS <<<', attrs = ['bold']))
print(fraud_cases.Amount.describe())
print(cl('--------------------------------------------', attrs = ['bold'],color="red"))


df['amount_log'] = np.log(df.Amount + 0.01) # Scale amount by log

ss = StandardScaler()
df['amount_scaled'] = ss.fit_transform(df['Amount'].values.reshape(-1,1)) #Scale amount by Standardization

norm = MinMaxScaler()
df['amount_minmax'] = norm.fit_transform(df['Amount'].values.reshape(-1,1)) #Scale amount by Normalization


X = df.drop(['Class','Amount','amount_minmax','amount_log'],axis=1) #remove those rows, we use amount_scaled instead of amount
y = df['Class'] # we will use y to compare it with the input


# we allocate 80% of the data for training and the remaining for testing
#X_train == training input, X_test == testing input, y_train == training output, y_test == testing output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True) 

print("X_train: ",X_train.shape)
print("y_train: ",y_train.shape)
print("X_test: ",X_test.shape)
print("y_test: ",y_test.shape)
print('\n')
print('............')
print('\n')


#duplicate examples from the fraud class (minority) to balance the data

smote= SMOTE(sampling_strategy='minority')
X_train_smote,y_train_smote=smote.fit_resample(X_train, y_train)
X_test_smote, y_test_smote = X_test, y_test

print("X_train_smote: ",X_train_smote.shape)
print("y_train_smote: ",y_train_smote.shape)
print("X_test_smote: ",X_test_smote.shape)
print("y_test_smote: ",y_test_smote.shape)



#store the results here for later comparaison
names=[]
aucs_tests = []   #AUC-ROC CURVE https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
accuracy_tests = []
precision_tests = []
recall_tests = []
f1_score_tests = []
mcc_score_tests = []
balanced_accuracy_tests = []
top_k_accuracy_tests = []
cohen_kappa_tests = []

# confusion matrix plot
def confusion_matrix_plot(y_test, y_test_pred):
  confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred) 
  plt.clf()
  plt.imshow(confusion_matrix, cmap=plt.cm.Accent)
  categoryNames = ['non fraud','fraud']
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  ticks = np.arange(len(categoryNames))
  plt.xticks(ticks, categoryNames,color='brown')
  plt.yticks(ticks, categoryNames,color='brown')
  s = [['TN','FP'], ['FN', 'TP']]
  
  for i in range(2):
      for j in range(2):
          plt.text(j,i, str(s[i][j])+" = "+str(confusion_matrix[i][j]),fontsize=16,color='white')
  plt.show()


# calculate the peformance of each classification methode with 9 different metrics

def performance(model):
    
    for name, model, X_train, y_train, X_test, y_test in model:
        
        #appending name
        names.append(name)
        
        # Build model
        model.fit(X_train, y_train)
        
        #predictions
        y_test_pred = model.predict(X_test)
        
        # calculate accuracy
        Accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
        accuracy_tests.append(Accuracy_test)
        
        # calculate auc
        Aucs_test = metrics.roc_auc_score(y_test , y_test_pred)
        aucs_tests.append(Aucs_test)
        
        #precision_calculation
        Precision_score_test = metrics.precision_score(y_test , y_test_pred)
        precision_tests.append(Precision_score_test)
        
        # calculate recall
        Recall_score_test = metrics.recall_score(y_test , y_test_pred)
        recall_tests.append(Recall_score_test)
        
        #calculating F1
        F1Score_test = metrics.f1_score(y_test , y_test_pred)
        f1_score_tests.append(F1Score_test)
        
        #calculating MCC
        MCC_score_tests = metrics.matthews_corrcoef(y_test, y_test_pred)
        mcc_score_tests.append(MCC_score_tests)
        
        # calculating balanced accuracy score
        Balanced_Accuracy_test = metrics.balanced_accuracy_score(y_test,y_test_pred)
        balanced_accuracy_tests.append(Balanced_Accuracy_test)
        
        # calculating top k accuracy
        
        Top_K_Accuracy_test = metrics.top_k_accuracy_score(y_test,y_test_pred)
        top_k_accuracy_tests.append(Top_K_Accuracy_test)

        #calculating cohenvkappa score
        Cohen_kappa_test = metrics.cohen_kappa_score(y_test,y_test_pred)
        cohen_kappa_tests.append(Cohen_kappa_test)
        
        # draw confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred) 
        
        print("Model Name :", name)
        print('Test Accuracy : {0:0.5f}'.format(Accuracy_test))
        print('Top K Accuracy Score : {0:0.5f}'.format(Top_K_Accuracy_test))
        print('Balanced Accuracy : {0:0.5f}'.format(Balanced_Accuracy_test))
        print('Test AUC : {0:0.5f}'.format(Aucs_test))
        print('Test Precision : {0:0.5f}'.format(Precision_score_test))
        print('Test Recall : {0:0.5f}'.format(Recall_score_test))
        print('Test F1 : {0:0.5f}'.format(F1Score_test))
        print("Matthews correlation coefficient : {0:0.5f}".format(MCC_score_tests))
        print('Cohen Kappa Score : {0:0.5f}'.format(Cohen_kappa_test))
        print('Confusion Matrix : \n',  confusion_matrix)
#         confusion_matrix_plot(y_test, y_test_pred)
       
        print("\n")
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred)
        threshold = thresholds[np.argmax(tpr-fpr)]
        print("threshold: ",threshold)
        auc = metrics.roc_auc_score(y_test, y_test_pred)
        plt.plot(fpr,tpr,linewidth=2, label=name + ", auc="+str(auc))
        
        
    plt.legend(loc=4)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    # plt.show()
    

time_result=[]
list_of_models = ['Logical Regression Classifier','Random Forest Classifier','Gaussian Naïve Bayes Classifier',
                 'Decision Tree Classifier','K-Nearest Neighbor Class','XG Boost Classifier','MLP Classifier'
                 ,'LinearSVC']
i = 0






LRmodel=[]
LRmodel.append(('LR IMBALANCED', LogisticRegression(solver='saga',multi_class='multinomial'),X_train, y_train, X_test, y_test))
LRmodel.append(('LR SMOTE', LogisticRegression(solver='saga',multi_class='multinomial'),X_train_smote, y_train_smote, X_test_smote, y_test_smote))


RFmodel = []
RFmodel.append(('RF IMABALANCED', RandomForestClassifier(),X_train,y_train,X_test,y_test))
RFmodel.append(('RF SMOTE', RandomForestClassifier(),X_train_smote, y_train_smote, X_test_smote, y_test_smote))


NBmodel = []
NBmodel.append(('NB IMBALANCED', GaussianNB(),X_train,y_train,X_test,y_test))
NBmodel.append(('NB SMOTE', GaussianNB(),X_train_smote, y_train_smote, X_test_smote, y_test_smote))


DTmodel = []
DTmodel.append(('DT IMBALANCED', DecisionTreeClassifier(),X_train,y_train,X_test,y_test))
DTmodel.append(('DT SMOTE', DecisionTreeClassifier(),X_train_smote, y_train_smote, X_test_smote, y_test_smote))

KNNmodel = []
KNNmodel.append(('KNN IMBALANCE', KNeighborsClassifier(),X_train,y_train,X_test,y_test))
KNNmodel.append(('KNN SMOTE', KNeighborsClassifier(),X_train_smote, y_train_smote, X_test_smote, y_test_smote))

xgBOOST=[]
xgBOOST.append(('XGBOOST IMBALANCED', XGBClassifier(n_estimators = 1000, verbosity = 1, scale_pos_weight = 580),X_train, y_train, X_test, y_test))
xgBOOST.append(('XGBOOST SMOTE', XGBClassifier(n_estimators = 1000, verbosity = 1, scale_pos_weight = 580),X_train_smote, y_train_smote, X_test_smote, y_test_smote))

MLPclassifier=[]
MLPclassifier.append(('MLPClassifier IMBALANCE', MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000),X_train,y_train,X_test,y_test))
MLPclassifier.append(('MLPClassifier SMOTE',MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000),X_train_smote, y_train_smote, X_test_smote, y_test_smote))

LSVC=[]
LSVC.append(('LinearSVC IMBALANCE', LinearSVC(random_state=0, tol=1e-5),X_train,y_train,X_test,y_test))
LSVC.append(('LinearSVC SMOTE',LinearSVC(random_state=0, tol=1e-5),X_train_smote, y_train_smote, X_test_smote, y_test_smote))
models = [LRmodel,RFmodel,NBmodel,DTmodel,KNNmodel,xgBOOST,MLPclassifier,LSVC]
   
def multiprocessing_func(x):
    time.sleep(1)
    performance(models[x])


a = False
from multiprocessing import Pool  # bypass the GIL by allowing multiprocessing
#https://docs.python.org/3/library/multiprocessing.html
models = [LRmodel,RFmodel,NBmodel,DTmodel,KNNmodel,xgBOOST,MLPclassifier,LSVC]




from multiprocessing import Pool  # bypass the GIL by allowing multiprocessing
#https://docs.python.org/3/library/multiprocessing.html
models = [LRmodel,RFmodel,NBmodel,DTmodel,KNNmodel,xgBOOST,MLPclassifier,LSVC]

def multiprocessing_func(x):
    time.sleep(2)
    performance(models[x])

a = False
if __name__ == '__main__':
    startTime = time.time()
    pool = Pool()
    pool.map(multiprocessing_func, range(0,8))
    pool.close()
    print('Pool Multi processing took {:10.2f} seconds or {:10.2f} minutes'.format((time.time() - startTime),(time.time() - startTime)/60))
    a = True
# COMPARE MCC SCORE FOR ALL DATASETS
time.sleep(2)
if(a):
    comparison={
    'Model': names,
    'Accuracy': accuracy_tests,
    'AUC': aucs_tests,
    'Precision Score' : precision_tests,
    'Recall Score': recall_tests, 
    'F1 Score': f1_score_tests,
    "Balanced Accuracy": balanced_accuracy_tests,
    "Top K Accuracy": top_k_accuracy_tests,
    "Cohen Kappa Score": cohen_kappa_tests,
    'MCC Score': mcc_score_tests,
}

    print(cl("Comparing performance of various Classifiers sorted by MCC Score : ",attrs=['bold'],color='blue'))
    comparison=pd.DataFrame(comparison)
    comparison.sort_values('MCC Score',ascending=False)


    # COMPARE Accuracy SCORE FOR ALL DATASETS
    comparison={
        'Model': names,
        'AUC': aucs_tests,
        'Precision Score' : precision_tests,
        'Recall Score': recall_tests, 
        'F1 Score': f1_score_tests,
        "Balanced Accuracy": balanced_accuracy_tests,
        "Top K Accuracy": top_k_accuracy_tests,
        "Cohen Kappa Score": cohen_kappa_tests,
        'MCC Score': mcc_score_tests,
        'Accuracy': accuracy_tests,
    }

    print(cl("Comparing performance of various Classifiers sorted by Accuracy : ",attrs=['bold'],color='blue'))
    comparison=pd.DataFrame(comparison)
    comparison.sort_values('Accuracy',ascending=False)


    


    print(cl('#',color='green')*40 + '\n')
    print(cl('Time taken by the program is : {:10.2f}s or {:10.2f}min\n'.format((time.time() - total_time),(time.time() - total_time)/60),attrs = ['bold'],color='green'))
    print(cl('#',color='green')*40 + '\n')

