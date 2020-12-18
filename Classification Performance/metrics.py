import numpy as np

# accuracy function
def accuracy(y_test,y_pred):
    cmx = confusion_matrix(y_test, y_pred) # Compute the confusion matrix
    total = 0
    correct_class = 0
    
    # compute total
    for i in range(0,len(cmx)):
        for j in range(0,len(cmx[i])):
            total += cmx[i][j]
            
    # compute correct class
    for i,j in zip(range(0,len(cmx)),range(0,len(cmx))):
        correct_class += cmx[i][j]
        
    return round((correct_class/total)*100, 2) 

# testing cases
from sklearn.metrics import accuracy_score

# precision and recall
def PRperClass(y_true,y_pred,target_names):
    # Compute the confusion matrix
    cmx = confusion_matrix(y_true, y_pred)   
    # compute TP, FN, FP
    TP = []
    FN = []
    FP = []
    for i in range(len(cmx)):
        # ini fn and fp
        fn = 0
        fp = 0
        for j in range(len(cmx)):
            if i == j:
                TP.append(cmx[i][j])
            else:
                fn += cmx[i][j]
                fp += cmx[j][i]
        
        # add it into list
        FN.append(fn)
        FP.append(fp) 
    # once we have TP,FN,FP, we can compute precision, recall for each class
    precision = {}
    recall = {}  
    # add it into list
    for i in range(len(target_names)):
        precision[target_names[i]] = round(TP[i]/(TP[i]+FP[i]),2)
        recall[target_names[i]] = round(TP[i]/(TP[i]+FN[i]),2)
        
    # return TP,FN,FP
    return precision, recall

# micro and macro
def PRmicro(y_true, y_pred):
    # Compute the confusion matrix
    cmx = confusion_matrix(y_true, y_pred) 
    # ini tp,fn,fp
    tp = 0
    fp = 0
    fn = 0
    # iterative 
    for i in range(len(cmx)):
        for j in range(len(cmx)):
            if i == j:
                tp += cmx[i][j]
            else:
                fn += cmx[i][j]
                fp += cmx[j][i]
                
    precision = round(tp/(tp+fp),2)
    recall = round(tp/(tp+fn),2)
    
    return precision, recall

def PRmacro(y_true, y_pred, target_names):
    # Compute the confusion matrix
    cmx = confusion_matrix(y_true, y_pred) 
    
    # we need to call function from PRperClass
    precision, recall = PRperClass(y_true, y_pred, target_names)
    
    # iterative to get sum and find the mean
    precision_total = 0
    recall_total = 0
    k = 0
    for i in range(len(target_names)):
        precision_total += precision[target_names[i]]
        recall_total += recall[target_names[i]]
        k += 1
    
    return round(precision_total/k,2), round(recall_total/k,2)


# cost matrix..
# define y_test and y_pred
y_test = [0,0,0,0,0,0,0,0,0,0,0,0,0]
y_pred = [0,0,0,1,1,1,1,1,1,1,1,1,1]
print(confusion_matrix(y_test, y_pred))

def errorCost(y_test,y_pred, cost_matrix):
    cost_matrix = np.array([[cost_matrix['TP'],cost_matrix['FP']],
                   [cost_matrix['FN'],cost_matrix['TN']]])
    cost = 0
    cmx = confusion_matrix(y_test, y_pred) 
    for i in range(len(cmx)):
        for j in range(len(cmx)):
            cost += cmx[i][j]*cost_matrix[i][j]
    
    return cost

errorCost(y_test, y_pred, cost_matrix)
