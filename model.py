
#-------- Import all libraries and such ---------
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import fileinput
import os
import shutil
shutil.rmtree('graphs')
os.mkdir('graphs')

from sklearn import datasets, linear_model

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import keras
from keras.layers import Dense
from keras.models import Sequential

# READS DATA, ENUMERATES SEX CATAGORY OF DATASET W OPTION FOR NORMALISATION
def cleanAndImportData():
    f = open('abalone.data','r')
    filedata = f.read()
    f.close()
    newdata = filedata.replace('M', '0')
    newdata = newdata.replace('F', '1')
    newdata = newdata.replace('I', '2')
    f = open('abaloneSexNumerated.data','w')
    f.write(newdata)
    f.close()
    data = np.genfromtxt('abaloneSexNumerated.data', delimiter =',')
    names = ['Sex','Length','Diamater','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight','Rings']
    return names,data

#MAKES HEATMAP
def initalHeatmapAndPichart(names,data):
    corrmat = np.corrcoef(data.T)
    fig, ax = plt.subplots()
    im = ax.imshow(corrmat)
    ax.set_title("Correlation Heatmap")
    ax.set_xticks(np.arange(len(names)), labels=names)
    ax.set_yticks(np.arange(len(names)), labels=names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, round(corrmat[i, j],2), ha="center", va="center", color="w")
    plt.imshow(corrmat)
    plt.savefig('graphs/corrmat.png')
    plt.clf()
    
    #make pychart of age classifaction
    data = np.where(data[:, -1] >= 6, 1, 0)
    numEntries = np.shape(data)
    olderThan7 = int(np.sum(data))
    numEntries = int(numEntries[0])
    youngerThan7 = numEntries-olderThan7
    plt.pie(np.array([olderThan7, youngerThan7]), labels=np.array(['Older than 7','Younger than 7']), autopct='%.2f')
    plt.title('Pie Chart' )
    plt.savefig('graphs/AgeClassificationPieChat.png')
    plt.show()
    plt.clf()

    return

#CREATES SCATTERS AS REQUIRED BY PART1
def initalScatterPlots(names,data,features):
    firstFeature = data[:,features[0]]
    secondFeature = data[:,features[1]]
    output = data[:,-1]

    #make first scatterplot
    plt.scatter(firstFeature,output)
    plt.xlabel(names[features[0]])
    plt.ylabel(names[-1])
    plt.title(names[features[0]] + ' vs ' + names[-1] + ' Scatter Plot' )
    plt.savefig('graphs/Feature'+str(features[0])+'_Scatter.png')
    plt.show()
    plt.clf()

    #make second scatterplot
    plt.scatter(secondFeature,output)
    plt.xlabel(names[features[1]])
    plt.ylabel(names[-1])
    plt.title(names[features[1]] + ' vs ' + names[-1] + ' Scatter Plot' )
    plt.savefig('graphs/Feature'+str(features[1])+'_Scatter.png')
    plt.show()
    plt.clf()
    return

#CREATES HISTOGRAMAS AS REQUIRED BY PART1
def initalHistograms(names,data,features):
    firstFeature = data[:,features[0]]
    secondFeature = data[:,features[1]]

    #make first histogram
    plt.hist(firstFeature, bins=20)
    plt.xlabel(names[features[0]])
    plt.ylabel('Frequency')
    plt.title(names[features[0]] + ' Histogram' )
    plt.savefig('graphs/Feature'+str(features[0])+'_Histogram.png')
    plt.show()
    plt.clf()

    #make second histogram
    plt.hist(secondFeature,bins=20)
    plt.xlabel(names[features[1]])
    plt.ylabel('Frequency')
    plt.title(names[features[1]] + ' Histogram' )
    plt.savefig('graphs/Feature'+str(features[1])+'_Histogram.png')
    plt.show()
    plt.clf()
    return

#PERFORMS TEST TRAIN SPLIT AND MAKES AGE BINARY CLASSIFICATION
def trainTestSplit(data,normalise,splitSeed,classification):
    #turn ring age into binary classification for logistic regression
    if classification:
        # for logistic regression
        datay = np.where(data[:, -1] >= 6, 1, 0)
    else:
        # for linear regression
        datay = data[:, -1]
    datax = data[:,:-1]
    if normalise:
        scaler = MinMaxScaler()
        scaler.fit(datax)
        datax = scaler.transform(datax)
    xtrain,xtest,ytrain,ytest = train_test_split(datax,datay,test_size=0.4,random_state=splitSeed)

    
    return xtrain,xtest,ytrain,ytest

# TRAINS A LINEAR REGRESSION MODEL AND RETURNS METRICS
def LinReg(xtrain, xtest, ytrain, ytest, graph, name):
    # train&prediect
    model_linear = linear_model.LinearRegression()
    model_linear.fit(xtrain, ytrain)
    ypred = model_linear.predict(xtest)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    r2 = r2_score(ytest, ypred)
    if graph:
        # PLOT A SCATTER PLOT
        plt.figure()
        plt.scatter(ytest, ypred, alpha=0.5, color='blue')
        plt.xlabel('Actual Rings')
        plt.ylabel('Predicted Rings')
        plt.title(f'Actual vs Predicted Rings - {name}')
        plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--', lw=2)
        plt.savefig(f'graphs/LinearRegression_Prediction_{name}.png')
        plt.show()
        plt.clf()
        #CODE TO TREAT LINEAR REG AS 24 RING CALSSIFICAITON PROBLEM IF NEED LATER
        #print('Confusion',confusion_matrix(ytest,np.round(ypred)))
        #acc = accuracy_score(ytest,np.round(ypred))
        #print(acc, ' is accuracy_score')


    return rmse, r2,

# RUNS A LINEAR REGRESSION EXPERIMENT AND RETURNS METRIC AVG/STD
def RunLinRegExperiments(data, epochs, normalise, name):
    LinRegRMSE = np.zeros(epochs)
    LinRegRSQU = np.zeros(epochs)
    for i in range(epochs):
        # make graphs on last experiment
        graph = False
        if i == epochs - 1:
            graph = True
        xtrain, xtest, ytrain, ytest = trainTestSplit(data, normalise, splitSeed=i + 1,classification=False)
        rmse, r2 = LinReg(xtrain, xtest, ytrain, ytest, graph, name)
        LinRegRMSE[i] = rmse
        LinRegRSQU[i] = r2
    meanRMSE, stdRMSE = np.mean(LinRegRMSE), np.std(LinRegRMSE)
    meanRSQU, stdRSQU = np.mean(LinRegRSQU), np.std(LinRegRSQU)
    # print results
    print('For: ' + name)
    print('----------------------')
    print(f'RMSE:   Mean: {meanRMSE:.5f}, STD: {stdRMSE:.5f}')
    print(f'R²:     Mean: {meanRSQU:.5f}, STD: {stdRSQU:.5f}')
    print('----------------------')
    return
#TRAINS A LOG REG MODEL AND RETURNS METRICS
def LogReg(xtrain,xtest,ytrain,ytest,graph,name):
    #train&prediect
    regr = linear_model.LogisticRegression()
    regr.fit(xtrain, ytrain)
    ypred = regr.predict(xtest)
    ypredprob = regr.predict_proba(xtest)[:, -1]
    

    #metrics
    acc = accuracy_score(ytest, ypred)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    rsquared = r2_score(ytest, ypred)
    cm = confusion_matrix(ytest, ypred)
    auc = roc_auc_score(ytest,ypredprob)

    if graph==True:
        #THIS CAN BE USED TO PLOT A SCATTER PLOT OF THIS LOG MODEL but i took it out
        #graph ring age in ShellWeight (best feature) dimension
        #plt.scatter(xtrain[:,-1],ytrain,s=4)
        #plt.scatter(xtest[:,-1],ytest,s=4)
        #plt.xlabel('ShellWeight')
        #plt.ylabel('Age Classification:\n {0 if age<7, 1 if age>7}')
        #plt.annotate('RMSE: ' + str(round(rmse,3))+'\nR_2: ' + str(round(rsquared,3))+'\nAcc: ' + str(round(acc,3)), xy=(np.mean(xtrain), np.mean(ytrain)))
        #plt.savefig('graphs/RingAgeLinearReg.png')
        #plt.show()
        #plt.clf()

        lr_fpr, lr_tpr, _ = roc_curve(ytest, ypredprob)
        plt.title('ROC Curve')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='LogisticModel')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.annotate('AUC: ' + str(round(auc,3)),xy=(np.mean(lr_fpr), np.mean(lr_tpr)))
        plt.savefig('graphs/' + name+ 'ROCCurve.png')
        plt.clf()

    return auc,acc,rmse,rsquared

#RUNS AN LOGREG EXPEIMRENT AND RETURNS METRIC AVG/STD
def RunLogregExperiments(data,epochs,normalise,name):
    LogregRMSE,LogregRSQU,LogregACC,LogregAUC = np.zeros(epochs),np.zeros(epochs),np.zeros(epochs),np.zeros(epochs)
    for i in range(0,epochs):
        #make graphs on last experiment
        graph = False
        if i == epochs -1:
            graph = True
        xtrain,xtest,ytrain,ytest = trainTestSplit(data,normalise,splitSeed=i+1,classification=True)
        auc,acc,rmse,rsquared = LogReg(xtrain,xtest,ytrain,ytest,graph,name)
        LogregRMSE[i],LogregRSQU[i],LogregACC[i],LogregAUC[i] = rmse,rsquared,acc,auc

    #print results
    meanACC,stdACC = np.mean(LogregACC),np.std(LogregACC)
    meanAUC,stdAUC = np.mean(LogregAUC),np.std(LogregAUC)
    meanRMSE,stdRMSE = np.mean(LogregRMSE),np.std(LogregRMSE)
    meanRSQU,stdRSQU = np.mean(LogregRSQU),np.std(LogregRSQU)
    print('For: ' + name)
    print('----------------------')
    print('ACC:    Mean: ',round(meanACC,5),'STD: ',round(stdACC,5))
    print('AUC:    Mean: ',round(meanAUC,5),'STD: ',round(stdAUC,5))
    print('RMSE:   Mean: ',round(meanRMSE,5),'STD: ',round(stdRMSE,5))
    print('RSQU:   Mean: ',round(meanRSQU,5),'STD: ',round(stdRSQU,5))
    print('----------------------')
    return

#RUN NEURAL NETWROKS EXPEIRMNETS - here i have 2 ephocs inputs, one for the model training and one for the 30 runs of the expriment. Bit confusiog sorry
def NeuralNetworkExperiments(data,trainepochs,runepochs,normalise,name):
    NNACC = np.zeros(runepochs)
    for i in range(0,runepochs):
        print('Starting Experiment: ' + str(i))
        xtrain,xtest,ytrain,ytest = trainTestSplit(data,normalise,splitSeed=i+1,classification=True)
        
        #define optimiser
        optimizer = keras.optimizers.SGD(learning_rate=0.02,momentum=0.02)
        #define model
        model = Sequential()
        model.add(Dense(10, input_dim=8, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # fit model
        history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=trainepochs, verbose=0)
        
        # evaluate the model
        _, train_acc = model.evaluate(xtrain, ytrain, verbose=0)
        _, test_acc = model.evaluate(xtest, ytest, verbose=0)
        #print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

        NNACC[i] = test_acc

        #make accuracy plots
        # plot history
        #plt.plot(history.history['accuracy'], label='train')
        #plt.plot(history.history['val_accuracy'], label='test')
        #plt.legend()
        #plt.savefig('nodp.png')

    meanNNACC,stdNNAC = np.mean(NNACC),np.std(NNACC)
    print('For: ' + name)
    print('----------------------')
    print('AUC:    Mean: ',round(meanNNACC,5),'STD: ',round(stdNNAC,5))
    return 

def main():
    #DATA PROCESSING SECTION
    names,data = cleanAndImportData()
    initalHeatmapAndPichart(names,data)
    initalScatterPlots(names,data,features=(2,7))
    initalHistograms(names,data,features=(2,7))

    # MODELING SECTION
    print("Modelling Question1:")
    RunLinRegExperiments(data, epochs=30, normalise=False, name='LinReg_EX1')
    RunLogregExperiments(data, epochs=30, normalise=False, name='LogReg_EX1')
    print("Modelling Question2:")
    RunLinRegExperiments(data, epochs=30, normalise=True, name='LinReg_EX2')
    RunLogregExperiments(data, epochs=30, normalise=True, name='LogReg_EX2')

    # prepair data into 2 features
    twoFeatureData = np.transpose(np.vstack((data[:, 2], data[:, 7], data[:, -1])))
    print("Modelling Question3:")
    RunLinRegExperiments(twoFeatureData, epochs=30, normalise=False, name='LinReg_EX3')
    RunLogregExperiments(twoFeatureData, epochs=30, normalise=False, name='LogReg_EX3')

    #NEURALNETWORKTIMEBABY
    print("Modelling Question4:")
    NeuralNetworkExperiments(data,trainepochs=100,runepochs=5,normalise=True,name='NN_EX1')

if __name__ == "__main__":
    main()
