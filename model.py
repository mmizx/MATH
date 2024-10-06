#-------- Import all libraries and such ---------
import numpy as np 
import random
import matplotlib.pyplot as plt
import math
import fileinput

from sklearn import datasets, linear_model

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error, r2_score


#  This function makes a new dataset with Sex enumerated
def cleanAndImportData(normalise):
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
    if normalise:
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
    names = ['Sex','Length','Diamater','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight','Rings']
    return names,data

# This function creates a correlation heatmat, and scatter plot of 2 most correlated features
def initalHeatmap(names,data): 
    #MAKE HEATMAP
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
    plt.savefig('corrmat.png')
    plt.clf()
    return 

def initalScatterPlots(names,data,features):
    firstFeature = data[:,features[0]]
    secondFeature = data[:,features[1]]
    output = data[:,-1]
    
    #make first scatterplot
    plt.scatter(firstFeature,output)
    plt.xlabel(names[features[0]])
    plt.ylabel(names[-1])
    plt.title(names[features[0]] + ' vs ' + names[-1] + ' Scatter Plot' )
    plt.savefig('FirstFeatureScatter.png')
    plt.show()
    plt.clf()

    #make second scatterplot
    plt.scatter(secondFeature,output)
    plt.xlabel(names[features[1]])
    plt.ylabel(names[-1])
    plt.title(names[features[1]] + ' vs ' + names[-1] + ' Scatter Plot' )
    plt.savefig('SecondFeatureScatter.png')
    plt.show()
    plt.clf()
    return

def initalHistograms(names,data,features):
    firstFeature = data[:,features[0]]
    secondFeature = data[:,features[1]]
    
    #make first histogram
    plt.hist(firstFeature)
    plt.xlabel(names[features[0]])
    plt.ylabel('Frequency')
    plt.title(names[features[0]] + ' Histogram' )
    plt.savefig('FirstHistogram.png')
    plt.show()
    plt.clf()

    #make second histogram
    plt.hist(secondFeature)
    plt.xlabel(names[features[1]])
    plt.ylabel('Frequency')
    plt.title(names[features[1]] + ' Histogram' )
    plt.savefig('SecondHistogram.png')
    plt.show()
    plt.clf()
    return

def trainTestSplit(data,splitSeed):
    datay = data[:,-1]
    datax = data[:,:-1]
    xtrain,xtest,ytrain,ytest = train_test_split(datax,datay,test_size=0.4,random_state=splitSeed)
    return xtrain,xtest,ytrain,ytest

def main():
    #DATA PROCESSING SECTION
    names,data = cleanAndImportData(normalise=True)
    initalHeatmap(names,data) 
    initalScatterPlots(names,data,features=(0,7)) 
    initalHistograms(names,data,features=(0,7)) 
    xtrain,xtest,ytrain,ytest = trainTestSplit(data,splitSeed=1)


if __name__ == "__main__":
    main()
