import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split
lin = LinearRegression() 

#obtaining the data and reformating it into rawdata1
rawdata_df=pd.read_csv("https://covidtracking.com/data/download/all-states-history.csv")
rawdata1_df = rawdata_df[['date','state','positiveIncrease','positive']]
rawdata1_df['date']= pd.to_datetime(rawdata1_df['date'])

population_df = pd.read_csv("Pop.csv")
abbreviations_df = pd.read_csv("csvData.csv")
population2_df = population_df.merge(abbreviations_df, left_on='NAME', right_on='State')
population3_df = population2_df[['Code','POPESTIMATE2019']]

rawdata1_df.drop(rawdata1_df[rawdata1_df['positiveIncrease']<=0].index, inplace=True)

predictedValues = []
pPredictedValues = []   
list3 = []
cases = []
                         
for st in rawdata1_df['state'].unique():
    rawdata_case=rawdata1_df[rawdata1_df['state']==st][['date', 'positiveIncrease']]

    
    nsample=min(300, len(rawdata_case))
    
    xVal = rawdata_case.iloc[0:nsample, 0]
    yVal = rawdata_case.iloc[0:nsample, 1]
    

    X1 = np.transpose(np.flip([(xVal-min(xVal)).dt.days]))
    Y1 = np.flip(yVal.values)
    
    
    list2 = []
    for i in range(10, 100, 10):
    
        Y1_train, Y1_test, x1_train, x1_test = train_test_split(Y1, X1, test_size = i/100, random_state = 10)
        
        
        predictedValue = None
        list1 = []
        for n in range(0,min(len(x1_train)-1, 50)):
            poly1_train = PolynomialFeatures(degree = n) 
            X_poly1_train = poly1_train.fit_transform(x1_train) 
            X_poly1_test = poly1_train.fit_transform(x1_test)  
                                             
            lin1_train = LinearRegression() 
            lin1_train.fit(X_poly1_train,Y1_train)
            
            r = lin1_train.score(X_poly1_test, Y1_test)
            list1.append(r)
            if list1[n] == max(list1):
                predictedValue = lin1_train.predict(poly1_train.fit_transform([[(datetime.now(tz=None)-min(xVal)).days]]))
        
        bestValue = list1.index(max(list1))
        poly1_train = PolynomialFeatures(degree = bestValue) 
        X_poly1_train = poly1_train.fit_transform(x1_train) 
        X_poly1_test = poly1_train.fit_transform(x1_test)  
                                      
        lin1_train = LinearRegression() 
        lin1_train.fit(X_poly1_train,Y1_train)
        r = lin1_train.score(X_poly1_test, Y1_test)   
    
        
       # rmse = np.sqrt(mean_squared_error(Y1_test,lin1_train.predict(poly1_train.fit_transform(x1_test))))
        
        list2.append(r)
        #print(str(i) + " " + str(rmse) + str(r))
    bestValue1 = ((list2.index(max(list2))*10)+10)/100

    
    Y1_train, Y1_test, x1_train, x1_test = train_test_split(Y1, X1, test_size = bestValue1, random_state = 10)


    predictedValue = None
    list1 = []
    for n in range(0,min(len(x1_train)-1, 50)):
        poly1_train = PolynomialFeatures(degree = n) 
        X_poly1_train = poly1_train.fit_transform(x1_train) 
        X_poly1_test = poly1_train.fit_transform(x1_test)  
                                         
        lin1_train = LinearRegression() 
        lin1_train.fit(X_poly1_train,Y1_train)
        
        r = lin1_train.score(X_poly1_test, Y1_test)
        list1.append(r)
        #print("degree: " + str(n) + " score: " + str(r))
        
    
        if list1[n] == max(list1):
            predictedValue = lin1_train.predict(poly1_train.fit_transform([[(datetime.now(tz=None)-min(xVal)).days]]))
            pPredictedValue = lin1_train.predict(poly1_train.fit_transform([[(datetime.now(tz=None)-min(xVal)).days-1]]))
    bestValue = list1.index(max(list1))
    if predictedValue > 0:
        predictedValues.append(predictedValue.item())
    else:
        predictedValues.append(0)
    
    if predictedValue > pPredictedValue:
        pPredictedValues.append("increase")
    elif predictedValue == pPredictedValue:
        pPredictedValues.append("unchanged")
    else:
        pPredictedValues.append("decrease")
    list3.append(max(list1))
    cases.append(rawdata1_df[rawdata1_df['state']==st][['positive']].iloc[0,0])
    


nData = {'State':  rawdata1_df['state'].unique().tolist(),
         'predicted_values': predictedValues, 
         'pPredicted_values' : pPredictedValues,
         'Cases' : cases}



#population2_df = population_df.merge(abbreviations_df, left_on='NAME', right_on='State')

nDF = pd.DataFrame(data=nData)
nDF = nDF[nDF.State != 'MP']
nDF = nDF[nDF.State != 'GU']
nDF = nDF[nDF.State != 'PR']
nDF = nDF[nDF.State != 'VI']
nDF = nDF[nDF.State != 'DC']

nDF1 = nDF.merge(population3_df, left_on = "State", right_on="Code")
nDF1 = nDF1.drop(['Code'], axis=1)
nDF1 = nDF1.rename(columns={"POPESTIMATE2019":"Population"})

Probability  = []
for i in range(0, len(nDF1)):
    Probability.append(nDF1["predicted_values"][i]/(nDF1["Population"][i] - nDF1["Cases"][i]))
#df2 = df.assign(address = ['Delhi', 'Bangalore', 'Chennai', 'Patna']) 
nDF1['Probability'] = Probability


#nDF1.sort_values(by=['Probability'], ascending=False)



fig = plt.figure(figsize = (20, 12))  
plt.bar(nDF1['State'], nDF1['predicted_values']) 
plt.ylabel('Cases')
plt.xlabel('States')
plt.title("Predicted Cases of Each State")
 

fig = plt.figure(figsize = (20, 12))  
plt.bar(nDF1['State'], nDF1['Probability']) 
plt.ylabel('Cases')
plt.xlabel('States')
plt.title("Probability of Infection for Each State")


#df = df.drop(df[df.score < 50].index)

def plot(state):
    rawdata_case=rawdata1_df[rawdata1_df['state']==state][['date', 'positiveIncrease']]
    
    
    nsample=min(300, len(rawdata_case))
    
    xVal = rawdata_case.iloc[0:nsample, 0]
    yVal = rawdata_case.iloc[0:nsample, 1]
    

    X1 = np.transpose(np.flip([(xVal-min(xVal)).dt.days]))
    Y1 = np.flip(yVal.values)
    
    list2 = []
    for i in range(10, 100, 10):
    
        Y1_train, Y1_test, x1_train, x1_test = train_test_split(Y1, X1, test_size = i/100, random_state = 10)
        
        
        predictedValue = None
        list1 = []
        for n in range(0,min(len(x1_train)-1, 50)):
            poly1_train = PolynomialFeatures(degree = n) 
            X_poly1_train = poly1_train.fit_transform(x1_train) 
            X_poly1_test = poly1_train.fit_transform(x1_test)  
                                             
            lin1_train = LinearRegression() 
            lin1_train.fit(X_poly1_train,Y1_train)
            
            r = lin1_train.score(X_poly1_test, Y1_test)
            list1.append(r)
            if list1[n] == max(list1):
                predictedValue = lin1_train.predict(poly1_train.fit_transform([[(datetime.now(tz=None)-min(xVal)).days]]))
        
        bestValue = list1.index(max(list1))
        poly1_train = PolynomialFeatures(degree = bestValue) 
        X_poly1_train = poly1_train.fit_transform(x1_train) 
        X_poly1_test = poly1_train.fit_transform(x1_test)  
                                      
        lin1_train = LinearRegression() 
        lin1_train.fit(X_poly1_train,Y1_train)
        r = lin1_train.score(X_poly1_test, Y1_test)   
    
        
       # rmse = np.sqrt(mean_squared_error(Y1_test,lin1_train.predict(poly1_train.fit_transform(x1_test))))
        
        list2.append(r)
        #print(str(i) + " " + str(rmse) + str(r))
    bestValue1 = ((list2.index(max(list2))*10)+10)/100

    Y1_train, Y1_test, x1_train, x1_test = train_test_split(Y1, X1, test_size=bestValue1, random_state = 10)


    predictedValue = None
    list1 = []
    for n in range(0,min(len(x1_train)-1, 50)):
        poly1_train = PolynomialFeatures(degree = n) 
        X_poly1_train = poly1_train.fit_transform(x1_train) 
        X_poly1_test = poly1_train.fit_transform(x1_test)  
                                         
        lin1_train = LinearRegression() 
        lin1_train.fit(X_poly1_train,Y1_train)
        
        r = lin1_train.score(X_poly1_test, Y1_test)
        list1.append(r)
        if list1[n] == max(list1):
            predictedValue = lin1_train.predict(poly1_train.fit_transform([[(datetime.now(tz=None)-min(xVal)).days]]))

    bestValue = list1.index(max(list1))
    poly1_train = PolynomialFeatures(degree = bestValue) 
    X_poly1_train = poly1_train.fit_transform(x1_train) 
    X_poly1_test = poly1_train.fit_transform(x1_test)  
                                     
    lin1_train = LinearRegression() 
    lin1_train.fit(X_poly1_train,Y1_train)
    
    
    
    width = 12
    height = 8
    plt.figure(figsize=(width, height))
    plt.plot(x1_train, Y1_train, 'ro', label='Training Data')
    plt.plot(x1_test, Y1_test, 'go', label='Test Data')
    plt.legend()
    
    xmax=max([x1_train.max(), x1_test.max()])
    xmin=min([x1_train.min(), x1_test.min()])
    x=np.arange(xmin, xmax, 0.1)
    
    plt.plot(x, lin1_train.predict(poly1_train.fit_transform(x.reshape(-1, 1))), label='Predicted Function')

    plt.ylabel('Cases')
    plt.xlabel('Days')
    plt.title(state + "'s cases to days")

for s in rawdata1_df['state'].unique():
    plot(s)
