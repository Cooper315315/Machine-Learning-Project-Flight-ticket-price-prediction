import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math


# Import dataset
df = pd.read_csv('Cleaned_2018_Flights.csv')


#EDA
# Visualise correlation
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True)
sns.boxplot(data=df, x="Quarter", y="PricePerTicket")
sns.boxplot(data=df, x="MktCoupons", y="PricePerTicket")
sns.boxplot(data=df, x="ContiguousUSA", y="PricePerTicket")
sns.boxplot(data=df, x="NumTicketsOrdered", y="PricePerTicket")
sns.boxplot(data=df, x="AirlineCompany", y="PricePerTicket")
# Visualise relationship between "Miles" and "PricePerTicket"
plt.figure(figsize=(20,10))
sns.lineplot(data = United_Air, x = 'Miles', y='PricePerTicket')


# Preprocessing
# Featureing engineering
# 1) Remove unnecessary columns
df.drop(['Unnamed: 0', 'ItinID', 'MktID','Quarter','OriginWac', 'DestWac'], axis = 1, inplace=True)
# 2) Add features
# Cleaning the airports ranking data
app_df = pd.read_excel('Airport Rankings 2018.xlsx',header=2,usecols=[0,1]).iloc[:-3,:]
app_df2 = pd.read_excel('Airport Rankings 2018.xlsx',sheet_name=3,header=4,usecols=[0,1,2])
app_df2['Airport'] = app_df2['Airport'].apply(lambda x: x.split(',')[0])
app = app_df.merge(app_df2, how='left',left_on='Airport',right_on='Airport')
map_dict2 = { name:code for name,code in zip(app[app.Code.isnull()].Airport, ['ORD','JFK','IAH','DTW','LGA','IAD','DCA','MDW','USA'])} #inspect null values and hardcode
app['Code'] = [ code if code == code else map_dict2.get(name) for name,code in zip(app['Airport'],app['Code'])]
app.to_csv('airports_ranking.csv')

# Ranking the airports in Origin and Dest
app = pd.read_csv('airports_ranking.csv')
map_dict = {code:rank for code,rank in zip(app['Code'],app['2018 Rank'])}
df['Origin'] = df['Origin'].map(map_dict).fillna(100)
df['Dest'] = df['Dest'].map(map_dict).fillna(100)
#drop it since don't improve r2 score
df.drop(columns=['Origin','Dest'], inplace=True)

# Classifier flight tickets' class
def classes(cols):
    if cols <=400:
        return 1
    elif (cols>400) & (cols<=800):
        return 2
    else:
        return 3
df["classes"]=df["PricePerTicket"].apply(classes)

# Calculate "profit" for choosing competitors
df['profit'] = df['NumTicketsOrdered']*df['PricePerTicket'] 
# Searching for competitors
plt.figure(figsize=(20,10))
df.groupby('AirlineCompany')['profit'].sum().sort_values(ascending=False).plot.bar()
plt.ylabel("Profit")
plt.xticks(rotation=0)
# Main competitors
# 1) WN -- Southwest Airlines Co., 2) DL -- Delta Air Lines Inc., 3) UA -- United Air Lines Inc., 4) AA -- American Airlines Inc.

# Create dataframe for different competitors
delta_air = df[df['AirlineCompany'] == 'DL']
delta_air = delta_air.drop(columns = 'AirlineCompany')

Southwest_Airlines = df[df['AirlineCompany'] == 'WN']
Southwest_Airlines = Southwest_Airlines.drop(columns = 'AirlineCompany')

United_Air = df[df['AirlineCompany'] == 'UA']
United_Air = United_Air.drop(columns = 'AirlineCompany')

American_Airlines = df[df['AirlineCompany'] == 'AA']
American_Airlines = American_Airlines.drop(columns = 'AirlineCompany')



#Models creation and evaluation

                                    # Create linear regression model for different airlines 
# Southwest Airlines
SW_y = Southwest_Airlines['PricePerTicket']
SW_X = Southwest_Airlines.drop(columns='PricePerTicket')
SW_X_train, SW_X_test, SW_y_train, SW_y_test = train_test_split(SW_X,SW_y,test_size=0.2,random_state=42 )
lr = LinearRegression()
lr.fit(SW_X_train, SW_y_train)
SW_y_pred = lr.predict(SW_X_test)
print('Coefficients: \n', lr.coef_)
print(f"Mean squared error:{mean_squared_error(SW_y_test, SW_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(SW_y_test, SW_y_pred)) :.2f}")
print(f'Variance score: {r2_score(SW_y_test, SW_y_pred):.2f}')
print("mean price tickets: {}".format(SW_y_pred.mean()))
# Lasso
from sklearn.linear_model import Lasso
SW_X_train, SW_X_test, SW_y_train, SW_y_test = train_test_split(SW_X,SW_y,test_size=0.2,random_state=42 )
regr = Lasso(alpha=0.5)
regr.fit(SW_X_train, SW_y_train)
SW_y_pred = regr.predict(SW_X_test)
print('Coefficients: \n', regr.coef_)
print(f"Mean squared error:{mean_squared_error(SW_y_test, SW_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(SW_y_test, SW_y_pred)) :.2f}")
print(f'Variance score: {r2_score(SW_y_test, SW_y_pred):.2f}')
# GridSearchCV
parameters = {'alpha': np.linspace(0.1,1.0,10) }
grid_search = GridSearchCV(estimator=regr, param_grid=parameters,  scoring="neg_mean_squared_error", cv = 10, n_jobs=-1)
Grid_search = grid_search.fit(SW_X_train, SW_y_train)
print(grid_search.best_score_) 
print(grid_search.best_params_)
SW_X_train, SW_X_test, SW_y_train, SW_y_test = train_test_split(SW_X,SW_y,test_size=0.2,random_state=42 )
regr = Lasso(alpha=0.1)
regr.fit(SW_X_train, SW_y_train)
SW_y_pred = regr.predict(SW_X_test)
print('Coefficients: \n', regr.coef_)
print(f"Mean squared error:{mean_squared_error(SW_y_test, SW_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(SW_y_test, SW_y_pred)) :.2f}")
print(f'Variance score: {r2_score(SW_y_test, SW_y_pred):.2f}')

# United Air
UA_y = United_Air['PricePerTicket']
UA_X = United_Air.drop(columns='PricePerTicket')
UA_X_train, UA_X_test, UA_y_train, UA_y_test = train_test_split(UA_X,UA_y,test_size=0.2,random_state=42 )
lr = LinearRegression()
lr.fit(UA_X_train, UA_y_train)
UA_y_pred = lr.predict(UA_X_test)
print('Coefficients: \n', lr.coef_)
print(f"Mean squared error:{mean_squared_error(UA_y_test, UA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(UA_y_test, UA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(UA_y_test, UA_y_pred):.2f}')
print("mean price tickets: {}".format(UA_y_pred.mean()))
# Lasso
UA_X_train, UA_X_test, UA_y_train, UA_y_test = train_test_split(UA_X,UA_y,test_size=0.2,random_state=42 )
regr = Lasso(alpha=0.5)
regr.fit(UA_X_train, UA_y_train)
UA_y_pred = regr.predict(UA_X_test)
print('Coefficients: \n', regr.coef_)
print(f"Mean squared error:{mean_squared_error(UA_y_test, UA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(UA_y_test, UA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(UA_y_test, UA_y_pred):.2f}')
#GridSearch
parameters = {'alpha': np.linspace(0.1,1.0,10) }
grid_search = GridSearchCV(estimator=regr, param_grid=parameters,  scoring="neg_mean_squared_error", cv = 10, n_jobs=-1)
Grid_search = grid_search.fit(UA_X_train, UA_y_train)
print(grid_search.best_score_) 
print(grid_search.best_params_)
UA_X_train, UA_X_test, UA_y_train, UA_y_test = train_test_split(UA_X,UA_y,test_size=0.2,random_state=42 )
regr = Lasso(alpha=0.1)
regr.fit(UA_X_train, UA_y_train)
UA_y_pred = regr.predict(UA_X_test)
print('Coefficients: \n', regr.coef_)
print(f"Mean squared error:{mean_squared_error(UA_y_test, UA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(UA_y_test, UA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(UA_y_test, UA_y_pred):.2f}')


# Delta Air
DA_y = delta_air['PricePerTicket']
DA_X = delta_air.drop(columns='PricePerTicket')
DA_X_train, DA_X_test, DA_y_train, DA_y_test = train_test_split(DA_X,DA_y,test_size=0.2,random_state=42 )
lr = LinearRegression()
lr.fit(DA_X_train, DA_y_train)
DA_y_pred = lr.predict(DA_X_test)
print('Coefficients: \n', lr.coef_)
print(f"Mean squared error:{mean_squared_error(DA_y_test, DA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(DA_y_test, DA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(DA_y_test, DA_y_pred):.2f}')
print("mean price tickets: {}".format(DA_y_pred.mean()))
#Lasso
DA_X_train, DA_X_test, DA_y_train, DA_y_test = train_test_split(DA_X,DA_y,test_size=0.2,random_state=42 )
regr = Lasso(alpha=0.5)
regr.fit(DA_X_train, DA_y_train)
DA_y_pred = regr.predict(DA_X_test)
print('Coefficients: \n', regr.coef_)
print(f"Mean squared error:{mean_squared_error(DA_y_test, DA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(DA_y_test, DA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(DA_y_test, DA_y_pred):.2f}')
#GridSearch
parameters = {'alpha': np.linspace(0.1,1.0,10) }
grid_search = GridSearchCV(estimator=regr, param_grid=parameters,  scoring="neg_mean_squared_error", cv = 10, n_jobs=-1)
Grid_search = grid_search.fit(DA_X_train, DA_y_train)
print(grid_search.best_score_) 
print(grid_search.best_params_)
DA_X_train, DA_X_test, DA_y_train, DA_y_test = train_test_split(DA_X,DA_y,test_size=0.2,random_state=42 )
regr = Lasso(alpha=0.1)
regr.fit(DA_X_train, DA_y_train)
DA_y_pred = regr.predict(DA_X_test)
print('Coefficients: \n', regr.coef_)
print(f"Mean squared error:{mean_squared_error(DA_y_test, DA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(DA_y_test, DA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(DA_y_test, DA_y_pred):.2f}')



# American Airlines
AA_y = American_Airlines['PricePerTicket']
AA_X = American_Airlines.drop(columns='PricePerTicket')
AA_X_train, AA_X_test, AA_y_train, AA_y_test = train_test_split(AA_X,AA_y,test_size=0.2,random_state=42 )
lr = LinearRegression()
lr.fit(AA_X_train, AA_y_train)
AA_y_pred = lr.predict(AA_X_test)
print('Coefficients: \n', lr.coef_)
print(f"Mean squared error:{mean_squared_error(AA_y_test, AA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(AA_y_test, AA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(AA_y_test, AA_y_pred):.2f}')
print("mean price tickets: {}".format(AA_y_pred.mean()))
# Lasso
AA_X_train, AA_X_test, AA_y_train, AA_y_test = train_test_split(AA_X,AA_y,test_size=0.2,random_state=42 )
regr = Lasso(alpha=0.5)
regr.fit(AA_X_train, AA_y_train)
AA_y_pred = regr.predict(AA_X_test)
print('Coefficients: \n', regr.coef_)
print(f"Mean squared error:{mean_squared_error(AA_y_test, AA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(AA_y_test, AA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(AA_y_test, AA_y_pred):.2f}')
#GridSearch
parameters = {'alpha': np.linspace(0.1,1.0,10) }
grid_search = GridSearchCV(estimator=regr, param_grid=parameters,  scoring="neg_mean_squared_error", cv = 10, n_jobs=-1)
Grid_search = grid_search.fit(AA_X_train, AA_y_train)
print(grid_search.best_score_) 
print(grid_search.best_params_)
AA_X_train, AA_X_test, AA_y_train, AA_y_test = train_test_split(AA_X,AA_y,test_size=0.2,random_state=42 )
regr = Lasso(alpha=0.1)
regr.fit(AA_X_train, AA_y_train)
AA_y_pred = regr.predict(AA_X_test)
print('Coefficients: \n', regr.coef_)
print(f"Mean squared error:{mean_squared_error(AA_y_test, AA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(AA_y_test, AA_y_pred)) :.2f}")
print(f'Variance score: {r2_score(AA_y_test, AA_y_pred):.2f}')



                                    # Create XGBoost model for different airlines 

airlines=[Southwest_Airlines,delta_air,American_Airlines,United_Air]
Air=["Southwest Airlines","Delta Air", "American Airlines" "United Air"]

!pip install xgboost

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

for i in range(4):
    y=airlines[i].PricePerTicket
    X=airlines[i].drop("PricePerTicket", axis=1)
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42, shuffle=True)
    xg = XGBRegressor()
    xg.fit(X_train, y_train)
    xg_train_pred = xg.predict(X_train)
    xg_test_pred = xg.predict(X_test)
    print("--------------"+Air[i]+"----------------")
    print(f"train set r2 score: {r2_score(y_train, xg_train_pred) :.2f}")
    print(f"test set r2 score: {r2_score(y_test, xg_test_pred) :.2f}")

                                    # Create kNN model for different airlines (Failed, ran out of memory)

#Normalise data set for kNN model (try with one company)
from sklearn.preprocessing import Normalizer
x=American_Airlines[["Miles","ContiguousUSA","NumTicketsOrdered","classes","MktCoupons"]]
scaler = Normalizer()
American_Airlines[["Miles","ContiguousUSA","NumTicketsOrdered","classes","MktCoupons"]]=scaler.fit_transform(x)
American_Airlines

#build kNN model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
k_range = range(20,40)
r2_train = {}
r2_train_list = []
r2_test = {}
r2_test_list = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_pred=knn.predict(X_train)
    test_pred=knn.predict(X_test)
    r2_train[k] = r2_score(y_train, train_pred)
    r2_train_list.append(r2_score(y_train, train_pred))
    r2_test[k] = r2_score(y_test, test_pred)
    r2_test_list.append(r2_score(y_test, test_pred))
#plot k_range
import matplotlib.pyplot as plt
plt.plot(k_range, r2_test_list)


                                    # Create DecisionTree model for different airlines 

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

for i in range(4):
    print("--------------"+Air[i]+"----------------")
    #Split Data
    y=airlines[i].PricePerTicket
    X=airlines[i].drop("PricePerTicket", axis=1)
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42, shuffle=True)

    # Fit DecisionTreeRegressor
    regr_2 = DecisionTreeRegressor()
    regr_2.fit(X_train, y_train)
   
    # Prediction
    predictions=regr_2.predict(X_test)#Test Data
    predictionst=regr_2.predict(X_train)#Train Data
    
    print(f"R2 Score is on Training Set {r2_score(y_train, predictionst):.2f}")
    print(f"R2 Score is on Test Set {r2_score(y_test, predictions):.2f}")
    print(f"Average Price: {predictionst.mean():.2f}")
    print("---")
    
    #Hyperparameter Tuning (Max Depth)
    parameters={"max_depth":list(range(1,15))}
    from sklearn.model_selection import GridSearchCV
    search=GridSearchCV(regr_2, param_grid=parameters, cv=10,n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Search Best Score {search.best_score_:.2f}")
    print(f"Search Best Params {search.best_params_}")
    print("")

#Random Forest
from sklearn.ensemble import RandomForestRegressor

for i in range(4):
    print("--------------"+Air[i]+"----------------")
    #Split Data
    y=airlines[i].PricePerTicket
    X=airlines[i].drop("PricePerTicket", axis=1)
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42, shuffle=True)

    # Fit RandomForestRegressor #n_estimators=100, IS DEFAULT 
    rnd_clf=RandomForestRegressor(n_estimators=100, max_leaf_nodes=26, n_jobs=-1, random_state=42)
    rnd_clf.fit(X_train, y_train)

    # Prediction
    y_pred_rf = rnd_clf.predict(X_test)
    print(f"Accuracy Score: {rnd_clf.score(X_test, y_test):.2f}")
    print(f"Average Price: {y_pred_rf.mean():.2f}")
    print("")
