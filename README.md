# Machine-Learning-Project


<h3>Goal: Flight ticket price prediction</h3>

Deployed models:
- Linear Regression.
- XGBoost
- Random Forest.
- Decision Tree.

<br/>
<h3>Original DataFrame:</h3>
<img width="500" alt="DF" src="https://user-images.githubusercontent.com/80112729/118389719-c5b32d80-b65d-11eb-88e2-5844d2d74b02.png">
<br/>


Column Descriptions:
1. Unnamed: drop this column (it's a duplicate index column)
2-3. ItinID & MktID: vaguely demonstrates the order in which tickets were ordered (lower ID #'s being ordered first)
4. MktCoupons: the number of coupons in the market for that flight
5. Quarter: 1, 2, 3, or 4, all of which are in 2018
6. Origin: the city out of which the flight begins
7. OriginWac: USA State/Territory World Area Code
8. Dest: the city out of which the flight begins
9. DestWac: USA State/Territory World Area Code
10. Miles: the number of miles traveled
11. ContiguousUSA: binary column -- (2) meaning flight is in the contiguous (48) USA states, and (1) meaning it is not (ie: Hawaii, Alaska, off-shore territories)
12. NumTicketsOrdered: number of tickets that were purchased by the user
13. Airline Company: the two-letter airline company code that the user used from start to finish (key codes below)
14. PricePerTicket: target prediction column

<h3>Check for correlations:</h3>
<br/>
<img width="500" alt="Correlation" src="https://user-images.githubusercontent.com/80112729/118389956-0fe8de80-b65f-11eb-843b-88d56d725ea5.png">
From the correlation heat map, it is observed that “Miles” is the most correlated feature to the target feature (price), the rest have very little influence on the target feature.
<br/>

<br/>
<h3>Relationship between price per ticket vs miles:</h3>
<img width="500" alt="Price vs Miles" src="https://user-images.githubusercontent.com/80112729/118390229-90f4a580-b660-11eb-9c05-11e18e34893e.png">
From the above lineplot, a mild linear relationship is observed between price per ticket and miles.
<br/>


<br/>
<h3>Main competitors:</h3>
<img width="500" alt="Miles" src="https://user-images.githubusercontent.com/80112729/118394569-0324b480-b678-11eb-898f-757341a0c443.png">
<br/>

Main competitors: 
1. WN -- Southwest Airlines Co.
2. DL -- Delta Air Lines Inc. 
3. AA -- American Airlines Inc.        
4. UA -- United Air Lines Inc.

Code: 
UA_y = United_Air['PricePerTicket']
UA_X = United_Air.drop(columns='PricePerTicket')

UA_X_train, UA_X_test, UA_y_train, UA_y_test = train_test_split(UA_X,UA_y,test_size=0.2,random_state=42 )

lr = LinearRegression()
lr.fit(UA_X_train, UA_y_train)

UA_y_pred = lr.predict(UA_X_test)

print('Coefficients: \n', lr.coef_)
# The mean squared error
print(f"Mean squared error:{mean_squared_error(UA_y_test, UA_y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(UA_y_test, UA_y_pred)) :.2f}")
# Explained variance score: 1 is perfect prediction
print(f'Variance score: {r2_score(UA_y_test, UA_y_pred):.2f}')
print("mean price tickets: {}".format(UA_y_pred.mean()))
