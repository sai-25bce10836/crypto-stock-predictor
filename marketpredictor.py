import yfinance as yf
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
symbol = input("Enter stock/crypto symbol (e.g.,BTC-USD, RELIANCE.NS): ")
start_date= input("Enter start date (YYYY-MM-DD): ")
end_date =input("Enter end date (YYYY-MM-DD): ")
data= yf.download(symbol, start=start_date, end=end_date)



if data.empty:
    print("no data found Please check symbol or date range")
    exit()
data["Return"] =data["Close"].pct_change()
data["Target"] =(data["Return"] > 0).astype(int)

data = data.dropna()
X=data[["Open", "High", "Low", "Close", "Volume"]]
y=data["Target"]


scaler=StandardScaler()
X=scaler.fit_transform(X)

#test
X_train,X_test, y_train, y_test = train_test_split(
    X, y,test_size=0.2, shuffle=False
)

#training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Using model for prediction
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test,y_pred)
print("\nModel Accuracy:",round(accuracy,2))

#Predict next day moment
latest_data = X[-1].reshape(1,-1)
prediction =model.predict(latest_data)
print("\nNext Day Prediction:")
if prediction[0] ==1:
    print("UP")
else:
    print("DOWN")