# Project: Airfoil noise with a curved frequency term 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_absolute_error 
# STEP 1: Load the data file 
df = pd.read_csv("AirfoilSelfNoise.csv") 
print("Rows and columns:", df.shape) 
# STEP 2: Clean the data 
import numpy as np
 # Noise first rises then falls with frequency -> add log_f squared so the line can bend 
df["log_f"] = np.log10(df["f"]) 
df["log_f_sq"] = df["log_f"] ** 2 
df["log_delta"] = np.log10(df["delta"]) 
df = df.dropna(subset=["log_f", "log_f_sq", "log_delta", "SSPL"])
 # STEP 3: Choose inputs X (several columns) and output y 
inputs = ["log_f", "log_f_sq", "log_delta"]
X = df[inputs] 
y = df["SSPL"] 
# STEP 4: Split: 80% train, 20% test 
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)
 # STEP 5: Train 
model = LinearRegression() 
model.fit(X_train, y_train) 
# STEP 6: Test 
y_pred = model.predict(X_test) 
print("R2 score     :", round(r2_score(y_test, y_pred), 3)) 
print("Average error:", round(mean_absolute_error(y_test, y_pred), 2), "dB") 
print("Effect of each input (slope):") 
for name, m in zip(inputs, model.coef_):    
  print("  ", name, "->", round(m, 4)) 
  # STEP 7: Predict a new case and draw Actual vs Predicted 
  new = pd.DataFrame({    "log_f": [3.3], "log_f_sq": [10.89], "log_delta": [-2.3]}) 
  print("Predicted noise:", round(model.predict(new)[0], 2), "dB") 
  plt.scatter(y_test, y_pred, s=12, alpha=0.6) 
  low, high = y_test.min(), y_test.max() 
  plt.plot([low, high], [low, high], color="red", label="Perfect prediction") 
  plt.xlabel("Actual noise (dB)") 
  plt.ylabel("Predicted noise (dB)") 
  plt.title("Airfoil noise with a curved frequency term")
  plt.legend() 
  plt.show()