### DEVELOPED BY : KULASEKARAPANDIAN K
### REGISTER NO : 212222240052
#### Date: 

# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

## AIM:
To implement the ARMA model using Python for the ETH15M dataset.

## ALGORITHM:
1. Import necessary libraries.
2. Load the dataset and select the 'high' price column.
3. Simulate an ARMA(1,1) process and plot the time series.
4. Display the Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots for ARMA(1,1).
5. Simulate an ARMA(2,2) process and plot the time series.
6. Display the ACF and PACF plots for ARMA(2,2).

## PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('ETH15M.csv')

# Use the 'high' price column for modeling
model = data['high'].dropna()

# Set figure size for plots
plt.rcParams['figure.figsize'] = [10, 7.5]

# Simulate ARMA(1,1) Process
ar1 = np.array([1, 0.33])  # AR(1) coefficient
ma1 = np.array([1, 0.9])   # MA(1) coefficient
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=len(model))

# Plot ARMA(1,1) process
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process for ETH15M')
plt.xlim([0, len(model)])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1, lags=40)
plt.title('ACF for ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1, lags=40)
plt.title('PACF for ARMA(1,1)')
plt.show()

# Simulate ARMA(2,2) Process
ar2 = np.array([1, 0.33, 0.5])  # AR(2) coefficients
ma2 = np.array([1, 0.9, 0.3])   # MA(2) coefficients
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=len(model))

# Plot ARMA(2,2) process
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process for ETH15M')
plt.xlim([0, len(model)])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2, lags=40)
plt.title('ACF for ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2, lags=40)
plt.title('PACF for ARMA(2,2)')
plt.show()

```

## OUTPUT :
#### SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/user-attachments/assets/b841598d-ffb0-4617-b9e6-8d2a2793d91b)

#### Partial Autocorrelation:
![image](https://github.com/user-attachments/assets/1aa77757-6579-414c-94a4-d05cb2b6960e)

#### Autocorrelation:
![image](https://github.com/user-attachments/assets/77f77af4-4faa-4fb5-b384-8336ad288c3e)

#### SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/b36165c8-93f3-4121-84c3-04af081eb264)

#### Autocorrelation :
![image](https://github.com/user-attachments/assets/649ded56-c14e-4107-9fb9-5e12954b3dd3)

#### Partial Autocorrelation : 
![image](https://github.com/user-attachments/assets/df56142b-b394-4158-8a96-63d77515e726)


## RESULT:
Thus, a python program is created to fit ARMA Model successfully.

