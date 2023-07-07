# %% [markdown]
# # **My first ML model**
# # Load data

# %%
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
df

# %% [markdown]
# # Data preparation
# ## Data separation as x and y

# %%
y = df['logS']
print("y values")
print(y)

x = df.drop('logS',axis=1)
print("x values")
print(x)

# %% [markdown]
# ## Data splitting
# Download scikit: `pip3 install -U scikit-learn`

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)

# %%
X_train

# %%
X_test

# %% [markdown]
# ### X_test contains 20% of the X data and the other 80% is in X_train

# %% [markdown]
# # **Model building**
# 
# ## Linear regression
# 
# ### Training the model

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

# %% [markdown]
# ### Applying the model to make predictions

# %%
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

y_lr_train_pred

# %%
y_lr_test_pred

# %% [markdown]
# ### **Evaluate model performance**

# %%
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print(f"LR train MSE: {lr_train_mse}")
print(f"LR train R2 score: {lr_train_r2}")

print(f"LR test MSE: {lr_test_mse}")
print(f"LR test R2 score: {lr_test_r2}")


# %%
lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
lr_results

# %% [markdown]
# ## Random forest
# 

# %% [markdown]
# ### Training the model

# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2,random_state=100)
rf.fit(X_train, y_train)

# %% [markdown]
# ### Applying the model to make a prediction

# %%
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# %% [markdown]
# ### Evaluate model performance

# %%
from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)


rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
rf_results

# %% [markdown]
# ## **Model comparison**

# %%
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

df_models

# %%
df_models.iloc[0][2]

# %% [markdown]
# ## **Data visualization of predicition results**

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c='#1ebba3', alpha=0.3)
plt.title('Training set vs. LR trained prediction')
plt.xlabel('Experimental log(S)')
plt.ylabel('Predicted log(S)')

# trend line
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), 'blue')



