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
# Training the model

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

# %% [markdown]
# Applying the model to make predictions

# %%
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

y_lr_train_pred

# %%
y_lr_test_pred


