# Load in our essentials
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os


# Base Dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(BASE_DIR, 'youtube')

# Load the data and create the data variables
dataset = pd.read_excel(os.path.join(DIR_DATA, 'advertising.xlsx'))
X = dataset.iloc[:, 1:4] # Todas as linhas do arquivo e da coluna 1 até a 4 sem considerar a coluna 0
y = dataset['Sales']  # Coluna que eu quero prever
X_train, X_test, y_train, y_test = train_test_split(X, y) # Divisão do dataset em trienamento e teste

# Create and fit the model for prediction
lin = LinearRegression() # Instânciou o modelo de regressão linear
lin.fit(X_train, y_train) # Treinamento
y_pred = lin.predict(X_test) # Predição

# Create coefficients
coef = lin.coef_
components = pd.DataFrame(zip(X.columns, coef), columns=['component', 'value'])
components = components.append({'component': 'intercept',
                                'value': lin.intercept_}, ignore_index=True)
