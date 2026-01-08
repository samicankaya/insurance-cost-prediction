import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

plt.ion() 

print(" Loading Dataset...")
try:
    data = pd.read_csv('insurance.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("ERROR: 'insurance.csv' file not found! Please place it in the same folder.")
    exit()

# Data Preprocessing
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data = pd.get_dummies(data, columns=['region'], drop_first=True)

print("\n Training Model (Polynomial Regression)...")

X = data.drop('charges', axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

print(" Preparing Plot...")
plt.figure(figsize=(10, 6))

sns.scatterplot(x='bmi', y='charges', hue='smoker', data=data, palette='coolwarm', alpha=0.6, edgecolor=None)

plt.title('Body Mass Index (BMI) and Effect of Smoking on Charges')
plt.xlabel('BMI')
plt.ylabel('Charges ($)')
plt.legend()
plt.draw()
plt.pause(0.1)

print(" Calculating Test Results...")
predictions = model.predict(X_test_poly)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"   -> Model Accuracy (R2): %{r2*100:.1f}")
print(f"   -> Mean Absolute Error: {mae:.2f} $")

print("\n  Generating Customer Scenario...")

# Random scenario variables
r_age = np.random.randint(18, 66)
r_sex = np.random.choice([0, 1]) 
r_bmi = round(np.random.uniform(16, 45), 2)
r_children = np.random.randint(0, 6)
r_smoker = np.random.choice([0, 1], p=[0.7, 0.3]) # 30% chance of being a smoker

regions = ['northeast', 'northwest', 'southeast', 'southwest']
selected_region = np.random.choice(regions)

r_nw = 1 if selected_region == 'northwest' else 0
r_se = 1 if selected_region == 'southeast' else 0
r_sw = 1 if selected_region == 'southwest' else 0

new_person = pd.DataFrame({
    'age': [r_age],
    'sex': [r_sex],
    'bmi': [r_bmi],
    'children': [r_children],
    'smoker': [r_smoker],
    'region_northwest': [r_nw],
    'region_southeast': [r_se],
    'region_southwest': [r_sw]
})

new_person_poly = poly.transform(new_person)
price_prediction = model.predict(new_person_poly)[0]

print("\n" + "="*50)
print("     I N S U R A N C E   Q U O T E")
print("="*50)
print(f"| {'FEATURE':<20} | {'VALUE':<23} |")
print("-" * 50)
print(f"| {'Age':<20} | {r_age:<23} |")
print(f"| {'Sex':<20} | {'Male' if r_sex==1 else 'Female':<23} |")
print(f"| {'BMI':<20} | {r_bmi:<23} |")
print(f"| {'Children':<20} | {r_children:<23} |")
print(f"| {'Smoker':<20} | {'YES' if r_smoker==1 else 'No':<23} |")
print(f"| {'Region':<20} | {selected_region.capitalize():<23} |")
print("-" * 50)
print(f"| {'ESTIMATED PRICE':<20} | {price_prediction:,.2f} $ {'':<11} |")
print("="*50)

print("\n(Plot is open. Press Enter to close...)")
input()
