
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.linear_model import LinearRegression
import numpy as np
matplotlib.use('Agg')

app = Flask(__name__)

# Load dataset
try:
    nutrition_data = pd.read_csv(r'U:\\aiml\\recomendation\\data\\indian_food.csv')
    print("csvLoaded Successfully!")
except Exception as e:
    print("Error loading csv:", e)
    nutrition_data = None
if nutrition_data is not None:
    print("Col in csv:", nutrition_data.columns)


if nutrition_data is not None:
    for col in ['calories', 'proteins', 'carbs', 'fats']:
        nutrition_data[col] = pd.to_numeric(nutrition_data[col], errors='coerce')


def train_calorie_regression():
    # fake set
    age_weight_data = np.array([
        [18, 50, 2000], [20, 60, 2200], [25, 70, 2500],
        [30, 80, 2700], [35, 90, 2900], [40, 100, 3100]
    ])
    
    X = age_weight_data[:, :2]  # Age & Weight
    y = age_weight_data[:, 2]   # Calorie Requirement

    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Train the regression model
calorie_model = train_calorie_regression()

# Helper func
def plot_macros(macros):
    labels = ['Protein', 'Carbs', 'Fats']
    sizes = [macros['protein'], macros['carbs'], macros['fats']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Macronutrient Distribution')
    plt.axis('equal')
    plt.savefig('static/macros.png')
    plt.close()

# calcullate caloric needs and macros using regression
def calculate_recommendations(user_data):
    # Predicting here
    predicted_calories = calorie_model.predict([[user_data['age'], user_data['weight']]])[0]
    if user_data['goal'] == 'muscle_gain':
        calories = predicted_calories * 1.2  
    elif user_data['goal'] == 'fat_loss':
        calories = predicted_calories * 0.8  
    else:
        calories = predicted_calories  

    # Calculate macronutrient distribution
    macros = {
        "protein": round(calories * 0.25 / 4, 2),  
        "carbs": round(calories * 0.50 / 4, 2),    
        "fats": round(calories * 0.25 / 9, 2)      
    }

  
    plot_macros(macros)
    
    return macros, round(calories, 2)


def get_meal_recommendations(calories, diet_preference, goal):
    if nutrition_data is None:
        return []
    if diet_preference == "vegetarian":
        food_data = nutrition_data[nutrition_data['diet'].str.lower() == 'vegetarian'].copy()
    else:
        food_data = nutrition_data[nutrition_data['diet'].str.lower() != 'vegetarian'].copy()

    food_data = food_data.dropna()
    if goal == "fat_loss":
        food_data = food_data[food_data['fats'] <= 15]
    elif goal == "muscle_gain":
        food_data = food_data[food_data['proteins'] >= 10]


    food_data['calories_diff'] = abs(food_data['calories'] - calories)
    recommendations = food_data.sort_values(by='calories_diff')

    # print first 5 using head
    return recommendations[['name', 'calories', 'proteins', 'carbs', 'fats']].head(5).to_dict(orient='records')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_data = {
            "age": int(request.form['age']),
            "weight": float(request.form['weight']),
            "goal": request.form['goal'],
            "diet": request.form['diet']
        }
        return redirect(url_for('recommendations', user_data=json.dumps(user_data)))
    return render_template('home.html')

@app.route('/recommendations')
def recommendations():
    user_data = json.loads(request.args.get('user_data'))
    macros, calories = calculate_recommendations(user_data)
    meal_suggestions = get_meal_recommendations(calories, user_data['diet'], user_data['goal'])
    
    return render_template('recommendations.html', macros=macros, meals=meal_suggestions, calories=calories)

if __name__ == '__main__':
    app.run(debug=True)
