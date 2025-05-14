from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
with open('random_forest_model.pkl', 'rb') as file:
  model = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    # Get form data
    form_data = request.form

    # Initialize all features with 0
    features = {
        'Has Table booking': 0,
        'Has Online delivery': 0,
        'Is delivering now': 0,
        'Switch to order menu': 0,
        'Cost_INR': 0,
        # Rating colors
        'Rating color_Dark Green': 0,
        'Rating color_Green': 0,
        'Rating color_Orange': 0,
        'Rating color_Red': 0,
        'Rating color_White': 0,
        'Rating color_Yellow': 0,
        # Rating text
        'Rating text_Average': 0,
        'Rating text_Excellent': 0,
        'Rating text_Good': 0,
        'Rating text_Not rated': 0,
        'Rating text_Poor': 0,
        'Rating text_Very Good': 0,
        # Price range
        'Price range_1': 0,
        'Price range_2': 0,
        'Price range_3': 0,
        'Price range_4': 0,
        # Cuisines (all initially 0)
        'Cuisine_Burger': 0,
        'Cuisine_Desserts': 0,
        'Cuisine_South Indian': 0,
        'Cuisine_European': 0,
        'Cuisine_Seafood': 0,
        'Cuisine_Cafe': 0,
        'Cuisine_Finger Food': 0,
        'Cuisine_Japanese': 0,
        'Cuisine_Street Food': 0,
        'Cuisine_Steak': 0,
        'Cuisine_Fast Food': 0,
        'Cuisine_Beverages': 0,
        'Cuisine_Bakery': 0,
        'Cuisine_North Indian': 0,
        'Cuisine_Asian': 0,
        'Cuisine_Mediterranean': 0,
        'Cuisine_Salad': 0,
        'Cuisine_Biryani': 0,
        'Cuisine_Mithai': 0,
        'Cuisine_Healthy Food': 0,
        'Cuisine_Chinese': 0,
        'Cuisine_Sushi': 0,
        'Cuisine_Indian': 0,
        'Cuisine_Pizza': 0,
        'Cuisine_Tea': 0,
        'Cuisine_Mexican': 0,
        'Cuisine_Mughlai': 0,
        'Cuisine_Sandwich': 0,
        'Cuisine_Lebanese': 0,
        'Cuisine_Ice Cream': 0,
        'Cuisine_Continental': 0,
        'Cuisine_Raw Meats': 0,
        'Cuisine_Italian': 0,
        'Cuisine_Thai': 0,
        'Cuisine_American': 0,
        'Num_Cuisines': 0
    }

    # Process binary features
    binary_features = [
        'Has Table booking', 'Has Online delivery', 'Is delivering now',
        'Switch to order menu'
    ]
    for feat in binary_features:
      features[feat] = 1 if form_data.get(feat) == 'yes' else 0

    # Process cost
    features['Cost_INR'] = float(form_data.get('Cost_INR', 0))

    # Process rating color (only one can be selected)
    rating_color = form_data.get('Rating_color')
    if rating_color:
      features[f'Rating color_{rating_color}'] = 1

    # Process rating text (only one can be selected)
    rating_text = form_data.get('Rating_text')
    if rating_text:
      features[f'Rating text_{rating_text}'] = 1

    # Process price range (only one can be selected)
    price_range = form_data.get('Price_range')
    if price_range:
      features[f'Price range_{price_range}'] = 1

    # Process cuisines (multiple can be selected)
    selected_cuisines = form_data.getlist('Cuisines')
    features['Num_Cuisines'] = len(selected_cuisines)
    for cuisine in selected_cuisines:
      features[f'Cuisine_{cuisine}'] = 1

    # Convert to array in correct order (must match training data order)
    feature_order = [
        'Has Table booking', 'Has Online delivery', 'Is delivering now',
        'Switch to order menu', 'Cost_INR', 'Rating color_Dark Green',
        'Rating color_Green', 'Rating color_Orange', 'Rating color_Red',
        'Rating color_White', 'Rating color_Yellow', 'Rating text_Average',
        'Rating text_Excellent', 'Rating text_Good', 'Rating text_Not rated',
        'Rating text_Poor', 'Rating text_Very Good', 'Price range_1',
        'Price range_2', 'Price range_3', 'Price range_4', 'Cuisine_Burger',
        'Cuisine_Desserts', 'Cuisine_South Indian', 'Cuisine_European',
        'Cuisine_Seafood', 'Cuisine_Cafe', 'Cuisine_Finger Food',
        'Cuisine_Japanese', 'Cuisine_Street Food', 'Cuisine_Steak',
        'Cuisine_Fast Food', 'Cuisine_Beverages', 'Cuisine_Bakery',
        'Cuisine_North Indian', 'Cuisine_Asian', 'Cuisine_Mediterranean',
        'Cuisine_Salad', 'Cuisine_Biryani', 'Cuisine_Mithai',
        'Cuisine_Healthy Food', 'Cuisine_Chinese', 'Cuisine_Sushi',
        'Cuisine_Indian', 'Cuisine_Pizza', 'Cuisine_Tea', 'Cuisine_Mexican',
        'Cuisine_Mughlai', 'Cuisine_Sandwich', 'Cuisine_Lebanese',
        'Cuisine_Ice Cream', 'Cuisine_Continental', 'Cuisine_Raw Meats',
        'Cuisine_Italian', 'Cuisine_Thai', 'Cuisine_American', 'Num_Cuisines'
    ]

    input_data = [features[col] for col in feature_order]

    # Make prediction
    prediction = model.predict([input_data])[0]

    return render_template('result.html', prediction=prediction)

  return render_template('index.html')


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)
