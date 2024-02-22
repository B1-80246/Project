from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the decision tree model
with open('IPL_SCORE_dt_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    current_runs = float(request.form['current_runs'])
    wickets = float(request.form['wickets'])
    overs = float(request.form['overs'])
    striker_runs = float(request.form['striker_runs'])
    non_striker_runs = float(request.form['non_striker_runs'])

    # Make a prediction using the model
    prediction = model.predict([[current_runs, wickets, overs, striker_runs, non_striker_runs]])

    return render_template('index.html', prediction=prediction[0] - 70)


if __name__ == '__main__':
    app.run(debug=True)
