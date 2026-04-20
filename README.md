# Avocado Price API

This project provides a machine learning API built with [FastAPI](https://fastapi.tiangolo.com/) that predicts the average price of an avocado based on various sales features. It utilizes a pre-trained XGBoost regression model.

## Features Used for Prediction
The API endpoint accepts the following parameters to make a prediction:
- `total_volume` (float): Total number of avocados sold
- `plus_4046` (float): Total number of avocados with PLU 4046 sold
- `plus_4225` (float): Total number of avocados with PLU 4225 sold
- `plus_4770` (float): Total number of avocados with PLU 4770 sold
- `total_bags` (float): Total number of bags sold
- `small_bags` (float): Total number of small bags sold
- `large_bags` (float): Total number of large bags sold
- `xlarge_bags` (float): Total number of extra-large bags sold
- `type` (str): Type of avocado (`conventional` or `organic`)
- `year` (int): Year of sale
- `region` (str): Region of the observation

## Project Structure
- `main.py`: The FastAPI application containing the `/root` GET endpoint for making predictions.
- `utils.py`: Contains data processing logic (`process_new`) and the scikit-learn preprocessing `FeatureUnion` pipeline to transform user inputs before passing them to the model.
- `Model_XGBoost.pkl`: The serialized XGBoost predictive model.
- `avocado.csv`: The dataset used for training and model evaluation.
- `notebook.ipynb` & `test.ipynb`: Jupyter notebooks for exploratory data analysis (EDA), model training, and testing.
- `requirements.txt`: Python package dependencies needed to run the API.

## How to Run Locally

1. Clone the repository and navigate to the project directory.

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI development server using `uvicorn`:
   ```bash
   uvicorn main:app --reload
   ```

4. Open your browser and navigate to `http://127.0.0.1:8000/docs` to test the API endpoint using the interactive Swagger UI.
