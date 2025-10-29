# Bankruptcy Predictor API

A simple web application built with FastAPI to predict company bankruptcy based on several risk factors. It uses an ensemble of three machine learning models (Logistic Regression, Random Forest, and Gaussian Naive Bayes) and combines their outputs using a hard-voting system.

## Features

- **FastAPI Backend**: A modern, high-performance web framework for building APIs.
- **Machine Learning Integration**: Serves predictions from three pre-trained `scikit-learn` models.
- **Voting Ensemble**: Combines model outputs to provide a single, more robust prediction.
- **Simple Web Interface**: An intuitive UI built with HTML and Jinja2 for user input.

## Project Structure

```
/Bankruptcy Predictor
|-- app.py              # FastAPI application logic
|-- requirements.txt    # Project dependencies
|-- mai.ipynb.          # Jupyter file of the Ml models creation
|-- Bankruptcy.csv      # Dataset
|-- /models             # Directory for saved ML models
|   |-- logistic_regression_model.pkl
|   |-- random_forest_model.pkl
|   |-- gaussian_nb_model.pkl
|   `-- label_encoder.pkl
|-- /templates
|   `-- index.html      # Frontend HTML template
`-- README.md           # This file
```

## Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
# If your project is in a git repository
git clone <https://github.com/Dineshkomara/Bankruptcy-Predictor.git>
cd Bankruptcy Predictor
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project-specific dependencies.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# Or on Windows
# venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Running the Application

Start the web server using `uvicorn`. The `--reload` flag will automatically restart the server when you make code changes.

```bash
uvicorn app:app --reload
```

Once the server is running, open your web browser and navigate to **http://127.0.0.1:8000**.
