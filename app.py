from fastapi import FastAPI
import joblib
from collections import Counter
from fastapi import Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="ML Model API", description="A simple API for an ML model.", version="1.0.0")
try:
    Logistic_Regression_Model = joblib.load('models/logistic_regression_model.pkl')
    Random_Forest_Model = joblib.load('models/random_forest_model.pkl')
    Gaussian_NB_Model = joblib.load('models/gaussian_nb_model.pkl')
    Label_Encoder = joblib.load('models/label_encoder.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, 
                  industrial_risk: float = Form(...),
                  management_risk: float = Form(...),
                  financial_flexibility: float = Form(...),
                  credibility: float = Form(...),
                  competitiveness: float = Form(...),
                  operating_risk: float = Form(...)):
    # Ensure the order of features matches what your model was trained on
    input_data = [[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]]
    
    lr_prediction = Logistic_Regression_Model.predict(input_data)
    rf_prediction = Random_Forest_Model.predict(input_data)
    gnb_prediction = Gaussian_NB_Model.predict(input_data)

    # --- Voting System Logic ---
    # 1. Collect all numerical predictions
    all_predictions = [lr_prediction[0], rf_prediction[0], gnb_prediction[0]]

    # 2. Find the most common prediction (the "vote")
    voted_prediction = Counter(all_predictions).most_common(1)[0][0]

    # 3. Convert the final voted prediction back to its original label
    final_label = Label_Encoder.inverse_transform([voted_prediction])[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "final_result": f"Final Prediction (by Voting): {final_label}"
    })