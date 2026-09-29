# Run the app

## Requirements

- Python installed
- `eye_disease_model.pkl` in the project root

## Install and start

From the project directory, run:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open <https://localhost:5000>. The app uses a temporary development certificate; accept the browser warning for local testing. Stop the server with `Ctrl+C`.

## Use

Choose an image by dragging it into the drop area, browsing for a file, or capturing it with the camera. The app displays the predicted class, confidence, and top three predictions.

The Flask development server is for local testing, not production deployment. Only load model pickle files you trust.
