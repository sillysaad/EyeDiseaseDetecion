# 🚀 Eye Disease Classifier App - Quick Start

## ✅ What's Created

- **`app.py`** - Flask web server (backend)
- **`templates/index.html`** - Beautiful web interface
- **`eye_disease_model.pkl`** - Your trained model (auto-saved from notebook)

---

## 📝 How to Run

### Step 1: Install Flask (if not already installed)
```bash
pip install flask
```

### Step 2: Start the app
```bash
python app.py
```

You'll see:
```
Starting Eye Disease Classifier Web App...
Open http://localhost:5000 in your browser
```

### Step 3: Open in Browser
- Go to **http://localhost:5000** in your web browser
- Done! 🎉

---

## 📖 How to Use

The web app has **3 input methods**:

### 🖱️ **Drag & Drop**
- Drag an eye image onto the box
- Results appear in seconds

### 📂 **Browse File**
- Click "Choose Image" button
- Select from your computer

### 📍 **File Path**
- Paste full image path
- Click "Analyze"

---

## 📊 Output

Shows:
- ✅ **Predicted Disease** (large text)
- ✅ **Confidence %** (e.g., 98.5%)
- ✅ **Top 3 Predictions** (all diseases ranked)

---

## 🛑 Troubleshooting

**"Model not loaded"?**
- Make sure you ran the model export cell in `model.ipynb`
- Check that `eye_disease_model.pkl` exists in the folder

**Port 5000 already in use?**
- Edit `app.py` last line: change `port=5000` to `port=8000`

**Want to stop the app?**
- Press `Ctrl + C` in terminal

---

## 📦 Supported Image Formats
- JPG / JPEG
- PNG
- GIF
- WebP

---

## 🎯 That's It!
Your eye disease classifier is now live as a web app! 🎉
