# AutoVision - Quick Start Guide

## 🚀 Running the Project

### Step 1: Start Backend
```powershell
cd e:\agentic\autovision\backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Start Frontend
```powershell
$env:PATH += ";D:\"
cd e:\agentic\autovision\frontend
npm run dev
```

### Step 3: Access Application
- **Frontend**: http://localhost:3002
- **API Docs**: http://localhost:8000/docs

---

## 📊 Simple Workflow (Your Use Case)

### ✅ What You Need to Do:

1. **Upload Dataset**
   - Click on "Dataset" page
   - Upload your image folder (cats/dogs, animals, etc.)
   - System auto-detects format and task type

2. **Click "Auto Preprocess"**
   - Done! No configuration needed
   - System handles:
     - Image resizing
     - Train/Val/Test split (70/15/15)
     - Data augmentation
     - Normalization

3. **Get LLM Recommendation**
   - Click "Get LLM Recommendation"
   - AI suggests best model for your data
   - Shows why (e.g., "CNN best for image classification")

4. **Auto Train**
   - Click "Auto Train Recommended Model"
   - Training runs automatically with:
     - Deep Learning models (CNN, ResNet)
     - Early stopping to prevent overfitting
     - Hyperparameter tuning from LLM
     - Real-time accuracy monitoring

5. **View Results**
   - Best model accuracy shown automatically
   - Confusion matrix and metrics
   - No manual intervention needed

---

## 🧠 Models Available (Auto-Selected by AI)

### For Image Classification:
- **CNN** - Fast, lightweight
- **ResNet** - Better accuracy, slightly slower
- **YOLOv8** - Object detection (if applicable)

### For Tabular Data:
- **Random Forest** - Quick, reliable
- **XGBoost** - Higher accuracy
- **LightGBM** - Very fast

### Automatic Overfitting Prevention:
✅ Early stopping (stops when validation loss stops improving)
✅ Train/Val/Test split (prevents data leakage)
✅ Regularization enabled by default
✅ Learning rate scheduling

---

## 📁 Dataset Format Supported

- **Image Folders**: `animals/cats/`, `animals/dogs/`
- **CSV Files**: Tabular data
- **COCO Format**: Object detection
- **YOLO Format**: Object detection

---

## ⚡ Tips

1. Start with small dataset (~100 images) to test
2. Use mixed validation set for robust evaluation
3. Training time depends on model size and data
4. Check API docs at http://localhost:8000/docs for all endpoints

---

## 🛠️ Troubleshooting

### Backend won't start
```powershell
cd e:\agentic\autovision\backend
pip install -r requirements.txt --force-reinstall
```

### Frontend not loading
```powershell
$env:PATH += ";D:\"
cd e:\agentic\autovision\frontend
npm install
npm run dev
```

### LLM not working
```powershell
ollama pull llama3.2
ollama serve
```

---

## 📊 Expected Output

After training:
```
✓ 3000 images · 2 classes
✓ Preprocessing: Done
✓ LLM Recommendation: ResNet (accuracy prediction: 94%)
✓ Training: 10 epochs
✓ Best Model Accuracy: 94.5%
✓ F1 Score: 0.943
```

**Your job**: Upload → Preprocess → Train
**My job**: Everything else! ✨
