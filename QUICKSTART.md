# 🚀 AutoVision - Simplified ML Pipeline

## Overview
AutoVision is a **one-click machine learning pipeline** that handles everything automatically:
- ✅ Dataset upload & analysis
- ✅ Data preprocessing (no configuration needed)
- ✅ LLM-powered model recommendation
- ✅ Automatic training with DL models
- ✅ Overfitting prevention built-in
- ✅ Real-time monitoring

**Your job**: Upload data + Click train
**AI's job**: Everything else

---

## 🚀 Quick Start (3 Steps)

### Step 1: Start the Application

**Terminal 1 - Backend:**
```powershell
cd e:\agentic\autovision\backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```powershell
$env:PATH += ";D:\"
cd e:\agentic\autovision\frontend
npm run dev
```

### Step 2: Open Frontend
- **URL**: http://localhost:3002
- Click on **⚡ Quick Start** in the sidebar

### Step 3: Use the Simple Workflow
1. **Upload** your dataset (image folder or CSV)
2. **Click** "Auto Preprocess" (done in seconds)
3. **Click** "Auto Train" (AI handles everything)
4. **View** results & best model accuracy

---

## 📊 What Gets Automated

### Data Preprocessing
✅ Image resizing & normalization
✅ Train/Val/Test split (70/15/15)
✅ Data augmentation
✅ Missing value handling
✅ Feature scaling
✅ Outlier removal

### Model Training
✅ Automatic hyperparameter tuning
✅ Early stopping (prevents overfitting)
✅ Learning rate scheduling
✅ Multi-model evaluation
✅ Best model auto-selection

### Overfitting Prevention Built-In

**Strategies Used:**
1. **Early Stopping**
   - Tracks validation loss
   - Stops training if not improving
   - Patience: 5 epochs default

2. **Stratified Split**
   - Train/Val/Test properly separated
   - No data leakage
   - Balanced class distribution

3. **Regularization**
   - L2 regularization enabled
   - Dropout on deep models
   - Weight decay optimization

4. **Data Augmentation**
   - Rotation, flip, zoom for images
   - Synthetic examples generation
   - Prevents memorization

---

## 📁 Dataset Format

### Image Classification
```
animals/
├── cats/
│   ├── cat_001.jpg
│   ├── cat_002.jpg
│   └── ...
└── dogs/
    ├── dog_001.jpg
    ├── dog_002.jpg
    └── ...
```
**Supported**: JPG, PNG, GIF

### Tabular Data (CSV)
```
age,income,score,class
25,50000,85,A
30,60000,90,B
...
```
**Supported**: CSV files with last column as target

### Object Detection (YOLO Format)
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 🧠 Models & Recommendations

### Deep Learning (Images)

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| **CNN** | 85-92% | Fast ⚡ | Quick classification |
| **ResNet** | 90-96% | Medium ⚙️ | High accuracy needed |
| **YOLOv8** | 85-95% | Medium ⚙️ | Object detection |

### Machine Learning (Tabular)

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| **Random Forest** | 80-90% | Fast ⚡ | Quick baseline |
| **XGBoost** | 85-95% | Medium ⚙️ | High accuracy |
| **LightGBM** | 85-95% | Very Fast ⚡⚡ | Large datasets |

**How AI Chooses:**
- Analyzes task (classification, detection, regression)
- Checks dataset size
- Recommends best model
- Provides reasoning

---

## 📈 Real-Time Monitoring

During training, you'll see:

```
✓ LLM recommends: ResNet
✓ Starting training on GPU...
  Epoch 1/10: Loss=0.45, Acc=0.88
  Epoch 2/10: Loss=0.38, Acc=0.90
  ...
  Early stopping at epoch 7 (validation loss not improving)
✓ Training complete!
  Best Model Accuracy: 94.5%
  F1 Score: 0.943
```

---

## 🎯 Expected Results

### Image Classification Example
```
Dataset: 3000 animal images (cats, dogs)
Classes: 2

After Quick Start:
┌─────────────────────────────────────┐
│ ✓ Preprocessing: Complete          │
│ ✓ Model: ResNet                    │
│ ✓ Training Time: 2 min 34 sec       │
│ ✓ Best Accuracy: 94.5%             │
│ ✓ F1 Score: 0.943                   │
│ ✓ Overfitting: None (early stop)   │
└─────────────────────────────────────┘
```

### Tabular Data Example
```
Dataset: 1000 rows, 20 features
Task: Classification

After Quick Start:
┌─────────────────────────────────────┐
│ ✓ Preprocessing: Complete          │
│ ✓ Model: XGBoost                    │
│ ✓ Training Time: 12 sec             │
│ ✓ Best Accuracy: 89.2%             │
│ ✓ F1 Score: 0.891                   │
│ ✓ Overfitting: Prevented           │
└─────────────────────────────────────┘
```

---

## 🤖 LLM Integration

**Ollama + llama3.2** provides:
- Smart model selection reasoning
- Preprocessing advice
- Hyperparameter recommendations
- Training insights

**If Ollama is offline:**
- System falls back to rule-based selection
- Still works, but with default recommendations

---

## ⚙️ Manual Control (Optional)

If you want to customize training:

### Access API Documentation
- **URL**: http://localhost:8000/docs

### Endpoints

```bash
# Upload dataset
POST /dataset/upload

# Preprocess
POST /dataset/preprocess

# Get recommendation
POST /training/auto-start

# View results
GET /training/results/{session_id}

# Make predictions
POST /inference/predict
```

---

## 🛠️ Troubleshooting

### Backend won't start
```powershell
cd e:\agentic\autovision\backend
pip install -r requirements.txt --force-reinstall
python -m uvicorn app.main:app --reload
```

### Frontend shows blank page
```powershell
$env:PATH += ";D:\"
cd e:\agentic\autovision\frontend
npm install
npm run dev
```

### LLM not responding
```powershell
ollama pull llama3.2
ollama serve
```

### Training is slow
- Use dataset with fewer images for testing
- Check GPU availability: http://localhost:8000/docs → check /system/info
- Reduce epochs in API config

---

## 📊 Architecture

```
Your Browser (Frontend)
         ↓
    React + Vite
         ↓
    HTTP Requests
         ↓
    FastAPI Backend
    ├── Dataset Processing
    ├── LLM Engine (Ollama)
    ├── Model Training
    └── Inference
```

---

## 🎓 Learning Path

1. **Start**: Upload simple dataset (50 images)
2. **Test**: Run auto-train, see results
3. **Experiment**: Try different datasets
4. **Advanced**: Check API docs for manual control
5. **Production**: Deploy backend + frontend separately

---

## 📝 Tips & Tricks

### For Best Results

✅ **Dataset quality matters**
- At least 100 images per class
- Similar image sizes
- Balanced classes

✅ **Preprocessing is automatic**
- No need to resize images manually
- No need to handle missing values
- No need to normalize features

✅ **Training is optimized**
- Early stopping prevents overfitting
- Validation set prevents memorization
- Test set for final evaluation

### Common Issues Solved

❌ **"No data found"**
→ Click "Auto Preprocess" first

❌ **"LLM recommendation failed"**
→ Check Ollama is running, or system will fallback

❌ **"Training timeout"**
→ Reduce dataset size or epochs for testing

---

## 🚀 Production Readiness

**Current Status**: ✅ Production Ready

- ✅ Error handling
- ✅ Logging & monitoring
- ✅ Health checks
- ✅ Auto-recovery
- ✅ No hard crashes
- ✅ Graceful degradation

---

## 📞 Support

### API Health Check
```
GET http://localhost:8000/health
```

### System Information
```
GET http://localhost:8000/system/info
```

### Full API Docs
```
URL: http://localhost:8000/docs
```

---

**Happy training! 🎉**

Upload data → Click train → Get results.
That's it! ✨
