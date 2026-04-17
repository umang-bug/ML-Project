# 🚀 Deployment Guide — Student Financial Profiler

---

## Step 1: Folder Structure

Make sure your project folder looks like this:

```
your-repo/
│
├── app.py                  ← Streamlit app (provided)
├── train_and_save.py       ← Run this ONCE locally
├── requirements.txt        ← For Streamlit Cloud
├── Response.csv            ← Your survey data (keep locally, DON'T push to GitHub)
│
└── models/                 ← Generated after running train_and_save.py
    ├── spend_nn_model.h5
    ├── ordinal_encoder.joblib
    ├── ohe_encoder.joblib
    ├── mms_scaler.joblib
    ├── kproto_model.joblib
    ├── kproto_cat_indices.joblib
    ├── kproto_feature_columns.joblib
    ├── rf_classifier.joblib
    ├── scaler_rf.joblib
    ├── rf_feature_columns.joblib
    ├── kmeans_model.joblib
    ├── score_scaler.joblib
    └── safe_centroid.joblib
```

---

## Step 2: Train & Save Models Locally

Open terminal in your project folder and run:

```bash
pip install -r requirements.txt
python train_and_save.py
```

You should see:
```
✅ Data cleaned.
✅ Random Forest trained.
✅ RF + KMeans models saved.
✅ K-Prototypes trained.
✅ K-Prototypes model saved.
✅ Neural Network trained.
✅ Neural Network + encoders saved.
🎉 All models saved to /models/
```

---

## Step 3: Test Locally

```bash
streamlit run app.py
```

Open http://localhost:8501 and verify everything works.

---

## Step 4: Push to GitHub

```bash
git init                        # (skip if repo already exists)
git add app.py requirements.txt models/
git commit -m "Add Streamlit app and trained models"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

> ⚠️ Do NOT push Response.csv — it contains student data.
> Add it to .gitignore:
> ```
> echo "Response.csv" >> .gitignore
> echo "*.csv" >> .gitignore
> ```

---

## Step 5: Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select:
   - **Repository**: your-repo
   - **Branch**: main
   - **Main file path**: `app.py`
5. Click **"Deploy"**

Wait 2–3 minutes. Your app will be live at:
`https://your-app-name.streamlit.app`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: kmodes` | Make sure `kmodes>=0.12.2` is in requirements.txt |
| `Model loading error` | Re-run train_and_save.py and re-push models/ folder |
| `tensorflow` too large / timeout | Use `tensorflow-cpu` in requirements.txt instead |
| App crashes on predict | Check that Response.csv columns match rename_map in train_and_save.py |

---

## Notes

- Models are trained once locally and loaded at runtime — **no training on Streamlit Cloud**
- The `models/` folder is pushed to GitHub so Streamlit Cloud can access them
- If you collect more survey responses, just re-run `train_and_save.py` and re-push
