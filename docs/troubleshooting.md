# Troubleshooting Guide

**Author:** Josiah Gordor  
**Last Updated:** February 2026

---

## Quick Diagnostics

### Health Check

Run this command to verify system status:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_name": "XGBoost Fair",
    "encoder_loaded": true,
    "scaler_loaded": true
}
```

---

## API Issues

### API Offline / Connection Refused

**Symptoms:**
- Dashboard shows "❌ API Offline"
- `Connection refused` error when calling API
- Health check fails

**Solutions:**

1. **Start the API server:**
```bash
cd /path/to/loan_approval
uvicorn src.api.main:app --port 8000
```

2. **Check if port is in use:**
```bash
lsof -i :8000
```

3. **Kill existing process and restart:**
```bash
pkill -f uvicorn
uvicorn src.api.main:app --port 8000
```

4. **Use a different port:**
```bash
uvicorn src.api.main:app --port 8001
```

---

### Models Not Loaded (503 Error)

**Symptoms:**
- API returns "Models not loaded"
- Health check shows `"model_loaded": false`
- 503 Service Unavailable errors

**Solutions:**

1. **Check model files exist:**
```bash
ls -la models/fair_models/
```

Expected files:
- `xgb_fair.pkl`
- `rf_fair.pkl`
- `lr_fair.pkl`
- `nn_fair.pkl`
- `glm_fair.pkl`
- `fasterrisk_fair.pkl`
- `gosdt_fair.pkl`

2. **Check scaler file:**
```bash
ls -la models/feature_scaler.pkl
```

3. **Verify file permissions:**
```bash
chmod 644 models/*.pkl
chmod 644 models/fair_models/*.pkl
```

4. **Retrain models if missing:**
```bash
python -m src.models.trainer
```

5. **Check API logs for specific errors:**
```bash
uvicorn src.api.main:app --port 8000 --log-level debug
```

---

### Slow Response Times

**Symptoms:**
- Predictions take > 1 second
- Dashboard feels sluggish
- Timeouts on requests

**Solutions:**

1. **First request after startup is slow (expected):**
   - Models load lazily on first request
   - Wait for initial load to complete

2. **Check system resources:**
```bash
top -l 1 | head -10  # macOS
htop                  # Linux
```

3. **Reduce concurrent requests:**
   - Avoid batch predictions during peak usage

4. **Consider fewer SHAP computations:**
   - Use `/predict` instead of `/explain` for quick checks

5. **Upgrade hardware:**
   - Add more RAM (recommended: 8GB+)
   - Use SSD storage for model files

---

### Validation Errors (422)

**Symptoms:**
```json
{
    "detail": [
        {"loc": ["body", "loan_amount"], "msg": "ensure this value is greater than 0"}
    ]
}
```

**Solutions:**

| Field | Valid Range | Fix |
|-------|-------------|-----|
| `loan_amount` | > 0 | Must be positive number |
| `property_value` | > 0 | Must be positive number |
| `income` | ≥ 0 | Cannot be negative |
| `interest_rate` | 0-100 | Percentage value |
| `loan_term` | 1-600 | Months (typical: 180 or 360) |
| `state_code` | 2 chars | e.g., "CA", "NY", "TX" |

---

### SHAP Explanation Errors

**Symptoms:**
- `/explain` endpoint returns 500 error
- "SHAP computation failed" message
- Explanation shows empty factors

**Solutions:**

1. **Check SHAP installation:**
```bash
pip install shap --upgrade
```

2. **Verify explainer is loaded:**
```bash
curl http://localhost:8000/health
# Check for "explainer_loaded": true
```

3. **Use prediction instead:**
   - `/predict` includes basic key factors
   - Fallback when SHAP fails

4. **Check for unusual input values:**
   - Extreme values may cause numerical issues
   - Try normalizing inputs

---

## Dashboard Issues

### Dashboard Won't Start

**Symptoms:**
- `ModuleNotFoundError`
- Streamlit crashes on startup
- Blank page

**Solutions:**

1. **Install dependencies:**
```bash
pip install streamlit plotly pandas requests
```

2. **Run with correct Python:**
```bash
which python  # Check Python path
python -m streamlit run src/dashboard/app.py
```

3. **Check for syntax errors:**
```bash
python -m py_compile src/dashboard/app.py
```

4. **Clear Streamlit cache:**
```bash
rm -rf ~/.streamlit/cache
streamlit run src/dashboard/app.py
```

---

### Dashboard Shows "API Offline" But API Is Running

**Symptoms:**
- API responds to curl
- Dashboard can't connect
- Network/CORS errors in browser

**Solutions:**

1. **Check API URL in dashboard:**
   - Default: `http://localhost:8000`
   - Ensure API and dashboard use same host

2. **Verify no firewall blocking:**
```bash
# macOS
sudo pfctl -s rules

# Linux
sudo iptables -L
```

3. **Check browser console for CORS errors:**
   - Press F12 → Console tab
   - Look for CORS-related messages

4. **Run both on same machine:**
```bash
# Terminal 1
uvicorn src.api.main:app --port 8000 --host 0.0.0.0

# Terminal 2
streamlit run src/dashboard/app.py --server.port 8501
```

---

### Fairness Charts Not Displaying

**Symptoms:**
- Empty fairness dashboard
- Charts show "No data"
- Plotly errors

**Solutions:**

1. **Check Plotly installation:**
```bash
pip install plotly --upgrade
```

2. **Verify browser compatibility:**
   - Use Chrome, Firefox, or Edge
   - Enable JavaScript

3. **Clear browser cache:**
   - Ctrl+Shift+Delete → Clear cache

4. **Check for data issues:**
   - Fairness data may be unavailable for certain models
   - Try switching to XGBoost Fair

---

### Form Inputs Not Saving

**Symptoms:**
- Values reset on page refresh
- Inputs disappear unexpectedly

**Solutions:**

1. **Don't refresh mid-submission:**
   - Wait for prediction to complete

2. **Enable session state:**
   - Check browser allows cookies/local storage

3. **Use stable connection:**
   - Avoid navigation during processing

---

## Model Issues

### "No attribute 'values'" Error

**Symptoms:**
```
AttributeError: 'numpy.ndarray' object has no attribute 'values'
```

**Solutions:**

This is handled internally by the API. If you see this error:

1. **Update to latest API code:**
```bash
git pull origin main
```

2. **Restart API:**
```bash
pkill -f uvicorn
uvicorn src.api.main:app --port 8000
```

---

### "predict() got unexpected keyword argument 'verbose'"

**Symptoms:**
- FasterRisk or GOSDT models fail
- TypeError on prediction

**Solutions:**

This is handled internally. If persists:

1. **Update API routes:**
   - Special handling exists for these models
   - Ensure `routes.py` has fallback logic

2. **Restart API after updates:**
```bash
pkill -f uvicorn
uvicorn src.api.main:app --port 8000
```

---

### Wrong Model Selected

**Symptoms:**
- Predictions from unintended model
- Model name mismatch in response

**Solutions:**

1. **Check sidebar model selector:**
   - Ensure desired model is selected

2. **Verify in API response:**
   - Response includes `"model_name"` field

3. **Clear session and re-select:**
   - Refresh page
   - Re-select model

---

## Installation Issues

### ImportError: No module named 'xxx'

**Solutions:**

```bash
pip install -r requirements.txt
```

Common missing packages:
```bash
pip install pandas numpy scikit-learn xgboost shap
pip install fastapi uvicorn pydantic
pip install streamlit plotly
```

---

### Package Version Conflicts

**Symptoms:**
- `werkzeug` ImportError
- `sklearn` vs `scikit-learn` issues

**Solutions:**

1. **Upgrade critical packages:**
```bash
pip install --upgrade werkzeug flask fastapi uvicorn
```

2. **Create fresh environment:**
```bash
conda create -n loan_approval python=3.10
conda activate loan_approval
pip install -r requirements.txt
```

---

### Conda Environment Issues

**Solutions:**

1. **Activate environment:**
```bash
source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate ds4b_101p
```

2. **Verify correct environment:**
```bash
which python
# Should show conda environment path
```

---

## Performance Issues

### High Memory Usage

**Solutions:**

1. **Reduce loaded models:**
   - API loads all models by default
   - Consider lazy loading for unused models

2. **Restart API periodically:**
   - Clears accumulated memory

3. **Increase system swap:**
   - If RAM is limited

---

### High CPU Usage

**Solutions:**

1. **Limit batch sizes:**
   - Max 100 applications per batch request

2. **Reduce SHAP computations:**
   - Use `/predict` for bulk processing

3. **Scale horizontally:**
   - Run multiple API instances behind load balancer

---

## Logging & Debugging

### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG uvicorn src.api.main:app --port 8000
```

### View API Logs

```bash
uvicorn src.api.main:app --port 8000 --log-level debug
```

### Save Logs to File

```bash
uvicorn src.api.main:app --port 8000 2>&1 | tee api.log
```

---

## Error Code Reference

| Status Code | Meaning | Action |
|-------------|---------|--------|
| 200 | Success | N/A |
| 422 | Validation Error | Check input values |
| 500 | Internal Server Error | Check API logs |
| 503 | Service Unavailable | Models not loaded |

---

## Getting Help

If issues persist:

1. **Collect diagnostic info:**
   - API logs
   - Browser console errors
   - System specs (OS, Python version, RAM)

2. **Contact support:**
   - Technical: IT Help Desk
   - Model questions: compliance@company.com

3. **Check documentation:**
   - [Configuration Guide](configuration_guide.md)
   - [FAQ](faq.md)

**Author:** Josiah Gordor
