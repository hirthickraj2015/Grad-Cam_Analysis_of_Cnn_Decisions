# 📓 Jupyter Notebooks Guide

## Available Notebooks

I've converted all Python scripts into Jupyter notebooks with **clear outputs beneath each cell**. You can see results as you run each cell!

---

## 🚀 Quick Start - ONE Notebook Does Everything!

### **COMPLETE_PIPELINE.ipynb** ⭐ **START HERE**

**This is the EASIEST way - one notebook with everything!**

```bash
# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook COMPLETE_PIPELINE.ipynb
```

**What it does:**
- ✅ Tests your installation
- ✅ Creates 20 sample images
- ✅ Runs all 4 methods on all images
- ✅ Performs statistical analysis
- ✅ Generates 4 publication figures
- ✅ Creates comparison tables
- ✅ Shows all results with clear outputs

**Runtime:** ~10 minutes on CPU, ~5 minutes on GPU

**Output:** Everything you need for A+ submission!

---

## 📚 Individual Notebooks (Step-by-Step Approach)

If you prefer to run steps separately:

### **1_quick_test.ipynb**
Tests that everything is installed correctly.

```bash
jupyter notebook 1_quick_test.ipynb
```

**What it does:**
- Checks all dependencies
- Tests GPU availability
- Loads ResNet-50 model
- Tests Adaptive Integrated Grad-CAM
- Tests evaluation metrics

**Runtime:** ~1 minute

---

### **2_create_sample_dataset.ipynb**
Creates synthetic medical-like images for testing.

```bash
jupyter notebook 2_create_sample_dataset.ipynb
```

**What it does:**
- Generates 20 synthetic images
- 3 complexity levels (low, medium, high)
- Saves to `medical_images/` directory
- Shows preview of all images

**Runtime:** ~30 seconds

**Output:** 20 JPG images in `medical_images/`

---

### **3_run_batch_experiments.ipynb**
Runs comprehensive experiments on all images.

```bash
jupyter notebook 3_run_batch_experiments.ipynb
```

**What it does:**
- Tests 4 methods: Adaptive, Fixed-25, Fixed-50, Fixed-100
- Computes 6 metrics per method per image
- Calculates image complexity (entropy, edges, variance)
- Saves results to CSV
- Shows summary statistics

**Runtime:** ~5 minutes for 20 images

**Output:**
- `results/experiment_results.csv`
- `results/summary_statistics.json`

---

## 💡 Which Notebook Should I Use?

### For Quick Demo & A+ Submission:
```
✅ Use: COMPLETE_PIPELINE.ipynb
```
**Why:** One notebook, run all cells, get everything. Perfect for submission.

### For Learning & Understanding:
```
✅ Use: Individual notebooks 1 → 2 → 3
```
**Why:** See each step clearly, understand the pipeline, easier to modify.

### For Testing Installation Only:
```
✅ Use: 1_quick_test.ipynb
```
**Why:** Quick verification that everything works.

---

## 🎯 How to Run Notebooks

### Option 1: Jupyter Notebook (Classic)
```bash
# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook

# Open any .ipynb file in the browser
```

### Option 2: JupyterLab (Modern)
```bash
# Activate environment
source .venv/bin/activate

# Launch JupyterLab
jupyter lab

# Open any .ipynb file
```

### Option 3: VS Code
```bash
# Open in VS Code
code COMPLETE_PIPELINE.ipynb

# VS Code will automatically use the .venv kernel
# Click "Run All" at the top
```

---

## 📊 What You'll See in Notebooks

### Clear Outputs Below Each Cell:

**Example from tests:**
```
✓ All dependencies installed!
  - PyTorch: 2.9.1
  - NumPy: 2.2.6
  - Pandas: 2.3.3
```

**Example from experiments:**
```
Processing images: 100%|██████████| 20/20 [05:19<00:00, 15.99s/it]

✓ Processed 20 images
✓ Collected 31 features per image
```

**Example from analysis:**
```
🎯 KEY FINDINGS:
================================================================================
1. Step Reduction: 20.2% fewer steps than Fixed-100
   (79.8 vs 100 steps average)

2. Time Savings: 10.1% faster than Fixed-100
   (4.73s vs 5.26s)

3. Quality Maintained: 100% retention
   Identical deletion/insertion AUC
```

**Visualizations show inline:**
- Step allocation histograms
- Efficiency-quality scatter plots
- Method comparison box plots
- Correlation analysis plots

---

## 📁 Files Created by Notebooks

After running, you'll have:

### Results Directory:
```
results/
├── experiment_results.csv          # All raw data
├── comparison_table.csv            # Summary table
├── summary_statistics.json         # Statistical summary
├── figure1_step_allocation.png     # 300 DPI
├── figure2_efficiency_quality.png  # 300 DPI
├── figure3_method_comparison.png   # 300 DPI
└── figure4_correlation_analysis.png # 300 DPI
```

### Dataset Directory:
```
medical_images/
├── sample_001_low.jpg
├── sample_002_low.jpg
├── ...
└── sample_020_medium.jpg
```

---

## 🔧 Troubleshooting

### Problem: Kernel not found
**Solution:**
```bash
source .venv/bin/activate
python -m ipykernel install --user --name=gradcam --display-name="Python (GradCAM)"
```
Then select "Python (GradCAM)" kernel in Jupyter.

### Problem: Module not found
**Solution:**
```bash
source .venv/bin/activate
pip install jupyter ipykernel
```

### Problem: Plots not showing
**Solution:** Add this at the top of notebook:
```python
%matplotlib inline
```

### Problem: Out of memory
**Solution:** Reduce number of images:
```python
NUM_IMAGES = 10  # Instead of 20
```

---

## 📖 Tips for Best Results

### 1. **Run Cells in Order**
Don't skip cells - they build on each other!

### 2. **Wait for Each Cell**
Some cells take time (especially experiments). Watch the progress bar.

### 3. **Save Often**
File → Save after major milestones

### 4. **Restart if Needed**
Kernel → Restart & Run All (if something breaks)

### 5. **Clear Outputs Before Submission**
Cell → All Output → Clear (makes file smaller for submission)

---

## 🎓 For Your Submission

### What to Include:

**1. The Notebook:**
- `COMPLETE_PIPELINE.ipynb` (with outputs)
- Or all individual notebooks

**2. The Results:**
- `results/` folder with all figures and CSVs

**3. The Report:**
- Use `RESULTS_README.md` as base
- Add figures from `results/`
- Include comparison table

**4. Presentation:**
- Use figures from notebooks
- Show key outputs (findings, tables)

---

## 🏆 Expected Results

### Step Allocation:
- Mean: ~80 steps (range: 55-100)
- 20% reduction vs Fixed-100
- Bimodal distribution

### Quality:
- Identical deletion/insertion AUC across all methods
- 100% quality retention

### Efficiency:
- 10% faster than Fixed-100
- Statistical significance: p < 0.001

### Correlation:
- Edge density → more steps (r ≈ 0.22)
- Confidence → fewer steps (r ≈ -0.19)

---

## 📞 Need Help?

**Check existing outputs:**
All notebooks already have example outputs from my run. Look at those if you're stuck!

**Read the guides:**
- `WHAT_HAPPENED.md` - Quick summary
- `RESULTS_README.md` - Detailed explanation
- `README.md` - Project overview

**Common issues:**
- Environment not activated → `source .venv/bin/activate`
- Kernel wrong → Select `.venv` kernel
- Module missing → `pip install -r requirements.txt`

---

## 🎉 You're Ready!

**Recommended workflow:**

1. **Quick test:**
   ```bash
   jupyter notebook 1_quick_test.ipynb
   # Run all cells → should see all ✓
   ```

2. **Full pipeline:**
   ```bash
   jupyter notebook COMPLETE_PIPELINE.ipynb
   # Run all cells → wait ~10 minutes → see all results!
   ```

3. **Review results:**
   ```bash
   open results/figure*.png
   cat results/comparison_table.csv
   ```

4. **Prepare submission:**
   - Export notebook as PDF (File → Export → PDF)
   - Include results/ folder
   - Write report using RESULTS_README.md

---

**Good luck with your A+ submission! 🚀**

*All notebooks have clear outputs and visualizations for easy understanding and presentation.*
