# Logs Analysis Notebook Guide

## 📊 Overview
`logs_analysis.ipynb` - Comprehensive Jupyter notebook for analyzing prediction validation logs from the EV Battery Safety Management System.

## 📁 File Information
- **File**: `logs_analysis.ipynb`
- **Size**: ~27KB
- **Total Cells**: 30 cells (15 markdown + 15 code)
- **Input Data**: `prediction_validation_log.json`

## 📖 Notebook Structure

### Section 1: Data Loading and Overview
- Import required libraries
- Load prediction validation logs
- Display basic data structure and overview

### Section 2: Basic Statistics
- Entry count and time range analysis
- Data quality assessment
- Missing data analysis

### Section 3: RL Agent Action Analysis
- Action distribution analysis
- Action frequency and patterns
- Confidence analysis by action
- Visual distribution (pie chart)

### Section 4: Anomaly Detection Analysis
- Anomaly probability statistics
- Threshold-based analysis (70%, 50%, 30%)
- Visual probability distribution
- Box plot analysis

### Section 5: Safety Assessment Analysis
- Safety status distribution
- Cross-tabulation: Safety vs Action
- Confidence by safety status
- Anomaly probability by safety status

### Section 6: Scenario-Based Analysis
- Action patterns by scenario
- Average anomaly probability per scenario
- Scenario-specific insights
- **Note**: Scenarios are testing labels, NOT AI inputs

### Section 7: RL Agent Validation
- Validate actions against safety principles
- Temperature-based validation
- SoC-based validation
- Anomaly-based validation
- Overall safety score

### Section 8: Confidence Analysis
- Confidence statistics by action
- Confidence vs anomaly correlation
- Box plots by action type
- Scatter plot analysis

### Section 9: Temporal Analysis
- Action evolution over time
- Confidence trends with moving averages
- Anomaly probability trends
- Sequential pattern detection

### Section 10: Advanced Analytics
- Feature correlation matrix
- Correlation heatmap visualization
- Alert analysis (critical/warning)
- Alert distribution analysis

### Summary Section
- Comprehensive analysis summary
- Key performance indicators
- System insights and recommendations

## 🚀 How to Use

### 1. Prerequisites
```bash
pip install jupyter pandas numpy matplotlib seaborn
```

### 2. Launch Jupyter
```bash
cd /home/floodai/Desktop/Shivam/EV-Battery-Safety-System-Clean
jupyter notebook logs_analysis.ipynb
```

### 3. Run Analysis
- **Run All Cells**: Kernel → Restart & Run All
- **Run Individual Sections**: Click on cell and press Shift+Enter

### 4. Expected Outputs
- Statistical summaries printed to output
- Multiple visualization plots (pie charts, histograms, scatter plots, heatmaps)
- Correlation matrices
- Validation scores

## 📊 Key Metrics Analyzed

1. **Action Distribution**: Which actions are most common
2. **Confidence Levels**: How confident is the RL agent
3. **Anomaly Detection**: Distribution of anomaly probabilities
4. **Safety Assessment**: Distribution of safety statuses
5. **Scenario Performance**: How system behaves across scenarios
6. **Validation Score**: % of correct vs risky vs wrong actions
7. **Temporal Trends**: How metrics evolve over time
8. **Feature Correlations**: Relationships between telemetry features

## 📈 Visualization Types

- **Pie Charts**: Action and safety distribution
- **Histograms**: Confidence and anomaly probability distributions
- **Box Plots**: Confidence by action, anomaly by safety
- **Scatter Plots**: Confidence vs anomaly correlation
- **Line Plots**: Temporal trends with moving averages
- **Heatmaps**: Feature correlation matrix
- **Bar Charts**: Alert distributions

## 🔍 Analysis Insights

### What the Notebook Reveals:
1. **RL Agent Behavior**: Most common actions and decision patterns
2. **Safety Performance**: How well the system maintains safety
3. **Anomaly Detection Accuracy**: Distribution and effectiveness
4. **Scenario Appropriateness**: Whether actions match scenarios
5. **System Reliability**: Consistency and confidence over time
6. **Feature Relationships**: Which sensors correlate with decisions

### Key Findings to Look For:
- ✅ High confidence (>80%) = Well-trained states
- ✅ Low confidence (<50%) = Needs more training data
- ✅ Correct actions (>80%) = Excellent safety performance
- ⚠️ High anomaly rate (>30%) = Potential system sensitivity
- ⚠️ Risky actions (>10%) = Review decision logic

## 📝 Notes

### Important Considerations:
1. **Scenario Labels**: Used for testing/analysis only, NOT by AI models
2. **Real-World Deployment**: AI uses only raw sensor values
3. **Data Source**: Synthetic telemetry for testing purposes
4. **Continuous Logging**: Last 1000 entries maintained automatically

### Customization:
- Modify threshold values in Section 4 and 7
- Adjust bin sizes in histograms
- Change moving average window in Section 9
- Add custom validation rules in Section 7

## 🛠️ Troubleshooting

### Common Issues:

**Issue**: `FileNotFoundError: prediction_validation_log.json`
- **Solution**: Run the dashboard first to generate logs

**Issue**: `ImportError: No module named 'seaborn'`
- **Solution**: `pip install seaborn matplotlib`

**Issue**: Empty plots or missing data
- **Solution**: Ensure logs have sufficient entries (run dashboard for longer)

**Issue**: Memory error with large log files
- **Solution**: Process logs in chunks or reduce sample size

## 📞 Support

For questions or issues, refer to:
- Main project documentation: `PROJECT_REPORT.md`
- Dashboard README: `README.md`
- Prediction log structure: Check first few entries in `prediction_validation_log.json`

## ✅ Quick Start Checklist

- [ ] Install dependencies (`jupyter`, `pandas`, `numpy`, `matplotlib`, `seaborn`)
- [ ] Generate logs by running the dashboard
- [ ] Open notebook: `jupyter notebook logs_analysis.ipynb`
- [ ] Run all cells: Kernel → Restart & Run All
- [ ] Review outputs and visualizations
- [ ] Check validation scores in Section 7
- [ ] Examine correlation matrix in Section 10

---

**Created**: October 2, 2025  
**Last Updated**: October 2, 2025  
**Version**: 1.0  
**Author**: EV Battery Safety System Team
