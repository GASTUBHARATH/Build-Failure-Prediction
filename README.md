# Build-Failure Prediction Using Machine Learning

## 1. Introduction

Software builds can succeed or fail for many reasons, such as large code changes,
test problems, or dependency updates. This project uses machine learning to
predict whether a build is likely to be **Success** or **Failure**.

This is a beginner-friendly B.Tech CSE AI/DS assignment. The dataset is simulated
for educational use and is **not real production or company data**.

## 2. Problem Statement

Build failures consume developer time and can delay software releases. If build
history is available, a machine-learning model can learn patterns from previous
builds and provide an early warning for a new build.

The goal is to predict the `build_status` using five build-related measurements.

## 3. Objectives

- Create a realistic simulated dataset of software builds.
- Clean and prepare the data for machine learning.
- Train Logistic Regression and Random Forest models.
- Compare both models using standard classification metrics.
- Save the better model for later predictions.
- Provide command-line and Streamlit prediction interfaces.

## 4. Technologies Used

- **Python 3.12** - programming language
- **Pandas** - data loading and tabular data processing
- **NumPy** - numerical operations and simulation support
- **Scikit-learn** - preprocessing, model training, and evaluation
- **Matplotlib** - charts and confusion-matrix visualization
- **Seaborn** - optional statistical visualization support
- **Streamlit** - interactive web dashboard
- **Joblib** - saving and loading the trained model

## 5. Dataset Description

The file `dataset/build_data.csv` contains **500 simulated software build
records**. It contains both `Success` and `Failure` labels, with realistic
variation and some missing feature values. The data intentionally contains noise,
so the models are not expected to make perfect predictions.

### Dataset columns

| Column | Description |
|---|---|
| `build_id` | Unique identifier for a build, such as `BUILD-0001`. |
| `build_duration` | Approximate time taken by the build, in minutes. |
| `code_changes` | Number of code changes included in the build. |
| `previous_failures` | Number of earlier failed builds associated with the build history. |
| `test_failures` | Number of tests that failed during the build. |
| `dependency_changes` | Number of dependency changes included in the build. |
| `build_status` | Target label: `Success` or `Failure`. |

`build_id` identifies a record but is not used as a model feature. The five
numeric columns are used to predict `build_status`.

## 6. Project Folder Structure

```text
Build-Failure-Prediction/
|
|-- dataset/
|   `-- build_data.csv              Simulated dataset with 500 build records
|
|-- src/
|   |-- preprocess.py               Loads, validates, and splits the data
|   |-- train_model.py              Trains and evaluates both models
|   `-- predict.py                  Makes a command-line prediction
|
|-- models/
|   |-- build_failure_model.joblib  Saved best model
|   |-- evaluation_results.json     Saved metrics and confusion matrix
|   `-- confusion_matrix.csv        Saved confusion matrix table
|
|-- screenshots/                    Screenshots for the college report
|-- report/                         Reserved for the future written report
|-- app.py                          Streamlit dashboard
|-- requirements.txt                Required Python libraries
|-- README.md                       Project documentation
`-- .gitignore                      Files excluded from Git
```

## 7. Machine-Learning Workflow

1. Load the simulated CSV file.
2. Check that the required columns are present.
3. Convert feature values to numeric values.
4. Handle missing numeric values using the training-set median.
5. Encode `Success` as `0` and `Failure` as `1`.
6. Split the data into 80% training data and 20% testing data.
7. Fit preprocessing only on the training data to avoid data leakage.
8. Train Logistic Regression and Random Forest classifiers.
9. Evaluate both models on the unseen test data.
10. Select the model with the better F1-score and save it with Joblib.

## 8. Preprocessing Steps

The preprocessing code is in `src/preprocess.py`.

- The CSV is loaded with Pandas.
- Required columns and target labels are validated.
- Numeric feature values are converted safely.
- Missing numeric values are filled with the median calculated from the training
  data through a Scikit-learn pipeline.
- Features and target are separated.
- The target is encoded as `0` for Success and `1` for Failure.
- Data is split using a fixed random state of `42` and stratification.
- Numeric features are standardized for Logistic Regression.

Using a pipeline ensures that information from the test set is not used while
fitting preprocessing.

## 9. Models

### Logistic Regression

Logistic Regression estimates the probability of a class using a weighted
combination of the input features. It is simple, fast, and easy to explain.
It works well as a baseline classification model.

### Random Forest

Random Forest combines many decision trees. Each tree learns different rules
from the data, and the forest combines their predictions. It can model
non-linear relationships and interactions between build features.

## 10. Model Evaluation Results

The saved results use a test set of 100 records:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 62.00% | 64.00% | 61.54% | 62.75% |
| Random Forest | 56.00% | 58.33% | 53.85% | 56.00% |

Logistic Regression was selected because it achieved the higher F1-score. These
results are measured on simulated data and should not be described as production
accuracy or as a guarantee of future build outcomes.

### Confusion matrix

The selected model's confusion matrix is:

```text
                    Predicted Success    Predicted Failure
Actual Success              30                  18
Actual Failure              20                  32
```

- **30 true successes** were correctly predicted as Success.
- **32 true failures** were correctly predicted as Failure.
- **18 false alarms** were predicted as Failure although the build succeeded.
- **20 missed failures** were predicted as Success although the build failed.

The matrix is useful because it shows the types of mistakes, not only the total
accuracy.

## 11. Streamlit Dashboard Features

Run `app.py` to open the dashboard. It provides:

- Project title and short explanation.
- Saved model evaluation metrics.
- Comparison of both trained models.
- Confusion matrix visualization.
- Input fields for a new build.
- Success or Failure prediction.
- Estimated probability of failure when supported by the model.
- A simple explanation and warning for a high predicted failure risk.
- Error messages when the saved model or metrics file is missing.

## 12. Installation

Open PowerShell and move to the project folder:

```powershell
cd D:\Build-Failure-Prediction
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The required libraries are listed in `requirements.txt`:
`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`,
and `joblib`.

## 13. Commands to Run the Project

### Train the models

```powershell
python -m src.train_model
```

This prints the evaluation results and creates or updates the files in `models/`.

### Sample prediction

```powershell
python -m src.predict --build-duration 18 --code-changes 25 --previous-failures 1 --test-failures 0 --dependency-changes 1
```

Example output format:

```text
Prediction: Success
Probability of failure: 35.0%
```

The exact probability depends on the saved model and input values.

### Launch the dashboard

```powershell
streamlit run app.py
```

Streamlit displays a local browser address, normally
`http://localhost:8501`.

## 14. Limitations

- The dataset is simulated and may not represent a real CI/CD environment.
- Only five numeric measurements are used.
- The model does not inspect source code, commit messages, or test details.
- Results come from one train/test split and may change with another split.
- A failure probability is an estimate, not a guarantee.
- More real and diverse build records would be needed for production use.

## 15. Future Enhancements

- Replace simulated data with anonymized CI/CD build history.
- Use cross-validation and hyperparameter tuning.
- Add repository, branch, developer, commit, and test-suite features.
- Compare additional models such as Gradient Boosting or XGBoost.
- Track model performance over time and detect data drift.
- Add prediction history and export options to the dashboard.
- Add automated tests and continuous integration for the project itself.

## 16. Conclusion

This project demonstrates a complete beginner-level machine-learning workflow
for build-failure prediction. It loads and preprocesses build data, compares two
classification algorithms, evaluates their errors, saves the better model, and
provides an accessible Streamlit dashboard. The results are useful for learning
the workflow, while real deployment would require larger, reliable production
data and additional validation.
