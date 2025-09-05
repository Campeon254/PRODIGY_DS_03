# PRODIGY_DS_03
Predicting whether a customer will subscribe to a term deposit after a marketing campaign.

## Overview

This project aims to analyze the factors influencing customer subscription to term deposits following a marketing campaign and predicting whether a customer will subscribe. By leveraging exploratory data analysis (EDA) and machine learning techniques, we seek to uncover insights from the dataset and build predictive models to enhance marketing strategies.

## Tables of Content
- [Overview](#overview)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Predictive Analysis](#predictive-analysis)
- [Recommendations](#recommendations)
- [Set-Up](#set-up)
- [License](#license)

## Data Cleaning and Preparation
The data was obtained from the Bank Marketing Dataset from the UCI Machine Learning Repository. The dataset contains information about customers who were contacted through a marketing campaign and whether they subscribed to a term deposit.

```python
# Import dataset
bank_marketing = fetch_ucirepo(id=222) 

# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 

# Load dataset
data = pd.concat([X, y], axis=1)
data.head()
```
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>NaN</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>NaN</td>
      <td>single</td>
      <td>NaN</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>

**Data Cleaning:**

This process involved:
- Filling Missing values with `unknown`.
- Checking for duplicates and removing them.
- Checking for outliers and handling them appropriately.
- Feature Engineering:
    - Creating age bins.
    - Creating duration bins.

## Exploratory Data Analysis (EDA)
In this stage, we performed various analyses to understand the dataset better and uncover insights. Key activities included:
- Visualizing call distribution by subscription outcome.
- Visualizing the distribution of age groups by subscription outcome.
- Visualizing the distribution of job types by subscription outcome.
- Visualizing the distribution of education levels by subscription outcome.
- Visualizing the distribution of marital status by subscription outcome.
- Visualizing the distribution of housing loan status by subscription outcome.
- Visualizing the distribution of personal loan status by subscription outcome.

To get a detailed view of the analysis, find it under the [PDF report](Customer%20Demographics.pdf) or access the Power BI dashboard.
[Power BI Dashboard](Customer%20Demographics.pbix)

**Insights:**
1. Call duration is a key determinant of subscription success. Calls that lasted between 5-15 minutes had the highest number of successful subscriptions compared to those that were shorter or longer.

2. Majority of customers are:
    - adults aged between 31-45 years
    - employed in management and technician roles
    - with secondary education
    - married
    - without housing or personal loans

## Predictive Analysis
In this stage, we built and evaluated machine learning models to predict whether a customer would subscribe to a term deposit based on the features in the dataset. We used Decision Tree and LightGBM classifiers, tuning their hyperparameters using GridSearchCV.

The following are the evaluation results from the Decision Tree classifier:
```
Best CV F1 Score: 0.7489117841384989
Test Accuracy: 0.8771425411920822
Classification Report:
               precision    recall  f1-score   support

          no       0.96      0.90      0.93      7985
         yes       0.48      0.71      0.57      1058

    accuracy                           0.88      9043
   macro avg       0.72      0.80      0.75      9043
weighted avg       0.90      0.88      0.89      9043

Confusion Matrix:
 [[7181  804]
 [ 307  751]]
Test ROC AUC: 0.8950639964110401
```
- The model accuracy is 0.8771.
- The model's F1 score is 0.7489.
- The ROC AUC score is 0.8951.
- Precision-Recall for `yes` is lower.

Since we are dealing with an imbalanced dataset, we need to pay special attention to the performance metrics for the minority class (`yes`).

We will focus on improving the recall score since it is better not to miss potential subscribers.

The following are the evaluation results for the LightGBM classifier:
```
Accuracy: 0.8558000663496628
ROC AUC: 0.9338273677133282

Classification Report:
               precision    recall  f1-score   support

          no       0.98      0.85      0.91      7985
         yes       0.44      0.88      0.59      1058

    accuracy                           0.86      9043
   macro avg       0.71      0.87      0.75      9043
weighted avg       0.92      0.86      0.87      9043


Confusion Matrix:
 [[6808 1177]
 [ 127  931]]
```
- The model accuracy is 0.8558 which is a slight drop from the Decision Tree model.
- The recall of the minority class increased to 0.88. This is better than the 0.71 recall achieved by the Decision Tree model.

`Our model is performing well in terms of recall for the minority class, which is crucial for identifying potential subscribers.`

## Recommendations
1. **Focus on Call Duration**: Since call duration is a key determinant of subscription success, consider implementing strategies to optimize call lengths, aiming for the 5-15 minute range.

2. **Targeted Marketing**: Utilize the insights from the EDA to create targeted marketing campaigns aimed at the identified key demographics (e.g., adults aged 31-45, management and technician roles).

3. **Monitor Model Performance**: Continuously monitor the performance of the LightGBM model, especially the recall for the minority class, and retrain the model as necessary with new data.

4. **Random Forest Classifier**: Consider implementing a Random Forest classifier as an additional model to compare performance and potentially improve recall for the minority class.

5. **Deployment**: Plan for the deployment of the model into a production environment, ensuring that it can be accessed by the necessary stakeholders.

## Set-Up
1. Clone the repository to your local machine.
```bash
git clone https://github.com/Campeon254/PRODIGY_DS_03.git
cd PRODIGY_DS_03
```
2. Install the required packages using pip:
```
pip install -r requirements.txt
```
3. Open the Jupyter Notebooks (`.ipynb` files) in Jupyter Notebook or JupyterLab to explore the data cleaning, EDA, and predictive analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
