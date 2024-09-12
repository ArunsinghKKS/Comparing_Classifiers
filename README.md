### Comparing Classifiers

#### Overview
In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. We will utilize a dataset related to marketing bank products over the telephone.

#### Data Source
The dataset used for this analysis is the Bank Marketing dataset from the UCI Machine Learning Repository. It contains information about various client attributes and their responses to a marketing campaign.

#### Workflow Summary

1. **Data Collection**
   - The dataset was downloaded from UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
   - It contains client information such as age, job, marital status, education, and campaign details like previous contacts and outcome of previous campaigns.

2. **Data Cleaning**
   - Duplicate records were removed to ensure data quality.
   
3. **Feature Encoding**
   - Categorical data (e.g., job, marital status) were converted to numerical values using a method called Label Encoding.
   - This process helps machine learning algorithms to interpret the data correctly.

4. **Data Scaling**
   - Numerical features were standardized to ensure they are on the same scale, improving the performance of certain algorithms.

6. **Data Splitting**
   - The data was split into training (80%) and test (20%) sets to evaluate the model's performance on unseen data.

7. **Model Training and Evaluation**
   - Following machine learning models were trained and evaluated:
     - **Baseline Model**: A dummy classifier was used to establish a baseline performance.
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Support Vector Machine (SVM)**
     - **Decision Tree**

8. **Hyperparameter Tuning**
   - Hyperparameters are settings that influence the behavior of a model.
   - Grid Search was used to find the best hyperparameters for each model, enhancing their performance.

#### Results

- **Dummy Classifier (Baseline)**
  - Accuracy: This model simply predicted the most frequent class.
  - Performance: Not very informative but establishes a baseline.

- **Baseline results using defaults for models Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree**

| Model                | Train Time | Train Accuracy | Test Accuracy |
|----------------------|------------|----------------|---------------|
| Logistic Regression | 0.381242   | 0.887557       | 0.886502      |
| K-Nearest Neighbors | 0.052704   | 0.890015       | 0.873513      |
| Decision Tree        | 0.186926   | 0.916601       | 0.861010      |
| SVM                  | 17.639997  | 0.887557       | 0.886502      |             

- **Best model results using hyperparameter tuning Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree**



| Model                 | Train Time | Train Accuracy | Test Accuracy |
|-----------------------|------------|----------------|---------------|
| Logistic Regression  | 0.077877   | 0.887557       | 0.886502      |
| K-Nearest Neighbors  | 0.047034   | 0.890349       | 0.883952      |
| Support Vector Machine | 68.118582 | 0.887587       | 0.886380      |
| Decision Tree         | 0.087841   | 0.891290       | 0.884438      |

 

#### Conclusion

The logistic regression model, after hyperparameter tuning, showed the best balance between simplicity, interpretability, and performance with an accuracy of 88.7% on test data. This model can help the bank identify clients more likely to subscribe to a term deposit, optimizing their marketing efforts.

#### Future Work

To further improve the model:
- Explore more advanced feature engineering techniques.
- Utilize ensemble methods like Random Forest or Gradient Boosting.
- Regularly update the model with new data to maintain its accuracy.

By following these steps, the bank can leverage data-driven insights to enhance their marketing strategies and improve client acquisition.
