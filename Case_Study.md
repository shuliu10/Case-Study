Case Study: Gender Prediction


Shu Liu


July 19, 2021


Introduction
Internet advertisers have provided us with a data set that contains eight columns. The name of the columns are user_id, app_id, device_name, app_categorty, interaction_with_app, ad_category, click, and gender. There are 3700 rows in the table, and each row represents an event. The advertisers want to target within apps, and they want us to develop a model that can predict device users' gender (male/female). The advertisers have recommended we take gender as the target, and take device_name, app_category, interaction_with_app, ad_category, and click as features. 
Discussion
Identifying dataset problems
The main problems with the dataset include data imbalance, sample size, missing data, the different data types of features, and different data scales. 
Data imbalance is obvious. For example, in the target column, the proportions for males and females are 0.72 and 0.28. In the app_category column and the ad_category column, there is a significant difference between the largest and smallest proportions. The most considerable imbalance occurs in the click column, with the No accounts for 0.995 proportion. The distribution of interaction_with-app is exponential, and most interaction occurs within a few minutes.
Because there are 5700 data points in the dataset, the data imbalance is likely to be more severe on the individual app and individual ad levels.
We can choose "balanced" as calss_weight in some logistic regression classifiers, such as the Stochastic Gradient Descent (SGD) for imbalanced features. The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_calsses*np.bincount(y)). 
When your target data is imbalanced, accuracy may not be the best measure of your model performance. If the target data is highly imbalanced, ROC-AUC curve is a good indicator of classifier performance. Since our target data is moderately imbalanced, I would use the F1 score, which is the harmonic mean of precision and recall. 
There are two types of feature types. The data type in the ineraction_with_app is continuous, and the data type of the rest of the features is categorical. Since Scikit-learn will not accept categorical features by default, we can encode categorical features numerically by converting them to "dummy variables."
The scales of the categorical features are different from one another. For example, there are 15 ad categories, but there are only two categories in the click column. The scale of the numerical feature, interaction_with_app, is significantly larger than the categorical features. 
Since features on larger scales can unduly influence the model, we can standardize features using the StabndardScaler() function in scikit-learn. 
Another problem with the dataset is missing data. The device_name column is incomplete, and 59% of the data is missing.  
Selecting features
We should take both logical and statistical approaches to select features. Selecting app_category, interaction_with_app, ad_category, and click makes logical sense because male and female users may spend different periods on their preferred apps and click on the ads they like. On the other hand, device_name should not be a feature because it does not have any relationship with gender, i.e., male and female users may have the same name, and both genders use all kinds of devices. Furthermore, 59% of the data in this column is missing.
To verify our logical considerations, we should perform hypothesis testing if our dataset is not highly imbalanced on the sub_app and sub_ad levels. 
The model we will develop will show the effect of the selected features by their coefficients. If we choose L1 regularization (lasso) in the model development, we can shrink the coefficients of less important features to zero.
Choosing model 
Binary logistic regression is the most suitable for our task. Among many models, The model I would try first is Stochastic Gradient Descent (SGD) because it is a simple and efficient approach to fitting linear classifiers under convex loss functions such as Support Vector Machines and Logistic regression. It is highly efficient and straightforward to implement, and there are many opportunities for code tuning.
SGD allows us to perform different classification models by using different loss parameters. For example, the "log" loss parameter gives logistic regression, and the "hinge" loss parameter gives a linear SVM.
As discussed above, SGD has the "balanced" mode in calss_weight for us to use to deal with our imbalanced features. 
SGD, however, requires tuning many hyperparameters such as loss, penalty, alpha, and l1_ratio. This requirement is a disadvantage for inexperienced data scientists. However, experienced scientists can use the grid search function to find a set of optimal features easily.
Another disadvantage is that SGD is sensitive to feature scaling. Any inaccurate data scaling could impact your model performance.
Demonstration
To demonstrate how the SGD classifier works for our case, I created a dataset. It has a similar structure to the actual dataset, with the target and selected features discussed above and 1,000 events. The techniques I used include data standardization, train-test split, cross-validation, grid search, SGD classification with both hinge and log loss function, model performance measurement with F1 score, and model prediction for a new event. Please see the demonstration.ipynb document in this repository for details.
Summary
The article identified some dataset problems and provided solutions to deal with them.  The issues identified include data imbalance, sample size, missing data, the different data types, and scales.
The app_category, interaction_with_app, ad_category, and click are selected logically as the features for our study. To verify our logical considerations, we should perform hypothesis testing if our dataset is not highly imbalanced on the sub_app and sub_ad levels. 
Stochastic Gradient Descent (SGD) is chosen as our classifier because It is highly efficient, straightforward to implement, and there are many opportunities for code tuning. We can use the grid search function to answer its requirement for hyperparameter tuning.
Performing the SGD classification in our case is demonstrated on a dataset with a similar structure to the actual dataset. The demonstration uses many techniques in model development, evaluation, and prediction.
