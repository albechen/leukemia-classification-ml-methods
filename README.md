## Classifying Type Leukemia by Gene Expression
##### Classification Problem of Acute Myeloid Leukemia (AML) and Acute Lymphoblastic Leukemia (ALL) by Gene Expression Monitoring

**Language:** R (caret, tidyr, ggplot) </br>

This study aims to find implement a classification model correctly determine if the cancer patient has Acute Myeloid Leukemia (AML) or Acute Lymphoblastic Leukemia (ALL) using results from gene expression testing. The dataset includes results from 38 patients training set and 34 patients for the testing set with gene expression results that include the numerical readout from the assay, and the associated categorical determination from the assay preformed – providing insight of presence or absence of the gene given the readout.

To avoid multicollinearity related results, the categorical and numerical data is split due to their direct correlation and the separate datasets and both used and compared for final testing results. In the data, 7129 genes are tested. Therefore, data reduction is performed. Two methods are used. First – PCA is preformed on both numerical and categorical data’s training set and applied to the testing set. Second – The highest correlated genes regarding the classes are sorted and the highest correlated genes regarding the training data is used. With the preprocessing completed – four datasets are formed, PCA vs. Correlation, and Numerical vs. Categorical.

The four datasets are implemented into various models. The models tested are Log Regression, Random Forest, K-Nearest-Neighbor, Linear Discriminant Analysis, Support Vector Machine, Neural Network. To implement each of the models with tuning parameters, a cross fold validation method is used for each with accuracy as the metric. Finally, each of the models using the four datasets are used to predict the testing dataset the final accuracy value.

Overall, it was observed that mapping correlation associated with each of the genes for numerical and categorical data was greatly effective at predicting the correct class of Leukemia. Comparatively, it was observed performing PCA on the numerical dataset had performed poorly across all models. Between the numerical and categorical datasets in which correlated genes were chosen, the results are within variance, but numerical dataset performed slightly better.

Given the results, it is recommended that the top twenty genes used for the tests can accurately determine the class of Leukemia. The dataset best suited is the numerical dataset and implementing either Neural Network, SVM or random forest, however other models like Log Regression and Naive Bayes work well.


##### File Summary
1. R: [Data Preperation and Seperation](https://github.com/albechen/leukemia-classification-ml-methods/blob/master/1-cancer_start.Rmd) 
2. R: [Gene Correlation](https://github.com/albechen/leukemia-classification-ml-methods/blob/master/2-cancer_cor.Rmd), [Principle Component Analysis](https://github.com/albechen/leukemia-classification-ml-methods/blob/master/2-cancer_pca.Rmd)
3. R: [Model Implementation](https://github.com/albechen/leukemia-classification-ml-methods/blob/master/3-cancer-automated.Rmd)
4. PDF: [Report Summary](https://github.com/albechen/leukemia-classification-ml-methods/blob/master/leukemia_classification_report.pdf)

## Visualization
### Model Implementaiton Summary
![alt text](/images/summary/summary_results.png "summary_results")

### Gene Correlation and PCA Examples
![alt text](/images/reduction/cat_cor.png "cat_cor")
![alt text](/images/reduction/pca_test_cat.png "pca_test_cat")
