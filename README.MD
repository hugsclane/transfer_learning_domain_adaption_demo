## Motivation
In a kaggle homeloan dataset https://www.kaggle.com/code/hugseal/home-loan-approval-keras,
the best predictive tool that is provided in the data set is credit history. If this metric is the the primary method of determing the outcome of a loan application, it will excluded a large percentage of the population who has not built up a credit history.

This paper outlines a method of measuring the security of a potential loan explicitly igoring credit history as a way of giving creditors a way lend to people who traditionally would not have a adequate credit score.
https://www.richdataco.com/building-a-credit-risk-model-using-transfer-learning-and-domain-adaptation/

It uses a transfer learning model outlined in this paper.
https://www.richdataco.com/transfer-learning-in-credit-risk/

This repo aims to replicate the transfer learning methods as an aide to personal technical development.


## Notes about dataset
The dataset linked in transfer-learning-in-credit-risk is not acceible via the link, instead this demo uses https://www.openintro.org/data/index.php?data=loans_full_schema, which seems like a similar datasets but it has the following discrepancies.

- The dataset in transfer-learning-in-credit-risk issue date, but the dataset used in this demo only has issue month/year, not the specific date
- The dataset has state instead of zip code, zipcode will be excluded.
- The data has total credit limit, if tot_hi_cred_lim an amalgamation of other data types instead of just total credit limit, they will be different.
- emp_length is already represented as floating point values in my dataset, however there is not data from &lt $ 1\text{ year}$ in the dataset.
- term, sub_grade, intrest_rate, installment,  debt_to_income, account_open_past_24mths, loan_status. loan_purpose, application_type, annual_income and loan_amount seem to be the same
- Revolving utilization is periodic utilization of lines of credit that are currently being repayed, since lines of credit might be closed in total_credit_utilized it may not idicatate Revolving_utilizaiton, since there are not better tools calculate it from what I can see I will use \[$\frac{\text{total credit utilized}}{\text{total credit limit}}]. This might be inaccurate, check later.
- cover is \[ $\frac{\text{annual income}}{\text{loan amount}} \]
- balance might be the same metric as average current balance.
- all_util is total_credit_utilized
- cover is generted from \[ $\frac{\text{annual income}}{\text{loan amount}} $\]
- policy codes seem to be an internal way for lending club to rate the security of its loans all loans shoud be classified as policy code unless they meet the following conditions according to https://www.fintechnexus.com/policy-code-2-loans-lending-club/. We can populate this code field by filtering for those conditions.
    - All credit datafields are blank, Credit datafields are, if it doesnt populate we can change it. 
        - total_credit_utilized
        - total_credit_limit
        - open_credit_lines
        - total_credit_lines
    - grade is F and G with intrest ranging from 23.5% to 26.06%
    - all 36 month loans
    - average of $8500
    - max of $15,000
- by these conditions, no policy code 2 borrower data exists in the dataset so policy code will be excluded.

sidenote "These are loans made to borrowers that do not meet Lending Club’s current credit policy standards." I wonder what the status of these loans are now that intrest rates aren't comfortably at 0.


Lending Club is a platorm that allows lending between individuals, the loans_purpose field will allow us to separate the data into separate domains.(check Rfile in source repo)
\n
Those domains are:
- moving
- debt_consolidation
- other
- credit_card
- home_improvement
- medical
- house
- small_business
- car
- major_purchase
- vacation
- renewable_energy


other,major_purchase are vauge enough that they will be exlcuded as target domains.
debt_consolidation and credit_card are the most plentiful (5144,2249 respectivly), so they will be our source and target domain for now.



## Notes on preprocessing
# encoding
- grade and sub_grade map A..Z to 7..1 and 1..5 to [9,7,5,3,1]
- employment length with invalid fields are set to 0
-

# other
after encoding catagorical data, the author of transfer-learning-in-credit-risk does 2 statistically important things to the data in preprocessing.
 - trims datavalues 10,000 std away from the mean as a way of removing outliers. (by filtering the dataframe)
 - normalizes the data (maps it to (0,1])

## Notes about the model.
Transfer learning component of the model will attempt to describe target domain partially in terms shared factors and partially in terms of new derived factors, the paper, they used small to medium bussiness loans as a target domain. I would initially like to try and use a different target domain.

The model will be a classifcation model, that has an additional optimization variable (layer?), the relative source/target feature data impact ratios (They might not be ratios, but that is how I will interpret it for now).

# GiniROC
The quality of the scoring model. was in the paper determined by computing the GiniROC over the results of the model, this implementation wont deviate, GiniROC will split data in Classification and regression tree algorithims in the same manner as GiniROC. This will give a measure of model quality, before the classification step which occurs in the credit decisioning process.

Next we want to produce a probability of default, there is a logistic regression model method and a credit scoring method, we will the credit scoring method which is \[ $PD=(A-B*\text{Score})^C$ \] where A,B, and C are parameters from the model calibration process.


