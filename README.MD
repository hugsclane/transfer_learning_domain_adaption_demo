## Motivation
In a kaggle homeloan dataset https://www.kaggle.com/code/hugseal/home-loan-approval-keras,
the best predictive tool that is provided in the data set is credit history. If this metric is the the primary method of determing the outcome of a loan application, it will excluded a large percentage of the population who has not built up a credit history.

This paper outlines a method of measuring the security of a potential loan explicitly igoring credit history as a way of giving creditors a way lend to people who traditionally would not have a adequate credit score.
https://www.richdataco.com/building-a-credit-risk-model-using-transfer-learning-and-domain-adaptation/

It uses a transfer learning model outlined in this paper
https://www.richdataco.com/transfer-learning-in-credit-risk/


# Notes about data
The dataset linked in transfer-learning-in-credit-risk is not acceible via the link, instead this demo uses https://www.openintro.org/data/index.php?data=loans_full_schema, which seems like a similar datasets but it has the following discrepancies.
- The dataset in transfer-learning-in-credit-risk issue date, but the dataset used in this demo only has issue month/year, not the specific date
- The dataset has state instead of zip code, zipcode will be excluded.
- The data has total credit limit, if tot_hi_cred_lim an amalgamation of other data types instead of just total credit limit, they will be different.
- term, sub_grade, intrest_rate, installment, emp_length, debt_to_income, account_open_past_24mths, loan_status. loan_purpose, application_type, annual_income and loan_amount seem to be the same
- Revolving utilization \[ $ru=\frac{\text{bal}}{\text{cred lim}}$ \] total_credit_utilized is defined as being the total credit balance, or \[ $\text{total credit}-\text{balance}$ \]. Balance is not provided but total_credit and total_credit_utilized is in the data. So Revolving_utilization in this dataset should be calculable as \[ $ru=\frac{\text{total credit}-\text{total credit utilized}}{\text{credit limit}}$ \]. Its possible that balance is in this dataset total balance. Since Revolving_utilization over a fixed period, might need a different method of calcualting it. TODO
- balance might be the same metric as average current balance, will have to look into it. TODO
- all_util is total_credit_utilized
- policy codes seem to be an internal way for lending club to rate the security of its loans all loans shoud be classified as policy code unless they meet the following conditions according to https://www.fintechnexus.com/policy-code-2-loans-lending-club/. We can populate this code field by filtering for those conditions.
    - All credit datafields are blank, Credit datafields are, if it doesnt populate we can change it. 
        - total_credit_utilized
        - total_credit_limit
        - open_credit_lines
        - total_credit_lines
    - grade is f and G with intrest ranging from 23.5% to 26.06%
    - all 36 month loans
    - average of $8500
    - max of $15,000
        sidenote "These are loans made to borrowers that do not meet Lending Club’s current credit policy standards." I wonder what the status of these loans are now that intrest rates aren't comfortably at 0.
