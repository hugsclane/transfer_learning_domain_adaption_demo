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
- The dataset has state instead of zip code.
- The data has total credit limit, if tot_hi_cred_lim an amalgamation of other data types instead of just total credit limit, they will be different.
- term, sub_grade,intrest_rate seem to be the same
- Revolving utilization $ \text{ru} =  \frac{\text{bal}}{\text{cred_lim}} $  total_credit_utilized is defined as being the total credit balance, or $\text{total_credit} - \text{balance}$ so Revolving_utilization in this data set should be calculable as $ \text{ru} =  \frac{\text{total_credit}- \text{balance}}{\text{cred_lim}}$
-
