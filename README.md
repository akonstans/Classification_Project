# Telco Churn Project

## Project Description
Telco is a telecommunications provider that has experienced high levels of churn and needs to understand why in order to adjust current business practices and policies to better develop and retain a customer base moving forwards

## Project Goal
- To discover the main drivers of churn within Telco
- Use the drivers to develop a ML program that predicts churn with at least 80% accuracy
- Deliver a report to a non techinical supervisor in a digestable manner

## Initial Thoughts and Hypothesis
I believe that the main drivers behind churn will be monthly cost and contract type (i.e. the way the contract is structured) with the assumption that lower monthly charges and longer contracts being less likely to churn compared to the high monthly charges and less committed contracts

## Planning
- Use the aquire.py already used in previous exerices to aquire the data necessary
- Use the code already written prior to aid in sorting and cleaning the data
- Isolate Main Drivers
  - First identify the drivers using statistical analysis
  - Create a pandas dataframe containing all relevant drivers as columns
- Develop a model using ML to determine churn based on the top 3 drivers
  - MVP will consist of one model per driver to test out which can most accurately predict churn
  - Post MVP will consist of taking most accurate and running it through multiple models
  - Goal is to achieve at least 80% accuracy with at least one model
- Draw and record conclusions

## Data Dictionary
| Feature | Description |
| --- | --- |
| Churn | When a customer cancels contract or subscription with the company |
| contract_type | The type of contract that the customer has with Telco |
| payment_type | The form in which the customer pays their monthly bill |
