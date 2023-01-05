# Telco Churn Project

## Project Description
Telco is a telecommunications provider that has experienced high levels of churn and needs to understand why in order to adjust current business practices and policies to better develop and retain a customer base moving forwards

## Project Goal
- To discover the main drivers of churn within Telco
- Use the drivers to develop a ML program that predicts churn with at least 80% accuracy
- Deliver a report to a non techinical supervisor in a digestable manner

## Questions to Answer
- Does monthly charges impact churn?
- Does payment type impact churn?
- Does contract type impact churn?
- Does having dependants impact churn?
- Of the listed variables, which has the most statistical significance in regards to churn?

## Initial Thoughts and Hypothesis
I believe that the main drivers behind churn will be monthly cost and contract type (i.e. the way the contract is structured) with the assumption that lower monthly charges and longer contracts being less likely to churn compared to the high monthly charges and less committed contracts. I also think that having dependants will also impact churn do to having more than one line to handle.

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
| churn | When a customer cancels contract or subscription with the company |
| contract_type | The type of contract that the customer has with Telco |
| payment_type | The form in which the customer pays their monthly bill |
| dependents | Whether or not the customer has a dependent on their account |
| monthly_charges | How much a customer pays per month |
| tenure | How long a customer has been with the company |
| total_charges | How much a customer has paid over their entire tenure |
| payment_type_id | Number assignments for stats purposes |
| contract_type_id | Number assignments for stats purposes |

## Takeaways and Conclusion
-  The models all reached around 79% accuracy which is about a 5% improvment over the baseline
-  The models all very accuractly prediciting retention but struggled to accurately predict who would churn
  - Most models reached around a 50% success rate of predicting active churn
- The statistical relevance of all drivers was extremely high with all being significantly lower p values than .05
- Visualizations imply that a lot of churn is related to the contract and payment type used by customers
  - Month to month contracts and electronic check payment plans had the most churn by far

## Recomendations
- The company should encourage and heavily incentivize customers to sign long term contracts to ensure higher retention rates
- The company should also look towards a direct deposit system of payment to also increase retention among customers

## Next Steps
- Given more time, an analysis of all variables would be preferable to find any better drivers of churn
- Creating more models and trying to increase accuracy to at least 80% would also be a good goal to pursue given more time
