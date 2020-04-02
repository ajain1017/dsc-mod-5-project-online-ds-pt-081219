
# Module 5 Final Project

(technical presentation can be found in index.ipynb file in project_files folder)


## Goal

Attempt to forecast the outcomes of March Madness during this year's NCAA Division I Men’s Basketball Championships.
In this notebook, we construct a classifier to attempt to forecast the outcomes of 2017-2019 NCAAM March Madness first round matchups. Although our goal is overall preditions of the first round, we will keep special attention to predicting the upsets that occur.

#### Define Upsets:
An upset is considered a lower seed beating a higher seed team (i.e. 15 beats 2, 10 beats 5, and even 9 beats 8)

## Data

#### Raw Data Description:

Each season there are thousands of NCAA basketball games played between Division I men's teams, culminating in March Madness, the 68-team national championship that starts in the middle of March. From Google's Kaggle NCAAM March Madness competition, I have gathered various csv datasets containing a large amount of historical data about college basketball games and individual teams from 1985. The main data used in this project includes:

#### Desired Train and Test Dataset Construction:

Our dataset contains 

| Team1ScoreDiff | Team2ScoreDiff | ... | Team1ORTG | Team2ORTG | ... | CoachDiff | ... | Team1ScoreDiff-1 | Team2ScoreDiff-1 | ... | Team1H | Team1H | ... | Team1fhf52 | Team2fhf52 | ... | Target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10.823 | 9.424 | ... | 1.007 | 1.040 | ... | 608 | ... | 6.456 | 5.345 | ... | 1 | 0 | ... | 0.416 | 0.200 | ... | 1 |
| 7.903 | 2.548 | ... | 1.101 | 0.975 | ... | 0 | ... | 8.623 | -1.456 | ... | 0 | 1 | ... | 0.750 | 1.000 | ... | 0 |
| 10.187 | 12.333 | ... | 1.044 | 1.024 | ... | -123 | ... | 0.235 | -2.349 | ... | 0 | 0 | ... | 0.500 | 0.333 | ... | 1 |


## Hypertuning Parameters and Modeling

Instead of modeling for home value, we modeled and forecast the historical return on investment for 2-years and 5-years instead. This is because, there are way more factors that influence home value and overtime, return on investment over a period of time limit the influence of those factors.
We keep our last forecasted value, RMSE and a variability metric for that forecast as well. Since our investor is risk averse, I wanted to make sure we choose the forecasts that are the best predicted. Therefore, by taking the quotient of size of confidence interval over highest value of in the interval, we can see the significance of the size of confidence interval given the forecasted value obtained.

Our code involved a four step process of finding number of differences and seasonal differences, finding order using those number of differences, fitting the model and then forecasting our return on investment.

## Interpreting Results



## Future Work

- Include individual player significance (injuries, player talent)  
- Include secondary tournament results  
- Include past tournament results  
- Find better parameters for classifiers (test for better model parameters in gridsearch)  
- Arrange target to show upset winning rather than lower Team ID (this should force the coefficents to value upsets in predictions)  
- Make functions more efficient (slow down runtime for computing features)