
# Predicting NCAAM 2017-2019 First Round Matchups

(technical presentation can be found in index.ipynb file in project_files folder)


## Goal

In this notebook, we construct a classifier to attempt to forecast the outcomes of 2017-2019 NCAAM March Madness first round matchups. Although our goal is overall preditions of the first round, we will keep special attention to predicting the upsets that occur.

#### Define Upsets:
An upset is considered a lower seed beating a higher seed team (i.e. 15 beats 2, 10 beats 7, and even 9 beats 8)


## Data

#### EDA:

![Average Points Scored in First Round](project_files/averages)


If you are to compare the average points scored by the winning team and the average points scored by the losing team, you can see a 10 point difference in every season. The highest average points scored by the losing team doesn't even reach the lowest average scored by the winning team, showing you that most of these games were not just won but dominated by the winning team.


![Number of Upsets Per Season](project_files/upsetsovertime)


Since 2003, there has always been 4 or more upsets per year. 2016 had the largest number of upsets, at 13, and 2004 had the lowest at 4. Thre three seasons we are looking to predict, 2017-2019, had an increasing number of upsets with 2019 having the most at 12.

2019 was full of upsets but not that many were suprises if you were to look at the teams closely. Many of the lower seeded teams that won were coming off a hot streak, one of them on a 17-game win streak heading into the tournament. Others had top-talented players who chose to go to smaller schools and showed up big time in the tournament. On the flip-side, some of the higher seeded teams had a lack of completeness to their game and roster, showing holes that could have been taken advantadge of.


#### Raw Data Description:

Each season there are thousands of NCAA basketball games played between Division I men's teams, culminating in March Madness, the 68-team national championship that starts in the middle of March. From Google's Kaggle NCAAM March Madness competition, I have gathered various csv datasets containing a large amount of historical data about college basketball games and individual teams from 1985. I have various datasets containing raw data all follwing different structures, but most consisting of a winnign team, a losing team, and their corresponding information. Some datasets are team specific and some are season specific. Below, you may visualize some of the raw data used in computing the features used in our model:

| Season | DayNum | WTeamID | WScore | LTeamID | LScore | WLoc | NumOT | WFGM | WFGA | ... | LStl | LBlk | LPF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2003 | 10 | 1104 | 68 | 1328 | 62 | N | 0 | 27 | 58 | ... | 9 | 2 | 20 |
| 2003 | 10 | 1272 | 70 | 1393 | 63 | N | 0 | 26 | 62 | ... | 8 | 6 | 16 |
| 2003 | 11 | 1266 | 73 | 1437 | 61 | N | 0 | 24 | 58 | ... | 2 | 5 | 23 |

| EventID | Season | DayNum | WTeamID | LTeamID | WFinalScore | LFinalScore | WCurrentScore | LCurrentScore | ElapsedSeconds | EventTeamID | EventPlayerID | EventType | EventSubType | X | Y | Area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5216688 | 2017 | 11 | 1104 | 1157 | 70 | 53 | 0 | 0 | 0 | 1104 | 6977 | sub | in | 0 | 0 | 0 |
| 5216689 | 2017 | 11 | 1104 | 1157 | 70 | 53 | 0 | 0 | 15 | 1157 | 1899 | foul | unk | 0 | 0 | 0 |
| 5216690 | 2017 | 11 | 1104 | 1157 | 70 | 53 | 0 | 0 | 15 | 1157 | 1899 | turnover | unk | 0 | 0 | 0 |

| Season | TeamID | FirstDayNum | LastDayNum | CoachName |
| --- | --- | --- | --- | --- |
| 1985 | 1102 | 0 | 154 | reggie_minton |
| 1985 | 1103 | 0 | 154 | bob_huggins |
| 1985 | 1104 | 0 | 154 | wimp_sanderson |

| TeamID | TeamName | FirstD1Season | LastD1Season |
| --- | --- | --- | --- |
|1101 | Abilene Chr | 2014 | 2020 |
|1102 | Air Force | 1985 | 2020 |
|1103 | Akron | 1985 | 2020 |

| Season | RankingDayNum | SystemName | TeamID | OrdinalRank |
| --- | --- | --- | --- | --- |
| 2003 | 35 | SEL | 1102 | 159 |
| 2003 | 35 | SEL | 1103 | 229 |
| 2003 | 35 | SEL | 1104 | 12 |
| 2003 | 35 | SEL | 1105 | 314 |


#### Desired Train and Test Dataset Construction:

The desired structure of our target variable going into our model fitting and forecasting is to predict whether the team with the lower Team ID wins the game. Therefore, our features will be labelled by Team1 (lower Team ID) and Team2 (higher Team ID). Each row will specify a specific game and the target will be the end result of that game. Our features will be rolling averages of difference in per game and advanced statistics for each team up to the date of that game. This means, the difference in scoring, field goal percentage in first 5 minutes of game, etc. for each team will be averages of the games played previously. There are also columns corresponding to current season statistics, like difference in how long that program has been in Division I basketball, difference in average ranking by rating systems on that date and more. There are also columns referring to the dates specific importance, like difference in coach tenure up to that game and location of game. Lastly, there are the same per game and advanced statistics total season averages for both teams but for the prior three seasons as well, as college players can play up to four years in NCAA.

The train data will consist of regular season matchups (Days 1-132), and our test data will consist of first round tournament games (Days 136-137) whose games have values for all of the features available.

In this notebook, train and test data is for season 2017-2019.

Below, you can see a rough idea of the structure of our train and test data:

| Team1ScoreDiff | Team2ScoreDiff | ... | Team1ORTG | Team2ORTG | ... | CoachDiff | ... | Team1ScoreDiff-1 | Team2ScoreDiff-1 | ... | Team1H | Team2H | ... | Team1fhf52 | Team2fhf52 | ... | Target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10.823 | 9.424 | ... | 1.007 | 1.040 | ... | 608 | ... | 6.456 | 5.345 | ... | 1 | 0 | ... | 0.416 | 0.200 | ... | 1 |
| 7.903 | 2.548 | ... | 1.101 | 0.975 | ... | 0 | ... | 8.623 | -1.456 | ... | 0 | 1 | ... | 0.750 | 1.000 | ... | 0 |
| 10.187 | 12.333 | ... | 1.044 | 1.024 | ... | -123 | ... | 0.235 | -2.349 | ... | 0 | 0 | ... | 0.500 | 0.333 | ... | 1 |


## Hypertuning Parameters and Modeling

Although inputing train values for future seasons to predict past seasons may influence results (in this case using 2018 and 2019 train data for 2017 test, and 2019 train data for 2018 test), I aim to find model that can be used to predict first round matchups for the NCAA tournament every year, not individual years. So we run our pipeline on our whole train dataset and later use the best classifier we find on seperate years train data.

We first use standardize our data and then we convert our features into a set of values of linearly uncorrelated variables using a Standard Scalar and Principal Component Analysis. I then test for classifiers XGBoost, RandomForest and Logistic Regression using a gridsearch to hypertune a set of parameters to find the best.

Logistic regression leads to simple understanding of our explanatory features and if speed is a factor, Logistic regression is the answer. Random Forest may be a better choice for unbalanced data as it builds each tree independently and combines results at the end of the process through averaging. XGBoost is an additive model that combines results as it trains. Instead of training independently like Random Forest, it trains models in succession and corrects the errors made by the previous models. However, XGBoost is the hardest to tune.

The resulting model is XGBoost, with 74 components and 50 estimators.


## Interpreting Results

Before going into results, know that 65.2% of our upsets had a higher team ID.

When training over combined regular season data, we get an accuracy score of 75.3%, having a true positive rate of 73.2% and true negative rate of 77.3%. Therefore, we were more successful in predicting that the team with the higher team ID wins. In 2017, we predicted 60% of our upsets correctly and 86.2% of our overall matchups. In 2018, we predicted 57.1% of our upsets correctly and 71.4% of our overall matchups. In 2019 however, we only predicted 45.5% of our upsets correctly and 67.9% of our overall matchups.

When training over each season independently, we get an accuracy score of 76.5% for all three years, having a true positive rate of 73.2% and true negative rate of 79.5%. Therefore, we were even more successful in predicting that the team with the higher team ID wins. In 2017, we predicted 60% of our upsets correctly and 86.2% of our overall matchups. In 2018, we predicted 42.9% of our upsets correctly instead but still 71.4% of our overall matchups. In 2019 however, we have now predicted 54.5% of our upsets correctly and also 71.4% of our overall matchups.

There are various factors that influence a basketball game, not only in the statistics but also external factors like elevation of performance and momentum heading into a game. How to value importance of those factors for each game is another issues as one variable may be more important than another depending on the opponent and day. It is important to note that we were successful in predicting just over 76% of first round matchups using solely regular season results. With more in depth understanding of feature engineering for this topic, feature importance and gathering of data outside of regular season, we may construct a more accurate model.


## Future Work

- Include individual player significance (injuries, player talent)  
- Include secondary tournament results  
- Include past tournament results  
- A better understanding of incorrectly predicted matches  
- Find better parameters for classifiers (test for better model parameters in gridsearch)  
- Arrange target to show upset winning rather than lower Team ID (this should force the coefficents to value upsets in predictions)  
- Make functions more efficient (slow down runtime for computing features)