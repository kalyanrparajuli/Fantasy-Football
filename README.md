# Fantasy Football Model and Data for the 2019/2020 Premier League Season
----------------------------------

## Model Overviews

### Team Model
A Bayesian model for team's goals in a particular game, based on Dixon and Coles with form (points from previous 5 games) coefficients included.

### Player Model
A hierarchal Bayesian model for player's individual performances (minutes played, assists, goals and games played).

### Optimization Model
A simulator, based on both the team and player models, for the expected points of any given player for any set of fixtures.

---------------------

## Training

The team model is trained initially on the previous three seasons of the Premier league (2016/2017, 2017/2018 and 2018/2019). The player model is trained initially on the previous two seasons of the Premier League (2017/2018 and 2018/2019). In both model, each season is time-weighted with respect to how recent it was. The team model is trained using MCMC. The hierarchal player models are trained using standard analytical Bayesian inference.

--------------------

## Model Updating

After each game-week, the team model parameters are updated by using a particle filter step, where the likelihood is based on the current game-week's results. The player model parameters are updated (incorporating a forgetting factor) using standard analytical Bayesian inference.

--------------------

## Player Performance Sampling

Using the parameters (trained and updated as described above) for the team and player models, the performances of any player in a particular upcoming fixture can be sampled from the predictive posterior distributions associated with the models.

---------------------

## Analysis of Performance for the 2018/2019 Season

![Screenshot](images/exp_points_vs_actual_20182019.png)

---------------------

#### Data Credit:
Available from `https://github.com/vaastav/Fantasy-Premier-League/tree/master/data'
