### Team Model - Training and Implementing

source("utils.R")


# load fixture lists
data20172018 <- read.csv("../data/premier_league_20172018.csv", header = FALSE)
data20182019 <- read.csv("../data/premier_league_20182019.csv", header = FALSE)
data20192020 <- read.csv("../data/premier_league_20192020.csv", header = FALSE)

# overall fixture list
df <- rbind(data20182019, data20192020)
names(df) <- c("Date", "Home Team", "Away Team", "Home Goals", "Away Goals")

# training period in 2019-2020
n_train = 100
df_train <- df[1:(nrow(df) - 100), ]

# teams
teams <- unique(df_train[, "Home Team"])

# compute ratings
ratings <- compute_ratings(df_train, teams, beta = 0.0007, niter = 90000,
                           forgetting_factor = 0.0)

ratings_df <- data.frame("Teams" = teams, "Mean Attack" = ratings$mean_attack, "Mean Defence" = ratings$mean_defence,
                         "Std Attack" = ratings$sd_attack, "Std Defence" = ratings$sd_defence)
