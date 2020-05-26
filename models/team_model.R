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
                           forgetting_factor = 0.003)

ratings_df <- data.frame("Teams" = teams, "Mean Attack" = ratings$mean_attack, "Mean Defence" = ratings$mean_defence,
                         "Std Attack" = ratings$sd_attack, "Std Defence" = ratings$sd_defence)

# test
df_test <- df[(nrow(df) - 100 + 1):nrow(df),]

# variables
df_test[, 'Expected Home Goals'] <- exp(ratings$mean_home_coefficient + ratings$mean_intercept + ratings_df$Mean.Attack[match(df_test[, 'Home Team'], ratings_df$Teams)] + ratings_df$Mean.Defence[match(df_test[, 'Away Team'], ratings_df$Teams)])
df_test[, 'Expected Away Goals'] <- exp(ratings$mean_intercept + ratings_df$Mean.Attack[match(df_test[, 'Away Team'], ratings_df$Teams)] + ratings_df$Mean.Defence[match(df_test[, 'Home Team'], ratings_df$Teams)])
df_test[, 'Actual Home Points'] <- (3 * (df_test[, 'Home Goals'] > df_test[, 'Away Goals'])) + (1 * (df_test[, 'Home Goals'] == df_test[, 'Away Goals']))
df_test[, 'Actual Away Points'] <- (3 * (df_test[, 'Home Goals'] < df_test[, 'Away Goals'])) + (1 * (df_test[, 'Home Goals'] == df_test[, 'Away Goals']))
df_test[, 'Expected Home Points'] <- (3 * (1 - pskellam(0, df_test[, 'Expected Home Goals'], df_test[, 'Expected Away Goals']))) + (1 * dskellam(0, df_test[, 'Expected Home Goals'], df_test[, 'Expected Away Goals']))
df_test[, 'Expected Away Points'] <- (3 * (1 - pskellam(0, df_test[, 'Expected Away Goals'], df_test[, 'Expected Home Goals']))) + (1 * dskellam(0, df_test[, 'Expected Home Goals'], df_test[, 'Expected Away Goals']))

df_home <- as.data.frame(df_test[, c("Date", "Home Team", "Expected Home Points", "Actual Home Points")])
names(df_home) <- c("Date", "Team", "ExpectedPoints", "ActualPoints")

df_away <- as.data.frame(df_test[, c("Date", "Away Team", "Expected Away Points", "Actual Away Points")])
names(df_away) <- c("Date", "Team", "ExpectedPoints", "ActualPoints")

df_overall <- rbind(df_home, df_away)

# actual vs expected
df_summary <- df_overall %>% group_by(Team) %>% summarise(SumExpectedPoints = sum(ExpectedPoints),
                                                          SumActualPoints = sum(ActualPoints))

df_summary %>% arrange(by = - SumActualPoints)

# old summary with no forgetting factor: old_df_summary