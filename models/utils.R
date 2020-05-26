## Utility functions for team model in R

library(dplyr)
library(skellam)


EstimateParameters <- function(fixture_lists, teams, beta, thetapriormeans,
                               thetapriorsds, niter = 1000, logu = FALSE, temp = 0, forgetting_factor = 0,
                               zerooutinds = NULL) {
  
  # priors
  if (logu) {
    
    if (length(thetapriormeans) > 1) {
      
      theta = array(0, length(thetapriormeans))
      
      for (i in 1:length(theta)) {
        
        theta[i] <- exp(rnorm(1, thetapriormeans[i], thetapriorsds[i]))
        
      }
      
    }
      
    else {
      
      theta <- exp(rnorm(1, thetapriormeans, thetapriorsds))
      
    }
    
  }
  
  else {
    
    if (length(thetapriormeans) > 1) {
      
      theta <- array(0, length(thetapriormeans))
      
      for (i in 1:length(thetapriormeans)) {
        
        theta[i] <- rnorm(1, thetapriormeans[i], thetapriorsds[i])
        
      }
      
    }
    
    else {
      
      theta <- rnorm(1, thetapriormeans, thetapriorsds)
      
    }
    
  }
  
  if (length(thetapriormeans) > 1) {
    
    thetaarray <- array(0, dim = c(niter, length(thetapriormeans)))
    
  }
  
  else {
    
    thetaarray <- array(0, niter)
    
  }
  
  accept_count <- 0
  
  for (j in 1:niter) {
    
    # Temperature
    T <- exp(-temp * ((j + 1) / niter))
    
    # random increment
    if (logu) {
      
      if (length(thetapriormeans) > 1) {
        
        thetastar <- exp(log(theta) + rnorm(length(theta), 0, sqrt(beta)))
        
      }
      
      else {
        
        thetastar <- exp(log(theta) + rnorm(1, 0, sqrt(beta)))
        
      }
      
    }
    
    else {
      
      if (length(thetapriormeans) > 1) {
        
        ind <- rnorm(length(theta), 0, sqrt(beta))
        thetastar = theta + ind
        
      }
      
      else {
      
        ind <- rnorm(1, 0, sqrt(beta))
        thetastar <- theta + 1
        
      }
      
    }
    
    # define coefficients
    intercept <- theta[1]
    mu <- theta[2]
    attack <- theta[3:(length(teams) + 2)]
    defence <- theta[(length(teams) + 3):((2 * length(teams)) + 2)]
    
    # zero out any teams
    if (length(zerooutinds) > 0) {
      
      attack[zerooutinds] <- 0
      defence[zerooutinds] <- 0
      
    }
    
    # zero sum constraint
    attack[1] <- - sum(attack[2:length(attack)])
    defence[1] <- - sum(defence[2:length(defence)])
    
    prioreval <- array(0, length(theta))
    
    # evaluate prior for previous iterate
    for (k in 1:length(theta)) {
      
      prioreval[k] <- dnorm(theta[k], thetapriormeans[k], thetapriorsds[k])
      
    }
    
    # compute likelihood for previous iterate
    Htheta <- compute_likelihood(fixture_lists, teams, intercept, mu, attack, defence, forgetting_factor = forgetting_factor) + sum(log(prioreval))
    
    # define coefficients
    intercept <- thetastar[1]
    mu <- thetastar[2]
    attack <- thetastar[3:(length(teams) + 2)]
    defence <- thetastar[(length(teams) + 3):((2 * length(teams)) + 2)]
    
    # zero out any teams
    if (length(zerooutinds) > 0) {
      
      attack[zerooutinds] <- 0
      defence[zerooutinds] <- 0
      
    }
    
    # zero sum constraint
    attack[1] <- - sum(attack[2:length(attack)])
    defence[1] <- - sum(defence[2:length(defence)])
    
    prioreval <- array(0, length(thetastar))
    
    # evaluate prior for proposal
    for (k in 1:length(thetastar)) {
      
      prioreval[k] <- dnorm(thetastar[k], thetapriormeans[k], thetapriorsds[k])
      
    }
    
    # compute likelihood for proposal
    Hthetastar <- compute_likelihood(fixture_lists, teams, intercept, mu, attack, defence, forgetting_factor = forgetting_factor) + sum(log(prioreval))
    
    # compute ratio
    alpha <- min(0, (1 / T) * (Hthetastar - Htheta))
    
    # sample uniformly
    u <- runif(1, 0, 1)
    
    # accept or not
    accept <- log(u) <= alpha
    
    if (accept) {
      
      theta <- thetastar
      accept_count <- accept_count + 1
      
    }
    
    if (length(thetapriormeans) > 1) {
      
      thetaarray[j, ] <- theta
      
      if (j%%10 == 0) {
        
        print(paste0('----- Completed iteration ', as.character(j), ' out of ', as.character(niter), ' --------'))
        print(paste0('Acceptance Ratio ', as.character(100 * (accept_count / j)), ' %'))
        
      }
      
    }
    
    else {
      
      thetaarray[j] <- theta
      
    }
    
    # normalize back
    if (length(thetapriormeans) > 1) {
      
      if (length(zerooutinds) > 0) {
        
        thetaarray[, (3 + zerooutinds)] <- 0
        thetaarray[, (3 + zerooutinds + length(teams))] <- 0
        
      }
      
      thetaarray[, 3] <- - rowSums(thetaarray[, 4:(length(teams) + 2)])
      thetaarray[, (length(teams) + 3)] <- - rowSums(thetaarray[, (length(teams) + 4):((2 * length(teams)) + 2)])
      
    }
    
  }
  
  return(thetaarray)
  
}

compute_likelihood <- function(fixture_lists, teams, intercept, mu, attack, defence, forgetting_factor = 0) {
  
  date <- as.Date(fixture_lists[, 'Date'], format = "%d/%m/%Y")
  
  # define home and away inds
  home_ind <- match(fixture_lists[, 'Home Team'], teams)
  away_ind <- match(fixture_lists[, 'Away Team'], teams)
  
  # attack
  home_attack <- attack[home_ind]
  away_attack <- attack[away_ind]
  
  # defence
  home_defence <- defence[home_ind]
  away_defence <- defence[away_ind]
  
  # scalars
  scale <- exp(- forgetting_factor * as.numeric(as.Date(Sys.Date()) - date))
  
  # lambdas
  home_lambdas <- exp(home_attack + mu + intercept + away_defence)
  away_lambdas <- exp(away_attack + home_defence + intercept)
  
  # likelihood
  likelihood <- sum(((log(dpois(fixture_lists[, 'Home Goals'], home_lambdas)) + log(dpois(fixture_lists[, 'Away Goals'], away_lambdas))) * scale))
  
  return(likelihood)
  
  
}

compute_ratings <- function(fixture_lists, teams, beta, thetapriormeans = NULL,
                            thetapriorsds = NULL, niter = 1000, temp = 0, forgetting_factor = 0,
                            zerooutinds = NULL) {
 
  # use non-NULL prior means and sds if updating dynamic rating
  if (length(thetapriormeans) == 0) {
    thetapriormeans <- c(0.15, 0.15, array(0.0, length(teams)), array(0.0, length(teams)))
    thetapriorsds <- array(0.15, (2 + (2 * length(teams))))
  }
  
  # find parameters
  batch <- EstimateParameters(fixture_lists, teams, beta, thetapriormeans = thetapriormeans,
                     thetapriorsds = thetapriorsds, niter = niter, temp = temp, forgetting_factor = forgetting_factor,
                     zerooutinds = zerooutinds)
  
  # burn in
  params <- batch[floor(nrow(batch) / 2):nrow(batch), ]

  home_coefficient <- params[, 2]
  intercept <- params[, 1]
  attack <- params[, 3:(length(teams) + 2)]
  defence <- params[, (length(teams) + 3):((2 * length(teams)) + 2)]

  # means
  mu_home_coefficient <- mean(home_coefficient)
  mu_intercept <- mean(intercept)
  mu_attack <- colSums(attack) / nrow(params)
  mu_defence <- colSums(defence) / nrow(params)

  # sds
  sd_home_coefficient <- sd(home_coefficient)
  sd_intercept <- sd(intercept)
  sd_attack <- sqrt((colSums(attack ** 2) / nrow(params)) - (mu_attack ** 2))
  sd_defence <- sqrt((colSums(defence ** 2) / nrow(params)) - (mu_defence ** 2))

  return(list("mean_home_coefficient" = mu_home_coefficient,
              "mean_intercept" = mu_intercept,
              "mean_attack" = mu_attack,
              "mean_defence" = mu_defence,
              "sd_home_coefficient" = sd_home_coefficient,
              "sd_intercept" = sd_intercept,
              "sd_attack" = sd_attack,
              "sd_defence" = sd_defence))
  
}