---
title: "LASSO estimator vs Bayesian LASSO"
author: "Rashied Amir"
date: "6/2/2021"
output: html_document
---

DATA IMPORT & WRANGLING

First, I'll import the data and create the dataset by merging two separate dataframes about student performance. I also check how many students are included

```{r}
d1=read.table("student-mat.csv",sep=";",header=TRUE)
d2=read.table("student-por.csv",sep=";",header=TRUE)

d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))

print(nrow(d3))
```
There are 382 students in the dataset. Now i'll load some packages to create dummy variables which i need later on to analyze the variables and fit the model

```{r}
library('fastDummies')

df5 <- dummy_cols(d3, select_columns = c('sex','nursery','internet', 'famsize','address','Pstatus','Medu','Fedu','Medu','Mjob','Fjob','reason','guardian.x','studytime.x','traveltime.x','failures.x','schoolsup.x','famsup.x','paid.y','activities.y','higher.y','romantic.y','famrel.y','freetime.y','goout.y','Dalc.x','Walc.y','health.y'),
                      remove_selected_columns = TRUE)
```
Next, I'll remove irrelevant/double columns 

```{r}
library(dplyr)

df6 <- select(df5, -c(1,3,4,5,6,7,8,9,10,11,12,16,17,18,19,20,21,22,23))

df7 <- select(df6, -c(sex_F, nursery_no, internet_no, famsize_GT3, address_R, Pstatus_A, schoolsup.x_no, famsup.x_no, paid.y_no, activities.y_no, higher.y_no, romantic.y_no))

df8 <- select(df7, -c(G2.x,G3.x,G1.y,G2.y,G3.y))
```
Now I start preparing the data for the classic LASSO model. First I will turn the dataset into a data table so I can transform it into a matrix. Also I will split the dataset into the outcome variable and the predictors. These will be used as x and y in the model code

```{r}
library(data.table)

df9 <- setDT(df8)

cijfer = df9[, G1.x]
df9[, G1.x := NULL]

```

DATA ANALYSIS

Classic Lasso 

Now we can fit the LASSO using the following code

```{r}
library(glmnet)


set.seed(123)
lasso <- cv.glmnet(as.matrix(df9), cijfer, family = gaussian, alpha = 1,
                lambda=10^seq(4,-1,-.1), standardize = TRUE)


# The glmnet function standardizes the data by default. alpha=1 gives the lasso estimator. family = gaussian as we are fitting linear data. 
```

Now we can get the best lambda value, get the coefficients, and fit the classic LASSO

```{r}
library(ggplot2)

best_lambda = lasso$lambda.1se
lasso_coef = lasso$glmnet.fit$beta[, lasso$glmnet.fit$lambda == best_lambda]



lasso_df <- data.table(lasso = lasso_coef)
lasso_df[, feature := names(lasso_coef)]
to_plot = melt(lasso_df, id.vars = 'feature', value.name = 'coefficient')
ggplot(to_plot, aes(x=feature, y=coefficient)) + coord_flip() +
  geom_bar(stat='identity') + guides(fill=FALSE)



```
Running a linear model with the selected variables on a new sample of the data
```{r}
#model met geselecteerde variabelen klassieke lasso
newmodel_lasso_classic <- lm(G1.x ~ sex_M + failures.x_0 + schoolsup.x_yes + failures.x_3 + Mjob_other + Medu_4 + higher.y_yes + freetime.y_2 + Fjob_teacher + Fedu_1,  data = train)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers
pred_lasso_classic <- predict(newmodel_lasso_classic, newdata = test)
pred_lasso_classic

#RMSE
library(Metrics)
rmse(pred_lasso_classic,test$G1.x)
```

Bayesian Lasso


Now we continue with the Bayesian LASSO. First load in some packages and drop some variables out of the dataset which I dont need

```{r}
library(brms)

df10 <- select(df7, -c(G2.x,G3.x,G1.y,G2.y,G3.y))

```
Now I can fit the model using the brm function as follows and get the posterior distributions

```{r}

bayes_lasso = brm(formula = G1.x ~  . , family=gaussian, data=df10, 
                  prior = set_prior('lasso(1)', class='b'),
                  algorithm = 'sampling',
                  iter=2000, warmup=1000,
                  sample_prior = TRUE)
summary(bayes_lasso)

posterior <- as.array(bayes_lasso)
dim(posterior)
dimnames(posterior)


#plot distribution of the model 

pp = brms::pp_check(bayes_lasso)
pp + theme_bw()

```
Lasso 1: with train and test set

```{r}

library(caret)

df15<- select(df7, -c(G2.x,G3.x,G1.y,G2.y,G3.y))

df16 <- setDT(df15)

dt = sort(sample(nrow(df16), nrow(df16)*.7))
train<-df16[dt,]
test<-df16[-dt,]


bayes_lasso_1 = brm(formula = G1.x ~ . -G1.x, family=gaussian, data=train, 
                  prior = set_prior('lasso(df = 1, scale = 1)', class='b'),
                  algorithm = 'sampling',
                  iter=2000, warmup=1000,
                  sample_prior = TRUE)

#run the model on the test data
predictions_bayes_lasso_1 <- bayes_lasso_1 %>% predict(test)

summary(bayes_lasso_1)

```
Calculate R2 of the model on the new data 
```{r}
bayes_R2(bayes_lasso_1, newdata = test)
```

Distribution of the model

```{r}

library(rmutil)

ndraws <- 10000
lasso_1 <- rep(NA, ndraws)
for(i in 1:ndraws){
lasso_scale = 1
lasso_df = 1
lasso_inv_lambda <- rchisq(1, df = lasso_df)
lasso_1[i] <- rmutil::rlaplace(1, m=0, s=lasso_scale * lasso_inv_lambda)
}
plot(density(lasso_1))

#pp = brms::pp_check(bayes_lasso)
#pp + theme_bw()

```

Lasso 2: with train and test set

```{r}

df15<- select(df7, -c(G2.x,G3.x,G1.y,G2.y,G3.y))

df16 <- setDT(df15)

dt2 = sort(sample(nrow(df16), nrow(df16)*.7))
train_2<-df16[dt,]
test_2<-df16[-dt,]


bayes_lasso_2 = brm(formula = G1.x ~ . -G1.x, family=gaussian, data=train_2, 
                  prior = set_prior('lasso(df = 50, scale = 1)', class='b'),
                  algorithm = 'sampling',
                  iter=2000, warmup=1000,
                  sample_prior = TRUE)

stancode(bayes_lasso_2)

predictions_bayes_lasso_2 <- bayes_lasso_2 %>% predict(test_2)


bayes_R2(bayes_lasso_2, newdata = test_2)

#pp_2 = brms::pp_check(bayes_lasso_2)
#pp_2 + theme_bw()

summary(bayes_lasso_2)

```

Plotting the distribution
```{r}
install.packages("rmutil")
library(rmutil)

ndraws <- 10000
lasso_2_dist <- rep(NA, ndraws)
for(i in 1:ndraws){
lasso_scale = 1
lasso_df = 10
lasso_inv_lambda <- rchisq(1, df = lasso_df)
lasso_2_dist[i] <- rmutil::rlaplace(1, m=0, s=lasso_scale * lasso_inv_lambda)
}
plot(density(lasso_2_dist))

ndraws <- 10000
lasso_3_dist <- rep(NA, ndraws)
for(i in 1:ndraws){
lasso_scale = 0.01
lasso_df = 1
lasso_inv_lambda <- rchisq(1, df = lasso_df)
lasso_3_dist[i] <- rmutil::rlaplace(1, m=0, s=lasso_scale * lasso_inv_lambda)
}
plot(density(lasso_3_dist))


# lasso 1 en 2 together in 1 plot

myData <- data.frame(lasso_1=lasso_1,
                     lasso_2=lasso_2_dist)

install.packages("ggforce")
library(ggforce)
library(ggplot2);library(reshape2)
data<- melt(myData)
ggplot(data) + aes(x=value, col = variable) + geom_density() + xlim(-50,50) + ggtitle("Distribution of Lasso 1 and Lasso 2")


# lasso 3 distribution plot

myData <- data.frame(lasso_3 = lasso_3_dist)

install.packages("ggforce")
library(ggforce)
library(ggplot2);library(reshape2)
data<- melt(myData)
ggplot(data) + aes(x=value) + geom_density() + xlim(-1,1) + ggtitle("Distribution of Lasso 3")




```

Lasso 3:  with train and test set

```{r}

df15<- select(df7, -c(G2.x,G3.x,G1.y,G2.y,G3.y))

df16 <- setDT(df15)

dt2 = sort(sample(nrow(df16), nrow(df16)*.7))
train_5<-df16[dt,]
test_5<-df16[-dt,]

bayes_lasso_3 = brm(formula = G1.x ~ . -G1.x, family=gaussian, data=train_5, 
                  prior = set_prior('lasso(df = 1, scale = 0.01)', class='b'),
                  algorithm = 'sampling',
                  iter=2000, warmup=1000,
                  sample_prior = TRUE)



predictions_bayes_lasso_3 <- bayes_lasso_3 %>% predict(test_5)


bayes_R2(bayes_lasso_3, newdata = test_5)

#pp_2 = brms::pp_check(bayes_lasso_2)
#pp_2 + theme_bw()

summary(bayes_lasso_3)

```

Variable selection

Lasso 1: Variable selection with projpred

```{r}

install.packages("projpred")

library(projpred)

vs_lasso_1 <- cv_varsel(bayes_lasso_1)
summary(vs_lasso_1)

plot(vs_lasso_1, stats = c('rmse'))

suggest_size(vs_lasso_1)

?suggest_size

?solution_terms

solution_terms(vs_lasso_1)

# model met de beste predictoren lm

lm_fit <- lm(G1.x ~ failures.x_0 + Medu_4 + schoolsup.x_yes + freetime.y_2 + Fedu_1 + Fedu_4 + Fjob_teacher + Fjob_other + sex_M + Mjob_other, data = train)

hoi_2 <- predict(lm_fit, newdata = test)
hoi_2

# nu moet je de bias berekenen tussen de geschatte en de echte waardes
install.packages("Metrics")
library(Metrics)
rmse(hoi_2,test$G1.x)


# proj pred fitten op nieuwe data

proj_lin_1 <- proj_predict(vs_lasso_1, nterms = 4, newdata = test)

head(proj_lin_1)

nrow(proj_lin_1)

colMeans(proj_lin_1)

test$G1.x

head(train)

?proj_linpred
```
Lasso 2: Variable selection with projpred
https://cloud.r-project.org/web/packages/projpred/vignettes/quickstart.html
```{r}

library(projpred)

vs_lasso_2 <- cv_varsel(bayes_lasso_2)
summary(vs_lasso_2)

plot(vs_lasso_2, stats = c('elpd', 'rmse'))

plot(vs_lasso_2, stats = c('rmse'))


suggest_size(vs_lasso_2)

solution_terms(vs_lasso_2)

?suggest_size

# model met de beste predictoren lm
newmodel_lasso_2 <- lm(G1.x ~ failures.x_0 + Medu_4 + schoolsup.x_yes + Fjob_teacher + freetime.y_2 + Fedu_1 + sex_M + Fjob_other + sex_M + internet_yes + Mjob_other + Fedu_4 + higher.y_yes + failures.x_3, data = train)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers

hoi <- predict(newmodel_lasso_2, newdata = test)
hoi

# nu moet je de bias berekenen tussen de geschatte en de echte waardes
install.packages("Metrics")
library(Metrics)
rmse(hoi,test$G1.x)


# negeer onderstaande

# proj pred manier van fitten op nieuwe data
newmodel_lasso_2_projpred <- proj_predict(vs_lasso_2, nterms = 9, newdata = test)

head(newmodel_lasso_2_projpred)

nrow(newmodel_lasso_2_projpred)

# tot hier


```
Lasso 3: variable selection 
```{r}
library(projpred)

vs_lasso_3 <- cv_varsel(bayes_lasso_3)

summary(vs_lasso_3)

plot(vs_lasso_3, stats = c('rmse'))

suggest_size(vs_lasso_3)

solution_terms(vs_lasso_3)

?suggest_size

# model met de beste predictoren lm
newmodel_lasso_3 <- lm(G1.x ~ failures.x_0 + Medu_4 + schoolsup.x_yes + freetime.y_2 + Fedu_1 + Fedu_4 + Fjob_teacher + Fjob_other + sex_M + internet_yes + Mjob_other, data = train)

summary(newmodel_lasso_3)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers

hoi_3 <- predict(newmodel_lasso_3, newdata = test_5)
hoi_3

# nu moet je de rmse berekenen tussen de geschatte en de echte waardes
install.packages("Metrics")
library(Metrics)
rmse(hoi_3,test$G1.x)

```
Variable selection with CI 

Lasso 1: Now let's try to select the variables based on different CI values. 80 and 95. http://mc-stan.org/rstanarm/reference/posterior_interval.stanreg.html

```{r}
#eerst 80%

lasso_1_80 <- brms::posterior_interval(bayes_lasso_1, prob = 0.8)

df30 <- as.data.frame(lasso_1_80)

df30$zero_int <- data.table::between(0, lasso_1_80[, 1], lasso_1_80[, 2])

# vervolgens kijken hoeveel variabelen worden geseletceerd en daarna daarmee lm runnen en errors berekenen

# eerst kijken welke variabelen allemaal geselecteerd zijn en waarbij 0 dus NIET in het interval liggen

library(dplyr)

selected_variables_lasso_1 <- filter(df30, zero_int == "FALSE")
selected_variables_lasso_1
#dit geeft dat er 6 variabelen worden geselecteerd
                                
# nu 95%

lasso_1_95 <- posterior_interval(bayes_lasso_1, prob = 0.95)

df40 <- as.data.frame(lasso_1_95)

df40$zero_int <- data.table::between(0, lasso_1_95[, 1], lasso_1_95[, 2])

# vervolgens kijken hoeveel variabelen worden geseletceerd en daarna daarmee lm runnen en errors berekenen

# eerst kijken welke variabelen allemaal geselecteerd zijn en waarbij 0 dus NIET in het interval liggen

library(dplyr)

selected_variables_lasso_1_95 <- filter(df40, zero_int == "FALSE")
selected_variables_lasso_1_95
#dit geeft dat er 3 variabelen worden geselecteerd. 

# next step is running an lm on the test data with the selected variables and calculating RMSE

#eerst 80% 

# model met de beste predictoren lm
newmodel_lasso_1_80 <- lm(G1.x ~ sex_M + Fjob_teacher + failures.x_0 + schoolsup.x_yes + famsup.x_yes + freetime.y_2, data = train)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers
pred_lasso_1_80 <- predict(newmodel_lasso_1_80, newdata = test)
pred_lasso_1_80

#RMSE
library(Metrics)
rmse(pred_lasso_1_80,test$G1.x)


# nu 95%

# model met de beste predictoren lm
newmodel_lasso_1_95 <- lm(G1.x ~ sex_M + failures.x_0 + schoolsup.x_yes, data = train)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers
pred_lasso_1_95 <- predict(newmodel_lasso_1_95, newdata = test)
pred_lasso_1_95

#RMSE
library(Metrics)
rmse(pred_lasso_1_95,test$G1.x)



```
Lasso 2: variable selection using CI

```{r}

lasso_2_80 <- brms::posterior_interval(bayes_lasso_2, prob = 0.8)

df50 <- as.data.frame(lasso_2_80)

df50$zero_int <- data.table::between(0, lasso_2_80[, 1], lasso_2_80[, 2])

# eerst kijken welke variabelen allemaal geselecteerd zijn en waarbij 0 dus NIET in het interval liggen

library(dplyr)

selected_variables_lasso_2_80 <- filter(df50, zero_int == "FALSE")
selected_variables_lasso_2_80
#dit geeft dat er 8 variabelen worden geselecteerd


# Nu voor 95% 

lasso_2_95 <- posterior_interval(bayes_lasso_2, prob = 0.95)

df60 <- as.data.frame(lasso_2_95)

df60$zero_int <- data.table::between(0, lasso_2_95[, 1], lasso_2_95[, 2])

selected_variables_lasso_2_95 <- filter(df60, zero_int == "FALSE")
selected_variables_lasso_2_95
#dit geeft dat er 3 variabelen worden geselecteerd

# next step is running an lm on the test data with the selected variables and calculating RMSE

#eerst 80% 

# model met de beste predictoren lm
newmodel_lasso_2_80 <- lm(G1.x ~ sex_M + internet_yes + Fjob_teacher + failures.x_0 + schoolsup.x_yes + famsup.x_yes + higher.y_yes + freetime.y_2, data = train)



summary(bayes_lasso_2,prob=.8)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers
pred_lasso_2_80 <- predict(newmodel_lasso_2_80, newdata = test)
pred_lasso_2_80

#RMSE
library(Metrics)
rmse(pred_lasso_2_80,test$G1.x)


# nu 95%

# model met de beste predictoren lm
newmodel_lasso_2_95 <- lm(G1.x ~ sex_M + failures.x_0 + schoolsup.x_yes, data = train)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers
pred_lasso_2_95 <- predict(newmodel_lasso_2_95, newdata = test)
pred_lasso_2_95

#RMSE
library(Metrics)
rmse(pred_lasso_2_95,test$G1.x)

cbind.data.frame(pred_lasso_2_95,pred_lasso_3_95)

```

Lasso 3: variable selection and prediction using CI

```{r}

lasso_3_80 <- brms::posterior_interval(bayes_lasso_3, prob = 0.8)

df70 <- as.data.frame(lasso_3_80)

df70$zero_int <- data.table::between(0, lasso_3_80[, 1], lasso_3_80[, 2])

# eerst kijken welke variabelen allemaal geselecteerd zijn en waarbij 0 dus NIET in het interval liggen

library(dplyr)

selected_variables_lasso_3_80 <- filter(df70, zero_int == "FALSE")
selected_variables_lasso_3_80
#dit geeft dat er 4 variabelen worden geselecteerd


# Nu voor 95% 

lasso_3_95 <- posterior_interval(bayes_lasso_3, prob = 0.95)

df80 <- as.data.frame(lasso_3_95)

df80$zero_int <- data.table::between(0, lasso_3_95[, 1], lasso_3_95[, 2])

selected_variables_lasso_3_95 <- filter(df80, zero_int == "FALSE")
selected_variables_lasso_3_95
#dit geeft dat er 2 variabelen worden geselecteerd

# next step is running an lm on the test data with the selected variables and calculating RMSE

#eerst 80% 

# model met de beste predictoren lm
newmodel_lasso_3_80 <- lm(G1.x ~ sex_M + failures.x_0 + schoolsup.x_yes + freetime.y_2, data = train)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers
pred_lasso_3_80 <- predict(newmodel_lasso_3_80, newdata = test)
pred_lasso_3_80

#RMSE
library(Metrics)
rmse(pred_lasso_3_80,test$G1.x)


# nu 95%

# model met de beste predictoren lm
newmodel_lasso_3_95 <- lm(G1.x ~ failures.x_0 + schoolsup.x_yes, data = train)

# nu op nieuwe data runnen en dan krijg je de estimates van de cijfers
pred_lasso_3_95 <- predict(newmodel_lasso_3_95, newdata = test)
pred_lasso_3_95

#RMSE
library(Metrics)
rmse(pred_lasso_3_95,test$G1.x)


```

Calculation of RMSE values for a model with all variables and a model with just 1 variable
```{r}

newmodel_lasso_one <- lm(G1.x ~ failures.x_0, data = train)
pred_lasso_one <- predict(newmodel_lasso_one, newdata = test)
rmse(pred_lasso_one,test$G1.x)


newmodel_lasso_all <- lm(G1.x ~ age + + sex_M + nursery_yes + internet_yes + famsize_LE3 + address_U + Pstatus_T + Medu_0 + Medu_1 + Medu_2 + Medu_3 + Medu_4 + Fedu_0 + Fedu_1 + Fedu_2 + Fedu_3 + Fedu_4 + Mjob_at_home + Mjob_health + Mjob_other + Mjob_services + Mjob_teacher + Fjob_at_home + Fjob_health + Fjob_other + Fjob_services + Fjob_teacher + reason_course + reason_home + reason_other + reason_reputation + guardian.x_father + guardian.x_mother + guardian.x_other + studytime.x_1 + studytime.x_2 + studytime.x_3 + studytime.x_4 + traveltime.x_1 + traveltime.x_2 + traveltime.x_3 + traveltime.x_4 + failures.x_0 + failures.x_1 + failures.x_2 + failures.x_3 + schoolsup.x_yes + famsup.x_yes + paid.y_yes + activities.y_yes + higher.y_yes + romantic.y_yes + famrel.y_1 + famrel.y_2 + famrel.y_3 + famrel.y_4 + freetime.y_5 + goout.y_1 + goout.y_2 + goout.y_3 + goout.y_4 + goout.y_5 + Dalc.x_1 + Dalc.x_2 + Dalc.x_3 + Dalc.x_4 + Dalc.x_5 + Walc.y_1 + Walc.y_2 + Walc.y_3 + Walc.y_4 + Walc.y_5 + health.y_1 + health.y_2 + health.y_3 + health.y_4 + health.y_5, data = train)

pred_lasso_all <- predict(newmodel_lasso_all, newdata = test)
rmse(pred_lasso_all,test$G1.x)

```

Plotting the distribution of a small and a large variable for all models

```{r}

library(tidyverse)
theme_set(theme_bw(base_size=16))

#final distribution function


plotdens<- function(varNum, varName, glmvar){
  post1 <- as.matrix(bayes_lasso_1)
  post2 <- as.matrix(bayes_lasso_2)
  post3 <- as.matrix(bayes_lasso_3)
  Lasso1 <- post1[, varNum]
  Lasso2 <- post2[, varNum]
  Lasso3 <- post3[, varNum]
  x <- data.frame(Lasso1, Lasso2, Lasso3)
  data<- melt(x)
  ggplot(data,aes(x=value, color=variable)) + labs(subtitle=varName) + geom_density(alpha=0.25) + 
    geom_vline(xintercept = glmvar, linetype = 2) + 
    labs(caption = "*dotted line represents the Classic Lasso point estimate", face = "italic")
}

visModel <- function(varNum, varName, glmvar){
    post1 <- as.matrix(bayes_lasso_1)
    post2 <- as.matrix(bayes_lasso_2)
    post3 <- as.matrix(bayes_lasso_3)
    LASSO1 <- post1[, varNum]
    LASSO2 <- post2[, varNum]
    LASSO3 <- post3[, varNum]
    x <- data.frame(LASSO1, LASSO2, LASSO3)
    data<- melt(x)
    p <- ggplot(data,aes(x=value, color=variable))+ labs(subtitle= (varName), caption = "*dotted line represents the Classic Lasso point estimate") + geom_density(alpha=0.25) + geom_vline(xintercept = glmvar, linetype = 2) 
    p + labs(col = "Models")
}

# hier vul ik de functie in voor beide variabelen. glmvar is de glmnet estimate van de klassieke lasso. Deze is te vinden in de summary van de glmnet output

library(reshape2)
visModel(44, "Failures.x_0", 2.25634741)

visModel(11,"Medu_2", 0)
colnames(test)


```

Coef plot reconstruction for large selection of variables for all models including the classic lasso 

```{r}
#coefficients lasso 1,2,3 as df

lasso_1_coef <- fixef(bayes_lasso_1)
lasso_1_coef_1 <- as.data.frame(lasso_1_coef)
lasso_1_coef_1$model <- 'Lasso 1'

lasso_1_coef_1$par <- rownames(lasso_1_coef_1)


lasso_2_coef <- fixef(bayes_lasso_2)
lasso_2_coef_1 <- as.data.frame(lasso_2_coef)
lasso_2_coef_1$model <- 'Lasso 2'
lasso_2_coef_1$par <- rownames(lasso_2_coef_1)


lasso_3_coef <- fixef(bayes_lasso_3)
lasso_3_coef_1 <- as.data.frame(lasso_3_coef)
lasso_3_coef_1$model <- 'Lasso 3'
lasso_3_coef_1$par <- rownames(lasso_3_coef_1)

#coefficients classic lasso as df

bbb <- coef.glmnet(lasso, s=best_lambda)
classic_lasso_coef_1 <- as.matrix(bbb)
classic_lasso_coef_1
classic_lasso_coefs <- as.data.frame(as.matrix(bbb))
colnames(classic_lasso_coefs) <- "Estimate"
colnames(classic_lasso_coefs)
classic_lasso_coefs$Est.Error <- NA
classic_lasso_coefs$Q2.5 <- NA
classic_lasso_coefs$Q97.5 <- NA
classic_lasso_coefs$model <- 'Classic lasso'
classic_lasso_coefs$par <- rownames(classic_lasso_coefs)


#combining the coef dfs


combined_all <- rbind.data.frame(lasso_1_coef_1, lasso_2_coef_1, lasso_3_coef_1, classic_lasso_coefs)
which(combined_all$par == "Intercept")
combined_all_2 <- combined_all[-c(which(combined_all$par == "Intercept")), ]

combined_all_3 <- combined_all_2[-c(which(combined_all_2$par == "(Intercept)")), ]

combined_all_3 <- combined_all_2[-c(which(combined_all_2$par == "(Intercept)")), ]

#code above removes the intercepts from the df




x <- combined_all_3[c(which(combined_all_3$par %in% c("failures.x_0", "sex_M", "Fjob_teacher", "higher.y_yes", 
                                                      "freetime.y_2", "internet_yes", "Fedu_4", "Medu_4", "Fedu_1", "Mjob_other",
                                                      "Fjob_other", "failures.x_3", "schoolsup.x_yes", "Walc.y_4", "studytime.x_3", "guardian.x_father",
                                                      "guardian.x_other", "Dalc.x_1", "health.y_1", "Medu_2"))), ]

# Finally de ggplot met alles erin



library(ggplot2)
ggplot(x, aes(x = Estimate, y = par, group = model)) + geom_point(aes(color = model),position=position_dodge(width=0.4)) +
  geom_errorbar(aes(xmin = Q2.5, xmax = Q97.5, color = model), position=position_dodge(width=0.4))

```
