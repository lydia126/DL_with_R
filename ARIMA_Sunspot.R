# https://rpubs.com/danielkurnia/timeseries_sunspots

# 2.0 Data
library(padr)

sunspot <- read.csv("Sunspots.csv") 
glimpse(sunspot)

#3 Data Import, Wrangling and EDA

sun_clean <- sunspot %>% 
  select(-X) %>% 
  mutate(Date = ymd(Date)) %>% 
  thicken(interval ="month", colname = "yearmonth") %>% # to convert "daily" observations to monthly observations as there is only one day per month  
  select(-"Date") %>%  # remove the original date column to use the yearmonth column as the new date column as basis for padding  
  pad(interval = "month")
anyNA(sun_clean)

sunspot_ts <- ts(data=sun_clean$Monthly.Mean.Total.Sunspot.Number,
                 start = c(1749,1),
                 frequency = 12)
autoplot(sunspot_ts)

autoplot(decompose(sunspot_ts))

sunspot_11yr <- ts(data=sun_clean$Monthly.Mean.Total.Sunspot.Number,
                   start = c(1749,1),
                   frequency = 12*11)
autoplot(decompose(sunspot_11yr))

sunspot_11yr_22yr <- msts(data = sun_clean$Monthly.Mean.Total.Sunspot.Number, seasonal.periods = c(12*11, 12*22))
sunspot_11yr_22yr %>% mstl() %>% autoplot()

sunspot_11yr_22yr_70yr <- msts(data = sun_clean$Monthly.Mean.Total.Sunspot.Number, seasonal.periods = c(12*11, 12*22, 12*70))
sunspot_11yr_22yr_70yr %>% mstl() %>% autoplot()

#4 Cross-Validation
sunspot_11yr_22yr_train <- sunspot_11yr_22yr %>%  head(-(12*11))
sunspot_11yr_22yr_test <- sunspot_11yr_22yr %>% tail(12*11)

# 5 Forecasting Model
#use stlm with the arima method.
model_11yr_22yr <- sunspot_11yr_22yr_train %>% stlm(
  method = "arima")
mdl_11yr_22yr_f <- forecast(model_11yr_22yr, h=12*11)
accuracy(mdl_11yr_22yr_f$mean, sunspot_11yr_22yr_test)

autoplot(sunspot_11yr_22yr %>% tail(12*11*5))+
  autolayer(mdl_11yr_22yr_f$mean, series = "prediction")+
  autolayer(sunspot_11yr_22yr_test, series = "actual test data")

#6 Attempts to improve the prediction
#6.1 Adding new seasonality
sunspot_11yr_22yr_70yr_train <- sunspot_11yr_22yr_70yr %>%  head(-(11*12))
sunspot_11yr_22yr_70yr_test <- sunspot_11yr_22yr_70yr %>% tail(11*12)

model_11yr_22yr_70yr <- sunspot_11yr_22yr_70yr_train %>% stlm(
  method = "arima")
mdl_11yr_22yr_70yr_f <- forecast(model_11yr_22yr_70yr, h=12*11)

accuracy(mdl_11yr_22yr_70yr_f$mean, sunspot_11yr_22yr_70yr_test)

autoplot(sunspot_11yr_22yr_70yr %>% tail(12*11*5))+
  autolayer(mdl_11yr_22yr_70yr_f$mean, series = "prediction")+
  autolayer(sunspot_11yr_22yr_70yr_test, series = "actual test data")

# 6.2 Subsetting from the whole data
newdata <- sunspot_11yr_22yr %>% tail(14*11*12)
autoplot(mstl(newdata))

newdata_train <- newdata %>% head(-(11*12))
newdata_test <- newdata %>% tail(11*12)

model2_11yr_22yr <- newdata_train %>% stlm(
  method = "arima")
mdl2_11yr_22yr_f <- forecast(model2_11yr_22yr, h=11*12)

accuracy(mdl2_11yr_22yr_f$mean, newdata_test)

autoplot(newdata %>% tail(12*11*5))+
  autolayer(mdl2_11yr_22yr_f$mean, series = "prediction")+
  autolayer(newdata_test, series = "actual test data")

# 6.3 Trying to predict a shorter time period
sunspot_11yr_22yr_train2 <- sunspot_11yr_22yr %>%  head(-(12))
sunspot_11yr_22yr_test2 <- sunspot_11yr_22yr %>% tail(12)

model3_11yr_22yr <- sunspot_11yr_22yr_train2 %>% stlm(
  method = "arima")
mdl3_11yr_22yr_f <- forecast(model3_11yr_22yr, h=12*11)
accuracy(mdl_11yr_22yr_f$mean, sunspot_11yr_22yr_test2)

#7 Assumptions check
Box.test(model_11yr_22yr$residuals, type = "Ljung-Box")
shapiro.test(model_11yr_22yr$residuals)
hist(model_11yr_22yr$residuals, breaks=50)
length(model_11yr_22yr$residuals)
