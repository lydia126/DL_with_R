# https://rpubs.com/danielkurnia/timeseries_sunspots

# 2.0 Data
library(padr)
library(dplyr)
library(lubridate)
library(forecast)

SY_Flow <- read.csv("SY_Flow_2016_2021.csv") 
glimpse(SY_Flow)

#3 Data Import, Wrangling and EDA
sy_flow_ts <- ts(data=SY_Flow[2],
                 start = c(2016,1),
                 frequency = 365)
autoplot(sy_flow_ts)

# 5 Forecasting Model
#use stlm with the arima method.
sy_flow_train <- sy_flow_ts %>% head(-(365*2))  #train from 2016 ~ 2020
sy_flow_test <- sy_flow_ts %>%  tail(365*2) #test 2021

model_sy_flow <- sy_flow_train %>% stlm(
  method = "arima")
sy_flow_forecasted <- forecast(model_sy_flow, h=365*2)
accuracy(sy_flow_forecasted$mean, sy_flow_test)

autoplot(sy_flow_ts %>% tail(365*2))+
  autolayer(sy_flow_forecasted$mean, series = "prediction")+
  autolayer(sy_flow_test <- sy_flow_ts, series = "actual test data")

#7 Assumptions check
Box.test(model_sy_flow$residuals, type = "Ljung-Box")
shapiro.test(model_sy_flow$residuals)
hist(model_sy_flow$residuals, breaks=50)
length(model_sy_flow$residuals)

# autoarima model  --- 예측결과가 직선으로 나옴
library(tseries)
adf.test(sy_flow_train)
ndiffs(sy_flow_train )
plot(sy_flow_train )
acf(sy_flow_train , lwd=2,
    main="AUtocorrelation for daily flow data")
pacf(sy_flow_train , lwd=2,
     main="Partial autocorrelation of daily flow data")

sy_flow.arima <- auto.arima(sy_flow_train)
sy_flow.arima
sy_flow_arima.f <- forecast(sy_flow.arima, h=365*5)
accuracy(sy_flow_arima.f$mean, sy_flow_test)

autoplot(sy_flow_ts %>% tail(365*5))+
  autolayer(sy_flow_arima.f$mean, series = "prediction")+
  autolayer(sy_flow_test <- sy_flow_ts, series = "actual test data")

plot(forecast(sy_flow.arima, h=365*5), col="darkorange", lwd=2,
     flty=1, flwd=3,
     fcol="orangered", shadecols = c("lavender","skyblue"),
     xlab="Year", ylab="Flow",
     main="Forecast for Flow for 5 years")