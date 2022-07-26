#https://robjhyndman.com/hyndsight/dailydata/

SY_Flow <- read.csv("SY_Flow.csv")

y <- ts(SY_Flow[2], frequency=7)
library(forecast)
fit <- ets(y)
fc <- forecast(fit, h=30)
plot(fc)

y <- msts(SY_Flow[2], seasonal.periods=c(180,365.25))
fit <- tbats(y)
fc <- forecast(fit, h=365*2)
plot(fc)

y <- msts(SY_Flow[2], seasonal.periods=c(7,365.25))
fit <- tbats(y)
fc <- forecast(fit, h=365*2)
plot(fc)

y <- msts(SY_Flow[2], seasonal.periods=c(7,365.25))
fit <- tbats(y)
fc <- forecast(fit, h=7)
plot(fc)