# load airquality data
data(airquality)

# check data summary
nrow(airquality)
names(airquality)

# linear regression
airquality.lm <- lm(Ozone ~ Solar.R + Wind + Temp, data=airquality)
summary(airquality.lm)

# see Multiple R-Squared
airquality.lm$residuals

# see average of error
mean(abs(airquality.lm$residuals))

# see function coefficients
airquality.lm$coefficients
