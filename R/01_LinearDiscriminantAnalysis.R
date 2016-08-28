# load iris data
data(iris)

# check data summary
nrow(iris)
names(iris)

# set odd & even number
odd.n <- 2*(1:50) -1
even.n <- 2*(1:50)
# check
odd.n
even.n

# divide data into train and test
iris.train <- iris[odd.n,]
iris.test <- iris[even.n,]

# load library
library(MASS)

# lda (linear discriminant analysis)
iris.lda <- lda(Species~., data=iris.train)
# ignore warning
# In lda.default(x, grouping, ...) : group virginica is empty
iris.lda

# plot summary
plot(iris.lda, dimen=1)
# predict
iris.pre <- predict(iris.lda, iris.test[,-5])
table(iris.test[,5], iris.pre$class)

x <- data.frame("Sepal.Length" = 1, "Sepal.Width" = 2, "Petal.Length" = 3, "Petal.Width" = 4)
predict(iris.lda, x)$class
