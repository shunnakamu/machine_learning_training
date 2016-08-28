iris.km <- kmeans(iris[,-5], 3)
iris.km$cluster
iris.pc <- prcomp(iris[1:4])

# plot result
par(mfrow=c(2,2))
# left: actual, right; cluster
plot(iris.pc$x[,1], iris.pc$x[,2], pch = 21, bg = c("red", "green3", "blue")[unclass(iris$Species)])
plot(iris.pc$x[,1], iris.pc$x[,2], pch = 21, bg = c("red", "green3", "blue", " black ", " white ")[unclass(iris.km$cluster)])

# if cluster number is 5
iris.km <- kmeans(iris[,-5], 5)
iris.km$cluster
iris.pc <- prcomp(iris[1:4])
par(mfrow=c(2,2))
plot(iris.pc$x[,1], iris.pc$x[,2], pch = 21, bg = c("red", "green3", "blue")[unclass(iris$Species)])
plot(iris.pc$x[,1], iris.pc$x[,2], pch = 21, bg = c("red", "green3", "blue", " black ", " white ")[unclass(iris.km$cluster)])
