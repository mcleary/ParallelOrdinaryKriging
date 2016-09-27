library(kriging)
library(foreign)
#library(maps)
 
# Create some random data

print('Reading data...')
#data = read.table('/Users/tsabino/Desktop/mpi-dtm2.xyz')
data = read.table('D:/Dropbox/Doutorado/arvores/mpi-dtm2.xyz')
x <- as.double(data$V1)
y <- as.double(data$V2)
z <- as.double(data$V3)

#x <- runif(50, min(p[[1]][,1]), max(p[[1]][,1]))
#y <- runif(50, min(p[[1]][,2]), max(p[[1]][,2]))
#z <- rnorm(50)

print('Kriging ...')
print(system.time(kriged <- kriging(x, y, z, pixels = 30)))

image(kriged, xlim = extendrange(x), ylim = extendrange(y))

