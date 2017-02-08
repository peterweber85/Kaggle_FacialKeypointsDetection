
library(doMC)
library(reshape2)
library(caTools)

registerDoMC()

setwd("~/Projects/Git/Kaggle/Facial_keypoints_detection/")

train.file = 'training.csv'
test.file = 'test.csv'
patch_size  <- 10
search_size <- 2
Train_split_as_test <- TRUE




if (Train_split_as_test == TRUE){
    
    ### caTools solution
    set.seed(0) 
    data  <- read.csv(train.file, stringsAsFactors=F)
    im <- foreach(im = data$Image, .combine=rbind) %dopar% {
      as.integer(unlist(strsplit(im, " ")))
    }
    data$Image  <- NULL
    sample      <- sample.split(data, SplitRatio = .8)
    train       <- subset(data, sample == TRUE)
    test        <- subset(data, sample == FALSE)
    train.image <- subset(im, sample == TRUE)
    test.image  <- subset(im, sample == FALSE)
    rm("data", "im")
} else {
  
    train = read.csv(train.file, stringsAsFactors = F)
    train.image = train$Image
    train$Image = NULL
    train.image <- foreach(im = train.image, .combine=rbind) %dopar% {
      as.integer(unlist(strsplit(im, " ")))
    }
    
    test  <- read.csv(test.file, stringsAsFactors=F)
    test.image <- foreach(im = test$Image, .combine=rbind) %dopar% {
      as.integer(unlist(strsplit(im, " ")))
    }
    test$Image <- NULL
  
}



### Start from here if data is preprocessed and saved
# save(train, train.image, test, test.image, file='data.Rd')
# load('data.Rd')
# patch_size  <- 10
# search_size <- 2


### list the coordinates we have to predict
coordinate.names <- gsub("_x", "", names(train)[grep("_x", names(train))])



# for each one, compute the average patch
mean.patches <- foreach(coord = coordinate.names) %dopar% {
  cat(sprintf("computing mean patch for %s\n", coord))
  coord_x <- paste(coord, "x", sep="_")
  coord_y <- paste(coord, "y", sep="_")
  
  # compute average patch
  patches <- foreach (i = 1:nrow(train), .combine=rbind) %do% {
    im  <- matrix(data = train.image[i,], nrow=96, ncol=96)
    x   <- train[i, coord_x]
    y   <- train[i, coord_y]
    x1  <- (x-patch_size)
    x2  <- (x+patch_size)
    y1  <- (y-patch_size)
    y2  <- (y+patch_size)
    if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
    {
      as.vector(im[x1:x2, y1:y2])
    }
    else
    {
      NULL
    }
  }
  matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)
}

# for each coordinate and for each test image, find the position that best correlates with the average patch
p <- foreach(coord_i = 1:length(coordinate.names), .combine=cbind) %dopar% {
  # the coordinates we want to predict
  coord   <- coordinate.names[coord_i]
  coord_x <- paste(coord, "x", sep="_")
  coord_y <- paste(coord, "y", sep="_")
  
  # the average of them in the training set (our starting point)
  mean_x  <- mean(train[, coord_x], na.rm=T)
  mean_y  <- mean(train[, coord_y], na.rm=T)
  
  # search space: 'search_size' pixels centered on the average coordinates 
  x1 <- as.integer(mean_x)-search_size
  x2 <- as.integer(mean_x)+search_size
  y1 <- as.integer(mean_y)-search_size
  y2 <- as.integer(mean_y)+search_size
  
  # ensure we only consider patches completely inside the image
  x1 <- ifelse(x1-patch_size<1,  patch_size+1,  x1)
  y1 <- ifelse(y1-patch_size<1,  patch_size+1,  y1)
  x2 <- ifelse(x2+patch_size>96, 96-patch_size, x2)
  y2 <- ifelse(y2+patch_size>96, 96-patch_size, y2)
  
  # build a list of all positions to be tested
  params <- expand.grid(x = x1:x2, y = y1:y2)
  
  # for each image...
  r <- foreach(i = 1:nrow(test), .combine=rbind) %do% {
    if ((coord_i==1)&&((i %% 100)==0)) { cat(sprintf("%d/%d\n", i, nrow(test))) }
    im <- matrix(data = test.image[i,], nrow=96, ncol=96)
    
    # ... compute a score for each position ...
    r  <- foreach(j = 1:nrow(params), .combine=rbind) %do% {
      x     <- params$x[j]
      y     <- params$y[j]
      p     <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
      score <- cor(as.vector(p), as.vector(mean.patches[[coord_i]]))
      score <- ifelse(is.na(score), 0, score)
      data.frame(x, y, score)
    }
    
    # ... and return the best
    best <- r[which.max(r$score), c("x", "y")]
  }
  names(r) <- c(coord_x, coord_y)
  r
}
validation = sqrt(mean((test-p)^2, na.rm=T))


### 
library(reshape2)

predictions = data.frame(ImageId = 1:nrow(test), p)
submission = melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")


### Sample submission
sample_submission = read.csv("SampleSubmission.csv")

### 
IdLookupTable = read.csv('IdLookupTable.csv')
sub.col.names = names(sample_submission)
IdLookupTable$Location = NULL

submission = merge(IdLookupTable, submission, all.x=T, sort=F)
submission = submission[, sub.col.names]
write.csv(submission, file="submission_means.csv", quote=F, row.names=F)
