rm(list = ls())
#===============================================================================
#                       LOAD PACKAGES AND MODULES                              #
#===============================================================================
library("data.table")
library("DataCombine")
library(dplyr)
na_count <-function (x) sapply(x, function(y) sum(is.na(y)))
#===============================================================================
#                                SETUP PARALLEL                                #
#===============================================================================
library(foreach)
library(parallel)
library(doParallel)
if(!exists("cl")){
  library(doParallel)
  cores.number = detectCores(all.tests = FALSE, logical = TRUE) -1
  cl = makeCluster(2)
  registerDoParallel(cl, cores=cores.number)
}


#===============================================================================
#                                 IMPORT DATA                                  #
#===============================================================================
if (dir.exists('/home/bc7_ntalavera/Dropbox/Data Science/Data Files/Xbox Back Compat Data/')) {
  dataLocale = '/home/bc7_ntalavera/Dropbox/Data Science/Data Files/Xbox Back Compat Data/'
} else if (dir.exists('/Users/nicktalavera/Coding/Data Science/Xbox-One-Backwards-Compatability-Predictions/Xbox Back Compat Data/')) {
  dataLocale = '/Users/nicktalavera/Coding/Data Science/Xbox-One-Backwards-Compatability-Predictions/Xbox Back Compat Data/'
  # setwd('/Users/nicktalavera/Coding/Data Science/Xbox-One-Backwards-Compatability-Predictions/Xbox Back Compat Data')
}  else if (dir.exists('/home/bc7_ntalavera/Data/Xbox/')) {
  dataLocale = '/home/bc7_ntalavera/Data/Xbox/'
}
markdownFolder = paste0(dataLocale,'MarkdownOutputs/')
dataOriginal = data.frame(fread(paste0(dataLocale,'dataUlt.csv'), stringsAsFactors = TRUE, drop = c("V1")))
data = data.frame(fread(paste0(dataLocale,'dataUltImputed.csv'), stringsAsFactors = TRUE, drop = c("V1")))
nums <- parSapply(cl = cl, data, is.logical)
nums = names(nums[nums==TRUE])
for (column in nums) {
  data[,column] = as.factor(data[,column])
}
na_count(data)
data$gameName = as.character(data$gameName)
data$releaseDate = as.numeric(data$releaseDate)
#===============================================================================
#                               PREPARE DATA                                   #
#===============================================================================
#We now apply our normalization function to all the variables within our dataset;
#we store the result as a data frame for future manipulation.
library(Ecfun)
library(normalr)

normalize = function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# Box cox transform continuous variables
# lambdaVal = getLambda(data[sapply(data, is.numeric)], lambda = seq(-10, 10, 1/100), parallel = TRUE)
# data[sapply(data, is.numeric)] = BoxCox(data[sapply(data, is.numeric)], lambdaVal, rescale=FALSE)
# Scale and center continuous variables
# data[sapply(data, is.numeric)] <- as.data.frame(lapply(data[sapply(data, is.numeric)], normalize))
# 
data = data[sapply(sapply(data,unique), length) > 1,]
data = data[sapply(data,is.numeric),]
# bcTemp = data$isBCCompatible
# data = as.data.frame(model.matrix(~ . -gameName -gameUrl -highresboxart, data))
# data$isBCCompatible = as.double(bcTemp)
# library(stringr)
# names(data) = str_replace_all(names(data), "[^[:alnum:]]", "")
#===============================================================================
#                               TRAINING/TESTING                               #
#===============================================================================
# Make array of unwanted columns
unwantedPredictors = c("gameName","gameUrl","highresboxart","Intercept")
data = VarDrop(data, unwantedPredictors)
tval = 1
# Training set
xb_train = data[which(data$isBCCompatible == tval | data$usesRequiredPeripheral == tval | data$isKinectRequired == tval),]
train_ids = xb_train$gameName
# Test set
xb_test = data[-which(data$isBCCompatible == tval | data$usesRequiredPeripheral == tval | data$isKinectRequired == tval),]
test_ids = xb_test$gameName
# data = VarDrop(xb_test, "isBCCompatible")
#===============================================================================
#                                   MODELS                                     #
#===============================================================================
str(data)
summary(data)

#Inspecting the output to ensure that the range of each variable is now between
#0 and 1.
summary(concrete_norm)

#Since the data has already been organized in random order, we can simply split
#our data into training and test sets based on the indices of the data frame;
#here we create a 75% training set and 25% testing set split.
concrete_train = concrete_norm[1:773, ]
concrete_test = concrete_norm[774:1030, ]

#Verifying that the split has been successfully made into 75% - 25% segments.
nrow(concrete_train)/nrow(concrete_norm)
nrow(concrete_test)/nrow(concrete_norm)

#Loading the neuralnet library for the training of neural networks.
library(neuralnet)

#Training the simplest multilayer feedforward neural network that includes only
#one hidden node.
set.seed(0)
concrete_model = neuralnet(strength ~ cement + slag +     #Cannot use the shorthand
                             ash + water + superplastic + #dot (.) notation.
                             coarseagg + fineagg + age,
                           hidden = 1, #Default number of hidden neurons.
                           data = concrete_train)

#Visualizing the network topology using the plot() function.
plot(concrete_model)

#Generating model predictions on the testing dataset using the compute()
#function.
model_results = compute(concrete_model, concrete_test[, 1:8])

#The model_results object stores the neurons for each layer in the network and
#also the net.results which stores the predicted values; obtaining the
#predicted values.
predicted_strength = model_results$net.result

#Examining the correlation between predicted and actual values.
cor(predicted_strength, concrete_test$strength)
plot(predicted_strength, concrete_test$strength)