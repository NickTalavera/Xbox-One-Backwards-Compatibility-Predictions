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

data = data[sapply(sapply(data,unique), length) > 1]
# bcTemp = data$isBCCompatible
# data = as.data.frame(model.matrix(~ . -isBCCompatible -gameName -gameUrl -highresboxart, data))
# data$isBCCompatible = as.double(bcTemp) - 1
# names(data) = str_replace_all(names(data), "[^[:alnum:]]", "")
#===============================================================================
#                               TRAINING/TESTING                               #
#===============================================================================
# Make array of unwanted columns
unwantedPredictors = c("gameName","gameUrl","highresboxart","Intercept")
data = as.data.frame(VarDrop(data, unwantedPredictors))
include = which(data$isBCCompatible == TRUE | data$usesRequiredPeripheral == TRUE | data$isKinectRequired == TRUE)
# Training set
xb_train = data[include,]
train_ids = xb_train$gameName
# Test set
xb_test = data[-include,]
xb_test = VarDrop(xb_test, "isBCCompatible")
test_ids = xb_test$gameName
#===============================================================================
#                                   MODELS                                     #
#===============================================================================
#Inspecting the output to ensure that the range of each variable is now between
#0 and 1.
summary(data)

#Verifying that the split has been successfully made into 75% - 25% segments.
nrow(xb_train)/nrow(data)
nrow(xb_test)/nrow(data)

#Loading the neuralnet library for the training of neural networks.
library(neuralnet)

#Training the simplest multilayer feedforward neural network that includes only
#one hidden node.
set.seed(0)
# xb_train$isBCCompatible = as.numericxb_train$isBCCompatible)
# xb_train$isBCCompatible = as.numeric(xb_train$isBCCompatible)
# xb_train = xb_train[sapply(xb_train,is.numeric) | sapply(xb_train,is.logical)]

# n <- names(xb_train)[names(xb_train) != 'Intercept']
# f <- as.formula(paste("as.factor(isBCCompatible) ~", paste(n[!n %in% "isBCCompatible"], collapse = " + ")))


library(e1071)
concrete_model = naiveBayes(isBCCompatible ~ .,
                           data = xb_train)
xb_train$isBCCompatible
#Generating model predictions on the testing dataset using the compute()
#function.

model_results = compute(concrete_model, xb_test)

#The model_results object stores the neurons for each layer in the network and
#also the net.results which stores the predicted values; obtaining the
#predicted values.
predicted_strength = as.data.frame(cbind(as.character(test_ids), as.logical(round(model_results$net.result - 1))))
predicted_strength = dplyr::select(predicted_strength, gameName = V1, predicted_isBCCompatible = V2)
dataOut = merge(x = dataOriginal, y = predicted_strength, by = "gameName", all.x = TRUE)
dataOut[is.na(dataOut$predicted_isBCCompatible),]$predicted_isBCCompatible = FALSE
dataOut$predicted_isBCCompatible[dataOut$isBCCompatible == TRUE] = dataOut$isBCCompatible[dataOut$isBCCompatible == TRUE]

# # #Examining the correlation between predicted and actual values.
# cor(as.logical(dataOut$predicted_isBCCompatible), as.logical(dataOut$isBCCompatible))
# plot(dataOut$predicted_isBCCompatible, dataOut$isBCCompatible)

# write.csv(submission, file = file.path(dataLocale, "dataWPrediction.csv"), row.names = FALSE)
print("...Done!")

# Stop parallel clusters
stopCluster(cl)