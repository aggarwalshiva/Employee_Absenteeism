setwd("C:/Users/Shivam/Desktop/Data Science edWisor/PROJECT/Cab Rental")
getwd()

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

## Reading Data
df_train = read.csv("train_cab.csv",header = T,na.strings = c(""," ","NA"))

############# Explore the Dataset #######################
str(df_train)

############# Missing Value Analysis ########################

missing_val = data.frame(apply(df_train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] = "Missing_Percentage"
missing_val$Missing_Percentage = (missing_val$Missing_Percentage/nrow(df_train)) * 100
missing_val = missing_val[order(-missing_val$Missing_Percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]


### "fare_amount" is our target variable.
## We have 24 observations having values 'NA' so we drop these values.
df_train = DropNA(df_train , Var= "fare_amount")

## As we have total of 16043 observation , and a missing observation 55 from "passenger_count" which is approx 0.34 %.
## So I drop these observation, inplace of imputing them by mean,median.
### We are not using Knn Imputaion on single variable.
df_train = DropNA(df_train,Var="passenger_count")


### Histogram plot
hist(df_train$passenger_count)
hist(df_train$fare_amount)
hist(df_train$pickup_longitude)
hist(df_train$pickup_latitude)
hist(df_train$dropoff_longitude)
hist(df_train$dropoff_latitude)


#### Observing Data
str(df_train)

## by observing the data we came to know mininmum value of 'fare_amount' is negative, which is not pratical.
## So we remove those observation from data.
df_train = df_train[which(df_train$fare_amount > 0 ),]


## Let's focus on longitude and latitude.

cat("The maximum pickoff_longitude: ", max(df_train$pickup_longitude))
cat("The minimum pickoff_longitude: ",min(df_train$pickup_longitude))
cat("The maximum pickoff_latitude: ",max(df_train$pickup_latitude))
cat("The minimum pickoff_latitude: ",min(df_train$pickup_latitude))
cat("The maximum dropoff_longitude: ",max(df_train$dropoff_longitude))
cat("The minimum dropoff_longitude: ",min(df_train$dropoff_longitude))
cat("The maximum dropoff_latitude: ",max(df_train$dropoff_latitude))
cat("The minimum dropoff_latitude: ",min(df_train$dropoff_latitude))

### from above observation we came to know that coordinates is belongs to NEW YORK.
## So,

# our latitude and longitude should not be equal to 0 becuase the dataset is based in NY
df_train = df_train[which(df_train$pickup_latitude != 0),]
df_train = df_train[which(df_train$pickup_longitude != 0),]
df_train = df_train[which(df_train$dropoff_latitude !=0),]
df_train = df_train[which(df_train$dropoff_longitude !=0),]


# latitude and longitude are bounded by 90 and -90. We shouldnt have any coordiantes out of that range
df_train = df_train[which((df_train$pickup_latitude <=90)&(df_train$pickup_latitude>=-90)),]
df_train = df_train[which((df_train$pickup_longitude <=90)&(df_train$pickup_longitude >=-90)),]
df_train = df_train[which((df_train$dropoff_latitude <=90)&(df_train$dropoff_latitude>=-90)),]
df_train = df_train[which((df_train$dropoff_longitude <=90)&(df_train$dropoff_longitude>=-90)),]

# I dont want to include destinations that have not moved from there pickup coordinates to there dropoff coordinates
df_train = df_train[which((df_train$pickup_longitude != df_train$dropoff_longitude) & (df_train$pickup_latitude != df_train$dropoff_latitude)),]

##################### Outlier Analysis #########
## Outlier analysis for "fare_amount"
boxplot(df_train$fare_amount)

val = df_train$fare_amount[df_train$fare_amount %in% boxplot.stats(df_train$fare_amount)$out]
df_train[,"fare_amount"][df_train[,"fare_amount"] %in% val] = NA


###  1348 outler in target variable "fare_amount".
### So we are first going to drop outlier from target variable "fare_amount"
df_train = DropNA(df_train,Var = "fare_amount")


##### Now we do analysis for "passenger_count" #########
table(df_train$passenger_count)

df_train = df_train[which((df_train$passenger_count < 7) & (df_train$passenger_count > 0)),]

## Distribution of "fare_amount" after outlier removal,
hist(df_train$fare_amount)

### Convert pickup_datetime from object to datetime object
df_train$pickup_datetime = as.POSIXlt(df_train$pickup_datetime , format= "%Y-%m-%d %H:%M:%S UTC")
df_train$pickup_year = as.numeric(format(df_train$pickup_datetime,"%Y"))
df_train$pickup_day = as.numeric(format(df_train$pickup_datetime,"%d"))
df_train$pickup_month = as.numeric(format(df_train$pickup_datetime,"%m"))
df_train$pickup_hour = as.numeric(format(df_train$pickup_datetime,"%H"))

#### We have 1 "NA" value in pickup_datetime.
df_train = DropNA(df_train,Var="pickup_datetime")


##Haversine Distance
haversine_distance = function(lon1, lat1, lon2, lat2){
  # convert decimal degrees to radians
  lon1 = lon1 * pi / 180
  lon2 = lon2 * pi / 180
  lat1 = lat1 * pi / 180
  lat2 = lat2 * pi / 180
  # haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  km = 6367 * c
  km
}

df_train$haversine_dist = haversine_distance(df_train$pickup_longitude,df_train$pickup_latitude,df_train$dropoff_longitude,df_train$dropoff_latitude)

### Sampling

df_train = subset(df_train,select=-c(pickup_datetime))

set.seed(1234)
data_index = createDataPartition(df_train$fare_amount , p=0.80,list = FALSE)
train_data = df_train[data_index,]
test_data = df_train[-data_index,]

#### Modelling
## Linear Regression

lm_model = lm(fare_amount ~., data = train_data)

#Summary of the model
summary(lm_model)

#Predict
predictions_LR = predict(lm_model, test_data[,2:11])

## Metrics Evaluation
regr.eval(test_data[,1],predictions_LR,stats = c("mae","mape","rmse"))
#mae       mape       rmse 
#3.5290356  0.4568612 28.9238505


##################### Linear Regression with Significant Variables #############
lm_model_signi = lm(fare_amount ~pickup_latitude+dropoff_longitude+dropoff_latitude+passenger_count+pickup_year+pickup_month+haversine_dist,data=train_data)
predictions_LRS = predict(lm_model_signi, test_data[,2:11])
regr.eval(test_data[,1],predictions_LRS,stats = c("mae","mape","rmse"))
#mae       mape       rmse 
#3.5166479  0.4554315 28.2546451


## Decision Tree

decision_tree = rpart(fare_amount ~.,data= train_data,method = "anova")
summary(decision_tree)

## Prediction
predictions_DT = predict(decision_tree, test_data[,2:11])
### Evaluation Metrics
regr.eval(test_data[,1],predictions_DT,stats = c("mae","mape","rmse"))
#mae      mape      rmse 
#1.7055753 0.2027999 2.3894631

## Random Forest
library(randomForest)

rf_model= randomForest(fare_amount ~.,train_data,importance=TRUE,ntree=500)

summary(rf_model)

rf_predict=predict(rf_model,test_data[,2:11])


regr.eval(test_data[,1],rf_predict,stats = c("mae","mape","rmse"))
#mae     mape     rmse (ntree = 200)  
#1.440986 0.174479 2.091133 

#mae      mape      rmse (ntree=500)
#1.4431025 0.1753406 2.0902512

#mae      mape      rmse (ntree=1000)
#1.4420241 0.1750757 2.0873624

#### We are selecting Random forest with ntree = 1000.

##### Now work with test data #########

df_test = read.csv("test.csv",header =T,na.strings = c(""," ","NA"))
str(df_test)


### Convert pickup_datetime from object to datetime object
df_test$pickup_datetime = as.POSIXlt(df_test$pickup_datetime , format= "%Y-%m-%d %H:%M:%S UTC")
df_test$pickup_year = as.numeric(format(df_test$pickup_datetime,"%Y"))
df_test$pickup_day = as.numeric(format(df_test$pickup_datetime,"%d"))
df_test$pickup_month = as.numeric(format(df_test$pickup_datetime,"%m"))
df_test$pickup_hour = as.numeric(format(df_test$pickup_datetime,"%H"))

df_test$haversine_dist = haversine_distance(df_test$pickup_longitude,df_test$pickup_latitude,df_test$dropoff_longitude,df_test$dropoff_latitude)

df_test = subset(df_test,select=-c(pickup_datetime))

##Getting the Predicted "fare_amount"

rf_predict=predict(rf_model,df_test[,1:10])

rf_predict
