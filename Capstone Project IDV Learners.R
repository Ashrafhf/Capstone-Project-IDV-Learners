#3Loading all needed libraries

library(ggplot2) 
library(readr) 
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(ROCR)
library(GGally)
library(datasets)
library(haven)
library(olsrr)
library(dplyr)
library(caTools)
library(mice)
library(mctest)
library(corrplot)
library(car)
library(tree)
library(textir)
library(naivebayes)
library(ROCR)
library(VIM)
library(class)
library(neuralnet)

####Diabetes patient's dataset from https://www.kaggle.com/uciml/pima-indians-diabetes-database
dataset <- read_csv("C:/Users/eng_a/OneDrive/Desktop/datasets_2.csv")
View(dataset)
attach(dataset)
summary(dataset)



#Creating subset to treat missing values
dataset1=select(dataset,Glucose,BloodPressure,SkinThickness,Insulin,BMI)

dataset1[dataset1=='0']=NA#Replacing 0 with NA 
summary(dataset1)#This will show the NA present in every individual variable
sum(is.na(dataset1))#To check total number is NA present 
#Replacing the treated variables with the untreated variables present in main data
dataset$Glucose=dataset1$Glucose
dataset$BloodPressure=dataset1$BloodPressure
dataset$SkinThickness=dataset1$SkinThickness
dataset$Insulin=dataset1$Insulin
dataset$BMI=dataset1$BMI
View(dataset)

#dataset <- na.omit(dataset) #Listwise Imputation method to remove NA 
#summary(dataset)
#View(dataset)
dataset<-mice(dataset,m=1)######imputation by using multiple imputation
dataset<-complete(dataset)
View(dataset)
summary(dataset)
##################check outliers(Mahalanobis)
Sx <- cov(dataset)
D2 <- mahalanobis(dataset, colMeans(dataset), Sx)
plot(density(D2, bw = 0.5),
     main="Squared Mahalanobis distances, n=768, p=8") ; rug(D2)
qqplot(qchisq(ppoints(100), df = 3), D2,
       main = expression("Q-Q plot of Mahalanobis" * ~D^2 *
                           " vs. quantiles of" * ~ chi[3]^2))
abline(0, 1, col = 'gray')
################boxplot for variables
par(mar=rep(2,4))
par(mfrow=c(1,8))
for(i in 1:8) {
  boxplot(dataset[,i], main=names(dataset)[i])
}
##############Check multicollinearity  and remove any one variable if VIF large than 5
mymodel<-lm(dataset$Outcome~.,data=dataset)
vif(mymodel)
##################################33convert outcome variable to factor
y <- as.factor(ifelse(dataset$Outcome == 0, "NotDiabetic", "Diabetic"))#####	Coding Response Feature 
summary(y)
plot(y)
set.seed(123)

############Split dataset to Train set &Test set

test_index <- createDataPartition(y, times = 1, p = 0.6, list = FALSE)
test_set <- dataset[-test_index, ]
train_set <-dataset[test_index, ]
hist(test_set$Outcome) ####to see how much diabetic and nondiabetic  in Test set.
#########################################################
######################SVM Algorithm #####################
#########################################################

############Support vector machine by using kernal=Radial############################
set.seed(235)
mod1<-svm(train_set$Outcome~.,data =train_set,kernal="Redial")

#prediction 

pred_svm<-predict(mod1, test_set)
pred_svm <- ifelse(pred_svm >= 0.5, 1, 0)

table(test_set$Outcome, pred_svm)
confusion_matrix<-table(Prediced=pred_svm, Actual=test_set$Outcome)######confusion matrix 

confusion_matrix
ctable <- as.table(confusion_matrix, nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin =1, main = "confusion_matrix for Radial")
###########calculated missclasification
missclassification =1-sum(diag(confusion_matrix))/sum(confusion_matrix) 
missclassification
Acc=1-missclassification    ###calculated accuracy
Acc
sensitivity(confusion_matrix)
specificity(confusion_matrix)
#########Support vector machine by using kernal=polynomial

mod3<-svm(Outcome~.,data =train_set,kernel="polynomial")
summary(mod3)

pred_svm<-predict(mod3, test_set)
pred_svm <- ifelse(pred_svm > 0.5, 1, 0)

table(test_set$Outcome, pred_svm)
######confusion matrix 
confusion_matrix<-table(Prediced=pred_svm, Actual=test_set$Outcome)

confusion_matrix
ctable <- as.table(confusion_matrix, nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin =1, main = "confusion_matrix for polynomial ")

###########calculated missclasification
missclassification =1-sum(diag(confusion_matrix))/sum(confusion_matrix)  
missclassification
Acc=1-missclassification    ###calculated accuracy
Acc
sensitivity(confusion_matrix)
specificity(confusion_matrix)
######################Support vector machine by using kernal=linear
mod4<-svm(Outcome~.,data =train_set,kernel="linear")
summary(mod4)
#prediction 

pred_svm<-predict(mod1, test_set)
pred_svm <- ifelse(pred_svm > 0.5, 1, 0)

table(test_set$Outcome, pred_svm)
confusion_matrix<-table(Prediced=pred_svm, Actual=test_set$Outcome)######confusion matrix 

confusion_matrix
ctable <- as.table(confusion_matrix, nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin =1, main = "confusion_matrix for linear")


###########calculated missclasification
missclassification =1-sum(diag(confusion_matrix))/sum(confusion_matrix) 
missclassification
Acc=1-missclassification    ###calculated accuracy
Acc
sensitivity(confusion_matrix)
specificity(confusion_matrix)
####################################
mod5<-svm(Outcome~.,data =train_set,kernel="sigmoid")
summary(mod5)
#prediction 

pred_svm<-predict(mod5, test_set)
pred_svm <- ifelse(pred_svm > 0.5, 1, 0)

table(test_set$Outcome, pred_svm)
confusion_matrix<-table(Prediced=pred_svm, Actual=test_set$Outcome)######confusion matrix 

confusion_matrix
ctable <- as.table(confusion_matrix, nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin =1, main = "confusion_matrix for sigmoid")
###########calculated missclasification
missclassification =1-sum(diag(confusion_matrix))/sum(confusion_matrix) 
missclassification
Acc=1-missclassification    ###calculated accuracy
Acc
sensitivity(confusion_matrix)
specificity(confusion_matrix)
##############Tuning
tmodel<-tune(svm,Outcome~.,data = train_set,
             ranges = list(epsilon=seq(0,1,0.1), cost=2^(2:9)))
plot(tmodel)
summary(tmodel)     

########best model
bestmodel<-tmodel$best.model
summary(bestmodel)
mod6 <- svm(Outcome~.,data=train_set,kernel='radial',gamma=0.125,cost=1,epslon=0.6)


pred_svm<-predict(mod6, test_set)
pred_svm <- ifelse(pred_svm > 0.5, 1, 0)

table(test_set$Outcome, pred_svm)
confusion_matrix<-table(Prediced=pred_svm, Actual=test_set$Outcome)######confusion matrix 

confusion_matrix
ctable <- as.table(confusion_matrix, nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin =1, main = "confusion_matrix for best model")
###########calculated missclasification
missclassification =1-sum(diag(confusion_matrix))/sum(confusion_matrix) 
missclassification
Acc=1-missclassification    ###calculated accuracy
Acc
sensitivity(confusion_matrix)
specificity(confusion_matrix)
###############################  Decision tree algorithm  ###############################
tree.model<-rpart(Outcome~.,data = train_set,minbucket = 20)
tree.model
tree.model <- rpart(Outcome ~ ., data = train_set, method = "class")
prp(tree.model) 
tree.predict <- predict(tree.model, test_set, type = "class")
table<-table(test_set$Outcome, tree.predict)
tree<-confusionMatrix(tree.predict,factor(test_set$Outcome))
acc_tree <- tree$overall['Accuracy']
acc_tree
tree

ctable <- as.table(table, nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin =1, main = "confusion_matrix for Decision tree ")


#######################Naive base#####################


naive_bayes<-train(factor(Outcome)~.,data=train_set,method="naive_bayes")

naive_bayes
p<-predict(naive_bayes,test_set)
p
c<- confusionMatrix(factor(p),factor(test_set$Outcome))
c

ctable <- as.table(c, nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin =1, main = "confusion_matrix for Naive Base ")

######Neural Netwok
nn <- neuralnet(Outcome~ ., data=train_set, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)
#Test the resultning output
temp_test <- subset(test_set, select = c("Pregnancies","Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI","DiabetesPedigreeFunction","Age"))
head(temp_test)

nn.results<-neuralnet::compute(nn, temp_test)

results <- data.frame(actual = test_set$Outcome, prediction = nn.results$net.result)#Test the resulting output
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(prediction,actual)
con<-confusionMatrix(table(actual,prediction))

ctable <- as.table(con, nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin =1, main = "confusion_matrix for Neural Network")

