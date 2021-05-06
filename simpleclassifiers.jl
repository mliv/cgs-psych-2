#= Simple classifiers
This is where we stand using the default parameters for 5 basic classifiers
Start the Julia REPL, drag to highlight a section, and use ctrl-enter to run it
This only needs one file, audio_recoded_combined_binary.csv, in the base folder
-Matei 
=#

# Logistic Regression ##########################################################
# https://coinslab.github.io/CognitiveModelingLabWebsite/page/LogisticRegression/
#
# Results (test data): 
#               precision    recall  f1-score   support
#            0       0.84      1.00      0.91     22560
#            1       0.00      0.00      0.00      4332
#     accuracy                           0.84     26892
#    macro avg       0.42      0.50      0.46     26892
# weighted avg       0.70      0.84      0.77     26892
# 
# Note that training data looks the same, it's not overfitting/memorizing
# it's just failing completely and guessing false on all inputs
#
# Cross-validation:               
#   0.8362637362637363
#   0.8364468864468865
#   0.8362637362637363
#   0.8361571572488323
#   0.8363403242055133
#
#   83.6% accuracy, because 83.6% of the data set is non-REM and it's guessing
#   non-REM for every single input, so it's guessing 83.6% of them right and 
#   all of the rest wrong.
#
# Confusion Matrix:
#   0    45661    16
#   1     8921     0
#            1     0
#
# This is fantastically useless: it's predicting 0 (false) on every single item
# Unless something is fixable here, solving this problem is probably impossible 
# with logistic regression and it's best to move on
# TODO: Someone better with the theory can probably figure out why
################################################################################

using DataFrames, CSV, ScikitLearn, PyPlot

data = CSV.File("audio_recoded_combined_binary.csv") |> DataFrame

X = convert(Array, data[!,Not(:Class)])
y = convert(Array, data[!,:Class]) # :Class is our target variable

@sk_import model_selection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state=42)

@sk_import linear_model: LogisticRegression
simplelogistic =LogisticRegression(penalty=:none)

fit!(simplelogistic,X_train,y_train)

@sk_import metrics: classification_report
# Training performance
y_pred = predict(simplelogistic,X_train)
print(classification_report(y_train,y_pred))
# Testing performance
y_pred = predict(simplelogistic,X_test)
print(classification_report(y_test,y_pred))

# Cross validation
@sk_import model_selection: cross_val_score
@time cross_val_score(LogisticRegression(penalty=:none ), X_train, y_train, cv=10)
@time cross_val_score(LogisticRegression(penalty=:none ), X_train, y_train, cv=10)

# Confusion matrix
@sk_import metrics: plot_confusion_matrix
plot_confusion_matrix(simplelogistic,X_train,y_train)
PyPlot.gcf()
plot_confusion_matrix(simplelogistic,X_test,y_test)
PyPlot.gcf()


# Neural network ###############################################################
# https://coinslab.github.io/CognitiveModelingLabWebsite/page/NeuralNetwork/
#
# Classification report (test set) 1:
#
#           precision    recall  f1-score   support
#    0           0.99      0.98      0.98     22560
#    1           0.91      0.93      0.92      4332
#
# accuracy                           0.97     26892
# macro avg      0.95      0.95      0.95     26892
# weighted avg   0.97      0.97      0.97     26892
#
# Cross Validation 1:
#   0.9662087912087912
#   0.9668498168498169
#   0.8365384615384616    -- This set failed to train completely! Why? 
#   0.9733492078029123
#   0.968220533015844
#
# Cross Validation 2:
#   0.9686813186813187
#   0.9574175824175825
#   0.9741758241758242
#   0.9707848704093781
#   0.9638245260554996
#
# Cross Validation 3 (10 k-sets this time):
#   0.9681318681318681
#   0.9686813186813187
#   0.9467032967032967
#   0.9701465201465201
#   0.9635531135531136
#   0.9622710622710623
#   0.9672161172161172
#   0.9683150183150183
#   0.9690419490749221
#   0.9736215424070342
#
# Confusion Matrix 1:
#   0    44353   1324
#   1      807   8114
#            1      0
#
# Looks great except for the bad k-set in the first cross-validation. 
# Other than that accuracy is a solid 97% and recall is from 83% to 93% so it's 
# not at the price of discarding too many of the REM inputs.
#
# Cross-validation k=10: 182.194503 seconds, 166.427203 seconds, etc
#
################################################################################

using DataFrames, CSV, ScikitLearn, PyPlot

data = CSV.File("audio_recoded_combined_binary.csv") |> DataFrame

X = convert(Array, data[!,Not(:Class)])
y = convert(Array, data[!,:Class]) # :Class is our target variable

@sk_import model_selection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

@sk_import neural_network: MLPClassifier
mlp_6layer = MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10))

fit!(mlp_6layer,X_train,y_train)

@sk_import metrics: classification_report
# Training performance
y_pred = predict(mlp_6layer,X_train)
print(classification_report(y_train,y_pred))
# Testing performance
y_pred = predict(mlp_6layer,X_test)
print(classification_report(y_test,y_pred))

# Cross validation
@sk_import model_selection: cross_val_score
cross_val_score( MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10)), X_train, y_train)
@time cross_val_score( MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10)), X_train, y_train, cv=10)
@time cross_val_score( MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10)), X_train, y_train, cv=10)

# Confusion matrix
@sk_import metrics: plot_confusion_matrix
plot_confusion_matrix(mlp_6layer,X_train,y_train)
PyPlot.gcf()

@sk_import metrics: plot_confusion_matrix
plot_confusion_matrix(mlp_6layer,X_test,y_test)
PyPlot.gcf()


# Decision Tree ################################################################
# https://coinslab.github.io/CognitiveModelingLabWebsite/page/tree/
#
# Classification report (test set):
#               precision    recall  f1-score   support
#            0       0.98      0.98      0.98     22560
#            1       0.90      0.90      0.90      4332
#     accuracy                           0.97     26892
#    macro avg       0.94      0.94      0.94     26892
# weighted avg       0.97      0.97      0.97     26892
#
# Cross-Validation:
#   0.9652014652014652
#   0.9666666666666667
#   0.967948717948718
#   0.9664835164835165
#   0.969047619047619
#   0.9659340659340659
#   0.9661172161172161
#   0.9677655677655678
#   0.9673932954753618
#   0.9629968858765342
#
# Results are great, comparable with the neural network
# However this runs MUCH faster (2 orders of magnitude vs neural network)
# k=10 cross validation runs in  1.579159 seconds
#
################################################################################

using DataFrames, CSV, ScikitLearn, PyPlot

data = CSV.File("audio_recoded_combined_binary.csv") |> DataFrame
X = convert(Array, data[!,Not(:Class)])
y = convert(Array, data[!,:Class]) # :Class is our target variable

@sk_import model_selection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

@sk_import tree: DecisionTreeClassifier
tree = DecisionTreeClassifier()
fit!(tree,X_train,y_train)

@sk_import metrics: classification_report
y_pred = predict(tree,X_train)
print(classification_report(y_train,y_pred))
y_pred = predict(tree,X_test)
print(classification_report(y_test,y_pred))

@sk_import metrics: plot_confusion_matrix
plot_confusion_matrix(tree,X_train,y_train)
PyPlot.gcf()

@sk_import metrics: plot_confusion_matrix
plot_confusion_matrix(tree,X_test,y_test)
PyPlot.gcf()

@sk_import model_selection: cross_val_score
cross_val_score( DecisionTreeClassifier(), X_train, y_train)
@time cross_val_score( DecisionTreeClassifier(), X_train, y_train, cv=10)
@time cross_val_score( DecisionTreeClassifier(), X_train, y_train, cv=10)

# Naive Bayes ##################################################################
# https://coinslab.github.io/CognitiveModelingLabWebsite/page/nbclassifier/
#
#               precision    recall  f1-score   support
#            0       0.87      1.00      0.93     22560
#            1       0.99      0.22      0.35      4332
#     accuracy                           0.87     26892
#    macro avg       0.93      0.61      0.64     26892
# weighted avg       0.89      0.87      0.84     26892
#
# Cross-Validation:
#   0.8697802197802198
#   0.8703296703296703
#   0.8725274725274725
#   0.8716117216117216
#   0.8745421245421245
#   0.8732600732600733
#   0.8710622710622711
#   0.8708791208791209
#   0.8728704891005679
#   0.8754350613665507
#
# This doesn't work well. It's just guessing "no" on almost all the inputs and 
# getting absolutely abysmal recall (22%). 
# TODO: Find out why.
# On the upside, it runs fast.  k=10 cross validation runs in 0.104042 seconds
#
################################################################################

using DataFrames, CSV, ScikitLearn, PyPlot

data = CSV.File("audio_recoded_combined_binary.csv") |> DataFrame
X = convert(Array, data[!,Not(:Class)])
y = convert(Array, data[!,:Class]) # :Class is our target variable

@sk_import model_selection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

@sk_import naive_bayes: GaussianNB
gnb = GaussianNB()
fit!(gnb,X_train,y_train)

@sk_import metrics: classification_report

y_pred = predict(gnb,X_train)
print(classification_report(y_train,y_pred))
y_pred = predict(gnb,X_test)
print(classification_report(y_test,y_pred))

@sk_import metrics: plot_confusion_matrix
plot_confusion_matrix(gnb,X_train,y_train)
PyPlot.gcf()
plot_confusion_matrix(gnb,X_test,y_test)
PyPlot.gcf()

@time cross_val_score( GaussianNB(), X_train, y_train, cv=10)
@time cross_val_score( GaussianNB(), X_train, y_train, cv=10)

# Random Forest ################################################################
# https://coinslab.github.io/CognitiveModelingLabWebsite/page/nbclassifier/
#
# Classification report (test set):
#               precision    recall  f1-score   support
#            0       0.98      0.99      0.99     22560
#            1       0.96      0.92      0.94      4332
#     accuracy                           0.98     26892
#    macro avg       0.97      0.96      0.96     26892
# weighted avg       0.98      0.98      0.98     26892
#
# Cross-Validation:
#   0.9782051282051282
#   0.9782051282051282
#   0.9793040293040293
#   0.9774725274725274
#   0.978021978021978
#   0.9798534798534798
#   0.9787545787545787
#   0.9805860805860805
#   0.9803993405385601
#   0.9778347682725774
#
# This is the best so far.
# Not as fast as Naive Bayes or Tree but 3 times as fast as the neural net.
# k=10 cross validation runs in 54.345652 seconds
#
################################################################################

using DataFrames, CSV, ScikitLearn, PyPlot

data = CSV.File("audio_recoded_combined_binary_13f.csv") |> DataFrame
X = convert(Array, data[!,Not(:Class)])
y = convert(Array, data[!,:Class]) # :Class is our target variable

@sk_import model_selection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

@sk_import ensemble: RandomForestClassifier
rfc = RandomForestClassifier()
fit!(rfc,X_train,y_train)

@sk_import metrics: classification_report
y_pred = predict(rfc,X_train)
print(classification_report(y_train,y_pred))
y_pred = predict(rfc,X_test)
print(classification_report(y_test,y_pred))

@sk_import metrics: plot_confusion_matrix
plot_confusion_matrix(rfc,X_train,y_train)
PyPlot.gcf()
plot_confusion_matrix(rfc,X_test,y_test)
PyPlot.gcf()

@sk_import model_selection: cross_val_score
@time cross_val_score( RandomForestClassifier(), X_train, y_train, cv=10)
@time cross_val_score( RandomForestClassifier(), X_train, y_train, cv=10)


