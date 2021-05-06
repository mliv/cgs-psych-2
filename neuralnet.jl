#= Neural Network Variations
Mainly tuning the Multi-Layer Perceptron
A big limitation is that this doesn't use GPU at all 
since ScikitLearn doesn't support it
So deeper models will take forever even on a good CPU
And I don't even want to try the 13-feature data set
-Matei 
=#

using DataFrames, CSV, ScikitLearn, PyPlot, Statistics

# Data Loading
################################################################################

data = CSV.File("audio_recoded_combined_binary.csv") |> DataFrame

X = convert(Array, data[!,Not(:Class)])
y = convert(Array, data[!,:Class]) # :Class is our target variable

@sk_import model_selection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

@sk_import neural_network: MLPClassifier

#=
mlp_1layer10  = MLPClassifier(hidden_layer_sizes=(10))
mlp_1layer25  = MLPClassifier(hidden_layer_sizes=(25))
mlp_1layer40  = MLPClassifier(hidden_layer_sizes=(40))
mlp_1layer50  = MLPClassifier(hidden_layer_sizes=(50))
mlp_1layer100 = MLPClassifier(hidden_layer_sizes=(100))

mlp_6layer    = MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10))
mlp_5layer    = MLPClassifier(hidden_layer_sizes=(50, 10, 10, 10, 10, 10))
=#

@sk_import metrics: classification_report
@sk_import metrics: plot_confusion_matrix
@sk_import model_selection: cross_val_score

function trainModel(model)
    fit!(model,X_train,y_train)
    # Training performance
    y_pred = predict(model,X_train)
    print(classification_report(y_train,y_pred))
    # Confusion matrices
    plot_confusion_matrix(model,X_test,y_test)
    PyPlot.gcf()
end

function crossValidateModel(model,folds=2)
    #fit!(model,X_train,y_train)
    @time cva = cross_val_score(model, X_train, y_train, cv=folds)
    println("mean accuracy: \n" *  string(mean(cva)))
    @time cvp = cross_val_score(model, X_train, y_train, cv=folds, scoring="precision")
    println("mean precision: \n" * string(mean(cvp)))
    @time cvr = cross_val_score(model, X_train, y_train, cv=folds, scoring="recall")
    println("mean recall: \n" *    string(mean(cvr)))
end

function testModel(model)
    # Testing performance
    y_pred = predict(model,X_test)
    print(classification_report(y_test,y_pred))
    # Confusion matrices
    plot_confusion_matrix(model,X_test,y_test)
    PyPlot.gcf()
end

mlp_2layer    = MLPClassifier(hidden_layer_sizes=(50, 10))
trainModel(         mlp_2layer)
crossValidateModel( mlp_2layer)
testModel(          mlp_2layer)

mlp_2layerlarge    = MLPClassifier(hidden_layer_sizes=(100, 50))
trainModel(         mlp_2layerlarge)
crossValidateModel( mlp_2layerlarge)
testModel(          mlp_2layerlarge)

mlp_3layer    = MLPClassifier(hidden_layer_sizes=(50, 10, 10))
trainModel(         mlp_3layer)
crossValidateModel( mlp_3layer)
testModel(          mlp_3layer)

mlp_4layer    = MLPClassifier(hidden_layer_sizes=(50, 10, 10, 10))
trainModel(         mlp_4layer)
crossValidateModel( mlp_4layer)
testModel(          mlp_4layer)

mlp_5layer    = MLPClassifier(hidden_layer_sizes=(50, 10, 10, 10, 10, 10))
trainModel(         mlp_5layer)
crossValidateModel( mlp_5layer)
testModel(          mlp_5layer)




testModel(mlp_1layer50)
@sk_import model_selection: cross_val_score
@time cva = cross_val_score(mlp_1layer50, X_train, y_train, cv=10)
println("mean accuracy:" * mean(cva))
mlp_1layer50_p  = MLPClassifier(hidden_layer_sizes=(50))
fit!(model,X_train,y_train)
@time cvp = cross_val_score(mlp_1layer50_p, X_train, y_train, cv=10, scoring="precision")
println("mean precision:" * mean(cvp))
@time cvr = cross_val_score(model, X_train, y_train, cv=10, scoring="recall")
println("mean recall:" * mean(cvr))

# Cross validation
################################################################################
#=
@sk_import model_selection: cross_val_score

function crossValidateModel(model)

    #fit!(model,X_train,y_train)
    @time cva = cross_val_score(model, X_train, y_train, cv=10)
    println("mean accuracy:" * mean(cva))
    @time cvp = cross_val_score(model, X_train, y_train, cv=10, scoring="precision")
    println("mean precision:" * mean(cvp))
    @time cvr = cross_val_score(model, X_train, y_train, cv=10, scoring="recall")
    println("mean recall:" * mean(cvr))

end




@time crossValidateModel(mlp_1layer50)
=#
# Validation and learning curves
# Will get this to work eventually -Matei
################################################################################

#=
@sk_import model_selection: validation_curve
@sk_import model_selection: learning_curve

using Plots, StatsPlots


validation_curve(mlp_1layer50, X, y, 
    param_name="gamma", param_range=np.logspace(-6, -1, 5), scoring="accuracy")

fit!(mlp_1layer50,X_train,y_train)
train_sizes, train_scores, valid_scores = learning_curve(
    mlp_1layer50, X, y, train_sizes=[100, 500, 1000], cv=2)

plot(train_scores, valid_scores)
=#

# Random Forest Comparison
################################################################################


rfc    = RandomForestClassifier()
trainModel(         mlp_2layer)
crossValidateModel( mlp_2layer)
testModel(          mlp_2layer)