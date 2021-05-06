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

mlp_1layer10  = MLPClassifier(hidden_layer_sizes=(10))
mlp_1layer25  = MLPClassifier(hidden_layer_sizes=(25))
mlp_1layer40  = MLPClassifier(hidden_layer_sizes=(40))
mlp_1layer50  = MLPClassifier(hidden_layer_sizes=(50))
mlp_1layer100 = MLPClassifier(hidden_layer_sizes=(100))

mlp_6layer    = MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10))

@sk_import metrics: classification_report
@sk_import metrics: plot_confusion_matrix

function testModel(model)

    fit!(model,X_train,y_train)
    # Training performance
    y_pred = predict(model,X_train)
    print(classification_report(y_train,y_pred))
    # Testing performance
    y_pred = predict(model,X_test)
    print(classification_report(y_test,y_pred))
    # Confusion matrices
    plot_confusion_matrix(model,X_train,y_train)
    PyPlot.gcf()
    plot_confusion_matrix(model,X_test,y_test)
    PyPlot.gcf()

end



testModel(mlp_1layer40)


# Cross validation
################################################################################

@sk_import model_selection: cross_val_score

function crossValidateModel(model)

    fit!(model,X_train,y_train)
    @time cva = cross_val_score(model, X_train, y_train, cv=10)
    print(mean(cva))
    @time cvp = cross_val_score(model, X_train, y_train, cv=10, scoring="precision")
    print(mean(cvp))
    @time cvr = cross_val_score(model, X_train, y_train, cv=10, scoring="recall")
    print(mean(cvr))

end

#=
@sk_import model_selection: cross_val_score
cross_val_score( MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10)), X_train, y_train)
@time cross_val_score( MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10)), X_train, y_train, cv=10)
@time cross_val_score( MLPClassifier(hidden_layer_sizes=(30, 50, 60, 10, 10, 10)), X_train, y_train, cv=10)

@sk_import model_selection: cross_val_score
@time cva = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=10)
mean(cva)
@time cvp = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=10, scoring="precision")
mean(cvp)
@time cvr = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=10, scoring="recall")
mean(cvr)
=#


# Confusion matrix
################################################################################


@sk_import metrics: plot_confusion_matrix



plot_confusion_matrix(mlp_6layer,X_train,y_train)
PyPlot.gcf()

@sk_import metrics: plot_confusion_matrix
plot_confusion_matrix(mlp_6layer,X_test,y_test)
PyPlot.gcf()
