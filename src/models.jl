# Load in models and wrap them to hide @load

const lbc = MLJ.@load LinearBinaryClassifier pkg=GLM
const etc = MLJ.@load EvoTreeClassifier pkg=EvoTrees
const rf = MLJ.@load RandomForestClassifier pkg=DecisionTree

linear_model(; kw...) = lbc(; kw...)
boosted_regression_tree(; kw...) = etc(; kw...)
random_forest(; kw...) = rf(; kw...)
