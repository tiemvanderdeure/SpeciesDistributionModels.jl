# Load in models and wrap them to hide @load

const lbc = MLJ.@load LinearBinaryClassifier pkg=GLM verbosity = 0
const etc = MLJ.@load EvoTreeClassifier pkg=EvoTrees verbosity = 0
const rf = MLJ.@load RandomForestClassifier pkg=DecisionTree verbosity = 0

linear_model(; kw...) = lbc(; kw...)
boosted_regression_tree(; kw...) = etc(; kw...)
random_forest(; kw...) = rf(; kw...)
