# Load in models and wrap them to hide the interfaces
# Do we want to get ride of these as dependencies?

using MLJGLMInterface: LinearBinaryClassifier
using EvoTrees: EvoTreeClassifier
using MLJDecisionTreeInterface: RandomForestClassifier

linear_model = LinearBinaryClassifier
boosted_regression_tree = EvoTreeClassifier
random_forest = RandomForestClassifier
