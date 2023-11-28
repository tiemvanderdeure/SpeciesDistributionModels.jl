import SpeciesDistributionModels as SDM
using SpeciesDistributionModels, MLJ

function classification_rates(scores, y)
    fpr, tpr, thresholds = roc_curve(scores, y)
    tnr = 1. .- fpr
    fnr = 1. .- tpr

    thresholds = [1.; thresholds]

    return (; fpr, tpr, tnr, fnr, thresholds)
end

struct ROC end
struct Sens end
struct Spec end
struct TSS end

xdata(::Any, rates) = rates.thresholds

xdata(::ROC, rates) = rates.fpr
ydata(::ROC, rates) = rates.tpr
ydata(::Sens, rates) = rates.tpr
ydata(::Spec, rates) = rates.tnr
ydata(::TSS, rates) = (rates.tpr .+ rates.tnr .- 1)


function interactive_evaluation(ensemble)
    idx_by_model = NamedTuple{keys(ensemble.models)}(
        [findall(getindex.(ensemble.trained_models, :model_key) .== key) for key in keys(ensemble.models)]
    )

    n_models = length(idx_by_model)

    test_predictions = predict(ensemble, :test)
    test_preds_by_model = map(idx_by_model) do idx
        y_hat = vcat([test_predictions[id].y_hat for id in idx]...)
        y = vcat([test_predictions[id].y for id in idx]...)
        return(; y_hat, y)
    end

    all_rates = map(pred -> classification_rates(pred.y_hat, pred.y), test_predictions)
    rates_by_model = map(pred -> classification_rates(pred.y_hat, pred.y), test_preds_by_model)

    # Figure itself
    fig = Figure()

    controls = fig[1,2] = GridLayout()
    plots = fig[1,1] = GridLayout()
    ax = Axis(plots[1,1]; limits = (0, 1.01, 0, 1.01))
    hidespines!(ax, :t, :r)

    funcs = [Sens(),Spec(), ROC(), TSS()]
    dropdown = Menu(controls[1,1],
        options = zip(["Sensitivity", "Specificity", "ROC", "TSS"], funcs),
        default = "ROC", tellheight = false)

    all_models_toggle = Toggle(fig, active = false)
    controls[2, 1] = hgrid!(all_models_toggle, Label(fig, "Show individuals models"))

    toggles = [Toggle(fig, active = true) for i in 1:length(idx_by_model)]
    labels = [Label(fig, String(key)) for key in keys(idx_by_model)]

    Label(controls[3, 1], "Select models"; font = :bold)
    controls[4, 1] = grid!(hcat(toggles, labels), tellheight = false)

    # Data to plot
    plot_data_avg = map(eachindex(rates_by_model)) do idx
        x = lift(s -> xdata(s, rates_by_model[idx]), dropdown.selection)
        y = lift(s -> ydata(s, rates_by_model[idx]), dropdown.selection)
        return(x, y)
    end

    plot_data_all = map(eachindex(all_rates)) do idx
        x = lift(f -> xdata(f, all_rates[idx]), dropdown.selection)
        y = lift(f -> ydata(f, all_rates[idx]), dropdown.selection)
        return(x, y)
    end

    ls_average = [lines!(ax, d[1], d[2], color = i, colormap = :turbo, colorrange = (1, n_models), linewidth = 3) for (i, d) in enumerate(plot_data_avg)]
    ls_members = [
        [
            lines!(ax, plot_data_all[j][1], plot_data_all[j][2];
            color = i, colormap = :turbo, colorrange = (1, n_models), linewidth = 1) for j in idxs
        ] for (i, idxs) in enumerate(idx_by_model)
    ]

    map(ls_average, ls_members, toggles) do line, lines, toggle
        connect!(line.visible, toggle.active)
        lines_active = @lift $(toggle.active) & $(all_models_toggle.active)

        for l in lines
            connect!(l.visible, lines_active)
        end
    end

    return fig #should return FigureAxisPlot somehow?
end
