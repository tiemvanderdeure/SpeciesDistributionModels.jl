function classification_rates(scores, y)
    fpr, tpr, thresholds = StatisticalMeasures.roc_curve(scores, y)
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
    Makie.hidespines!(ax, :t, :r)

    funcs = [Sens(),Spec(), ROC(), TSS()]
    dropdown = Makie.Menu(controls[1,1],
        options = zip(["Sensitivity", "Specificity", "ROC", "TSS"], funcs),
        default = "ROC", tellheight = false)

    all_models_toggle = Makie.Toggle(fig, active = false)
    controls[2, 1] = Makie.hgrid!(all_models_toggle, Label(fig, "Show individuals models"))

    toggles = [Makie.Toggle(fig, active = true) for i in 1:length(idx_by_model)]
    labels = [Makie.Label(fig, String(key)) for key in keys(idx_by_model)]

    Makie.Label(controls[3, 1], "Select models"; font = :bold)
    controls[4, 1] = Makie.grid!(hcat(toggles, labels), tellheight = false)

    # Data to plot
    plot_data_avg = map(eachindex(rates_by_model)) do idx
        x = Makie.lift(s -> xdata(s, rates_by_model[idx]), dropdown.selection)
        y = Makie.lift(s -> ydata(s, rates_by_model[idx]), dropdown.selection)
        return(x, y)
    end

    plot_data_all = map(eachindex(all_rates)) do idx
        x = Makie.lift(f -> xdata(f, all_rates[idx]), dropdown.selection)
        y = Makie.lift(f -> ydata(f, all_rates[idx]), dropdown.selection)
        return(x, y)
    end

    ls_average = [Makie.lines!(ax, d[1], d[2], color = i, colormap = :turbo, colorrange = (1, n_models), linewidth = 3) for (i, d) in enumerate(plot_data_avg)]
    ls_members = [
        [
            Makie.lines!(ax, plot_data_all[j][1], plot_data_all[j][2];
            color = i, colormap = :turbo, colorrange = (1, n_models), linewidth = 1) for j in idxs
        ] for (i, idxs) in enumerate(idx_by_model)
    ]

    map(ls_average, ls_members, toggles) do line, lines, toggle
        Makie.connect!(line.visible, toggle.active)
        lines_active = lift((t, t2) -> t & t2, (toggle.active), (all_models_toggle.active))

        for l in lines
            Makie.connect!(l.visible, lines_active)
        end
    end

    return fig #should return FigureAxisPlot somehow?
end

# Plot output from shapley
function interactive_response_curves(shapley::SDMshapley)
    f = Figure()
    ax = Axis(f[1,1], ylabel = "shapley value", xlabel = "variable value")

    preds = shapley.ensemble.predictors

    var_menu = Makie.Menu(f[1, 2], options = zip(String.(preds), preds), tellheight = false, tellwidth = true, width = 100)
    indices = Makie.Observable(1:length(shapley))

    var_indices = lift((x,y) -> (x,y), var_menu.selection, indices)

    function update(var, indices)
        ys[] = collect(Base.Iterators.flatten(shap[var] for shap in shapley.values[indices]))
        xs[] = repeat(shapley.ensemble.data.predictor[var], size(indices, 1))

        us[] = range(extrema(xs[])...; length = 100)

        smooth_model = Loess.loess(xs[], ys[]; span = Statistics.std(xs[])/2, degree = 2)
        res_model = Loess.loess(xs[], abs.(Loess.residuals(smooth_model)); span = Statistics.std(xs[])/2, degree = 2)

        smooth_ys[] = Loess.predict(smooth_model, us[])
        res_ys = Loess.predict(res_model, us[])
        lower_ys[] = smooth_ys[] .- res_ys
        upper_ys[] = smooth_ys[] .+ res_ys

        Makie.reset_limits!(ax)
    end

    ys, xs, us, smooth_ys, lower_ys, upper_ys = (Makie.Observable{Vector{Float64}}([]) for i in 1:6)

    update(var_menu.selection[], indices[])

    scatter!(xs, ys, markersize = 3)
    lines!(ax, us, smooth_ys, linewidth = 7, color = :black, alpha = 1)
    Makie.band!(ax, us, lower_ys, upper_ys, alpha = 0.5)

    Makie.on(x -> update(x...), var_indices)

    f
end
