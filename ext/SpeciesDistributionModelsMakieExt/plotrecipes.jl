function _model_controls!(fig, ensemble)
    toggles = [Makie.Toggle(fig, active = true) for i in 1:Base.length(ensemble)]
    labels = [Makie.Label(fig, String(key)) for key in SDM.model_keys(ensemble)]
    g = Makie.grid!(hcat(toggles, labels))
    return g, toggles
end


function Makie.boxplot(ev::SDMensembleEvaluation, measure::Symbol)
    modelnames = collect(Base.string.(model_keys(ev.ensemble)))
    f = Makie.Figure()
    
    for (i, t) in enumerate((:train, :test))
        ax = Makie.Axis(
            f[1,i]; 
            limits = (nothing, nothing, 0, 1),
            xticks = (1:Base.length(ev), modelnames),
            xticklabelrotation = -pi/4,
            title = Base.string(t)
        )
        y = machine_evaluations(ev)[t][measure]
        x = mapreduce(vcat, enumerate(ev)) do (i, e)
            fill(i, Base.length(e))
        end
    
        Makie.boxplot!(ax, x, y)
    end
    return f
end

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

xdata(::ROC, conf_mats) = 1 .- SDM.selectivity.(conf_mats)
ydata(::ROC, conf_mats) = SDM.sensitivity.(conf_mats)
ydata(::Sens, conf_mats) = SDM.sensitivity.(conf_mats)
ydata(::Spec, conf_mats) = SDM.selectivity.(conf_mats)
ydata(::TSS, conf_mats) = SDM.kappa.(conf_mats)


function SDM.interactive_evaluation(ensemble; thresholds = 0:0.01:1)
    xdata(::Any, conf_mats) = thresholds

    idx_by_model = map(enumerate(ensemble)) do (i, e)
        fill(i, Base.length(e))
    end |> NamedTuple{Tuple(model_keys(ensemble))}

    n_models = length(idx_by_model)

    conf_mats = mapreduce(hcat, ensemble) do gr
        map(gr) do sdm_machine
            rows = SDM.test_rows(sdm_machine)
            y_hat = SDM.MLJBase.predict(sdm_machine.machine; rows)
            y = data(sdm_machine).response[rows]
            _conf_mats_from_thresholds(SDM.MLJBase.pdf.(y_hat, true), y, thresholds)
        end
    end

    # Figure itself
    fig = Figure()

    controls = fig[1,2] = GridLayout()
    plots = fig[1,1] = GridLayout()
    ax = Axis(plots[1,1]; limits = (0, 1.01, 0, 1.01))
    Makie.hidespines!(ax, :t, :r)

    funcs = [Sens(),Spec(), ROC(), TSS()]
    dropdown = Makie.Menu(controls[1,1],
        options = zip(["Sensitivity", "Selectivity", "ROC", "TSS"], funcs),
        default = "ROC", tellheight = false)

    all_models_toggle = Makie.Toggle(fig, active = false)
    controls[2, 1] = Makie.hgrid!(all_models_toggle, Label(fig, "Show individuals models"))

    Makie.Label(controls[3, 1], "Select models"; font = :bold)
    controls[4,1], toggles = _model_controls!(fig, ensemble)

    # Data to plot
    plot_data_all = map(conf_mats) do conf_mat
        x = Makie.lift(f -> xdata(f, conf_mat), dropdown.selection)
        y = Makie.lift(f -> ydata(f, conf_mat), dropdown.selection)
        return(x, y)
    end

    plot_data_avg = map(eachcol(plot_data_all)) do data
        x = map(getindex.(data, 1)...) do d...
            Statistics.mean(d)
        end
        y = map(getindex.(data, 2)...) do d...
            Statistics.mean(d)
        end
        return (x, y)
    end

    ls_average = [Makie.lines!(ax, d[1], d[2], color = i, colormap = :turbo, colorrange = (1, n_models), linewidth = 3) for (i, d) in enumerate(plot_data_avg)]
    ls_members = [
        Makie.lines!(ax, plot_data_all[r, c][1], plot_data_all[r, c][2];
            color = c, colormap = :turbo, colorrange = (1, n_models), linewidth = 1)
        for r in 1:size(plot_data_all, 1), c in 1:size(plot_data_all, 2)
    ]

    map(ls_average, eachcol(ls_members), toggles) do line, lines, toggle
        Makie.connect!(line.visible, toggle.active)
        lines_active = lift((t, t2) -> t & t2, (toggle.active), (all_models_toggle.active))

        for l in lines
            Makie.connect!(l.visible, lines_active)
        end
    end

    return fig 
end

# Plot output from shapley
function SDM.interactive_response_curves(expl::SDMensembleExplanation)
    fig = Figure()
    controls = fig[1,2] = GridLayout()
    ax = Axis(fig[1,1], ylabel = "Shapley value", xlabel = "Variable value")

    preds = Base.keys(SDM.data(expl))
    machine_group_indices = mapreduce(vcat, enumerate(expl)) do (i, g)
        fill(i, length(g))
    end

    var_menu = Makie.Menu(controls[1, 1], options = zip(String.(preds), preds), tellheight = false, valign = :center)
    var = var_menu.selection
    Makie.Label(controls[2, 1], "Select models"; font = :bold)
    controls[3,1], toggles = _model_controls!(fig, expl.ensemble)
    any_toggle = lift((t...) -> any(t), [t.active for t in toggles]...)
    Makie.Label(controls[4,1], "Smoothness"; font = :bold)
    span_slider = Makie.Slider(controls[5,1], range = 0:0.01:1, startvalue = 0.5)

    # data
    ys = mapreduce(vcat, expl) do gr_expl
        map(SDM.machine_explanations(gr_expl)) do me
            @lift me.values[$var]
        end
    end

    xs = @lift SDM.data(expl)[$var]
    us = Makie.lift(xs -> range(extrema(xs)...; length = 100), xs)

    scatters = map(ys) do ys
        scatter!(xs, ys, markersize = 3, color = :lightblue) 
    end

    map(machine_group_indices, scatters) do m_idx, scatter
        Makie.connect!(scatter.visible, toggles[m_idx].active)
    end

    function update()
        indices = map(m_idx -> toggles[m_idx].active[], machine_group_indices)
        if any(indices)
            newxs = repeat(xs[], Base.sum(indices))
            newys = mapreduce(vcat, SDM.machine_explanations(expl)[indices]) do me
                    me.values[var[]]
                end

            span = span_slider.value[]
            smooth_model = Loess.loess(newxs, newys; span, degree = 2)
            res_model = Loess.loess(newxs, abs.(Loess.residuals(smooth_model)); span, degree = 2)

            smooth_ys[] = Loess.predict(smooth_model, us[])
            res_ys = Loess.predict(res_model, us[])
            lower_ys[] = smooth_ys[] .- res_ys
            upper_ys[] = smooth_ys[] .+ res_ys

            Makie.reset_limits!(ax)
        end
    end

    smooth_ys, lower_ys, upper_ys = (Makie.Observable{Vector{Float64}}([]) for i in 1:6)

    update()

    loess_l = Makie.lines!(ax, us, smooth_ys, linewidth = 7, color = :black, alpha = 1)
    loess_b = Makie.band!(ax, us, lower_ys, upper_ys, alpha = 0.5)
    Makie.connect!(loess_l.visible, any_toggle)
    Makie.connect!(loess_b.visible, any_toggle)


    Makie.on(x -> update(), var_menu.selection)
    Makie.on(x -> update(), span_slider.value)
    [Makie.on(x -> update(), t.active) for t in toggles]

    fig
end
