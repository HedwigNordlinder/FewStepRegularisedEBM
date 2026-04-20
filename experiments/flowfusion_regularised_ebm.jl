# Flowfusion-flavoured version of `fewstep_regularised_ebm.jl`: same raw-vs-
# regularised comparison, but now the EBM is trained with `floss` on a
# Flowfusion `BrownianMotion` bridge (Option B — X̂₁-predicting EBM via
# Tweedie-style parameterisation) and sampled with `gen`. The teacher
# dataset (−log p̂ from the few-step map) and the auxiliary loss are
# identical to the DSM version.

include(joinpath(@__DIR__, "fewstep_map.jl"))

using ForwardBackward, Flowfusion, CannotWaitForTheseOptimisers
using Jester: grad_fd

# GPU-friendly override of the upstream `endpoint_conditioned_sample` —
# upstream uses CPU `randn(eltype(μ), size(μ))`, breaking CuArray bridges.
function ForwardBackward.endpoint_conditioned_sample(Xa::ContinuousState, Xc::ContinuousState,
                                                     p::BrownianMotion, t_a, t_b, t_c)
    xa, xc = Xa.state, Xc.state
    d  = ndims(xa)
    tF   = ForwardBackward.expand(t_b .- t_a, d)
    tB   = ForwardBackward.expand(t_c .- t_b, d)
    Ttot = ForwardBackward.expand(t_c .- t_a, d)
    w1 = (Ttot .- tF) ./ Ttot
    w2 = tF ./ Ttot
    μ  = xa .* w1 .+ xc .* w2
    σ2 = p.v .* tF .* tB ./ Ttot
    σ  = sqrt.(σ2)
    return ContinuousState(μ .+ σ .* randn!(similar(μ)))
end

const BM_SIGMA = T(0.15)
const FD_EPS   = T(1e-3)

# =====================================================================
#  Flowfusion-style EBM  (scalar energy head + Tweedie-ish X̂₁ wrapper)
# =====================================================================
struct EBM{A}
    layers::A
end
Flux.@layer EBM

function EBM(; embeddim = 256, spacedim = SPACEDIM, nlayers = 3)
    embed_t     = Chain(RandomFourierFeatures(1 => embeddim, 1f0),
                        Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(spacedim => embeddim, 1f0),
                        Dense(embeddim => embeddim, swish))
    project_in  = spacedim + 1 + 2 * embeddim
    project     = Dense(project_in => embeddim, swish)
    # Tuple — Jester.grad_fd's rrule recurses through Tuple/NamedTuple tangents.
    ffs         = Tuple(Dense(embeddim => embeddim, swish) for _ in 1:nlayers)
    decode      = Dense(embeddim => 1)
    EBM((; embed_t, embed_state, project, ffs, decode))
end

function (m::EBM)(x, t)
    l  = m.layers
    tv = reshape(t, 1, :)
    et = l.embed_t(tv)
    ex = l.embed_state(x)
    h  = l.project(vcat(x, tv, et, ex))
    for ff in l.ffs
        h = h .+ ff(h)
    end
    vec(l.decode(h))
end

struct EBMFlowWrapper{M}
    ebm::M
end
Flux.@layer EBMFlowWrapper

function (w::EBMFlowWrapper)(t, Xt)
    x     = tensor(Xt)
    batch = size(x, 2)
    tv = if t isa AbstractArray
        reshape(t, :)
    else
        buf = similar(x, batch); fill!(buf, T(t)); buf
    end
    f(x_, m_) = sum(m_(x_, tv))
    score = -grad_fd(f, x, w.ebm; ε = FD_EPS)
    scale = reshape(T(1.05) .- tv, 1, :)
    return ContinuousState(x .+ score .* scale)
end

# =====================================================================
#  Losses
# =====================================================================
# Auxiliary: offset-invariant MSE between the EBM's energy head at t=0
# and the teacher's −log p̂. Same as in fewstep_regularised_ebm.jl.
function aux_loss(ebm::EBM, x1_aux, t_aux, neg_logp_teacher)
    E   = ebm(x1_aux, t_aux)
    E_c = E                .- mean(E)
    g_c = neg_logp_teacher .- mean(neg_logp_teacher)
    mean((E_c .- g_c) .^ 2)
end

# =====================================================================
#  Trainer — raw or regularised Flowfusion EBM
# =====================================================================
"""
    train_ebm!(model; ...)

Raw training if `teacher_x1 === nothing`: pure `floss` on the BrownianMotion
bridge. Otherwise adds the energy-vs-teacher-logp auxiliary loss on a random
teacher minibatch once every `aux_cadence` iters, after `warmup_iters` of
warmup.
"""
function train_ebm!(model::EBMFlowWrapper;
                    iters = 20000, batch = 4096, eta = 1f-3,
                    teacher_x1 = nothing, teacher_neg_logp = nothing,
                    aux_weight = T(0.1),
                    aux_cadence = 5,
                    aux_batch = 1024,
                    warmup_iters = 50,
                    cooldown_frac = 0.2,
                    cooldown_final_factor = 0.01,
                    seed = 0,
                    desc = nothing)
    Random.seed!(seed)
    regularised = teacher_x1 !== nothing
    n_teacher   = regularised ? size(teacher_x1, 2) : 0
    if regularised
        @assert size(teacher_x1, 2) == length(teacher_neg_logp)
        @assert aux_batch <= n_teacher
    end
    P      = BrownianMotion(BM_SIGMA)
    η_type = typeof(float(eta))
    η_base = η_type(eta)
    η      = η_base
    opt    = Flux.setup(Muon(eta = η), model)
    tot_hist = Float32[]; fm_hist = Float32[]; aux_hist = Float32[]
    eta_hist = Float32[]; aux_active_hist = Bool[]
    cooldown_start  = iters - floor(Int, cooldown_frac * iters)
    cooldown_length = max(iters - cooldown_start, 1)
    final_factor    = η_type(cooldown_final_factor)
    prog_desc = desc === nothing ?
                (regularised ? "training reg FF-EBM" : "training raw FF-EBM") : desc
    prog = Progress(iters; desc = prog_desc, dt = 0.5, showspeed = true)
    for iter in 1:iters
        X0 = ContinuousState(sampleX0(batch))
        X1 = ContinuousState(sampleX1(batch))
        t  = rand(T, batch) |> DEV
        Xt = bridge(P, X0, X1, t)
        use_aux = regularised && iter > warmup_iters && (iter % aux_cadence == 0)
        if use_aux
            idx     = rand(1:n_teacher, aux_batch)
            x1_aux  = teacher_x1[:, idx]
            nlp_aux = teacher_neg_logp[idx]
            t0_aux  = fill!(similar(x1_aux, aux_batch), T(0))
        end
        val, g = Flux.withgradient(model) do m
            fm = floss(P, m(t, Xt), X1, scalefloss(P, t))
            if use_aux
                a     = aux_loss(m.ebm, x1_aux, t0_aux, nlp_aux)
                total = fm + aux_weight * a
                (total, fm, a)
            else
                (fm, fm, zero(fm))
            end
        end
        total_val, fm_val, aux_val = val
        Flux.update!(opt, model, g[1])
        push!(tot_hist, Float32(total_val))
        push!(fm_hist,  Float32(fm_val))
        push!(aux_hist, Float32(aux_val))
        push!(aux_active_hist, use_aux)
        if iter > cooldown_start
            progress = η_type((iter - cooldown_start) / cooldown_length)
            η = η_base * (one(η_type) - progress * (one(η_type) - final_factor))
            Optimisers.adjust!(opt, η)
        end
        push!(eta_hist, Float32(η))
        next!(prog; showvalues = [(:iter, iter),
                                  (:flow,  round(fm_val;  digits = 4)),
                                  (:aux,   round(aux_val; digits = 4)),
                                  (:tot,   round(total_val; digits = 4)),
                                  (:eta,   Float32(η))])
    end
    return (; tot_hist, fm_hist, aux_hist, eta_hist, aux_active_hist,
            regularised, aux_weight, aux_cadence, warmup_iters)
end

# =====================================================================
#  Teacher dataset (same as fewstep_regularised_ebm.jl)
# =====================================================================
function generate_teacher_dataset(fewstep_model_cpu, n; nsteps = 4, ε = T(1e-3))
    @info "generating teacher dataset" n nsteps
    x0 = sampleX0_cpu(n)
    x1, logp = sample_fewstep_with_logpdf(fewstep_model_cpu, x0; nsteps = nsteps, ε = ε)
    keep = isfinite.(logp)
    return x1[:, keep], T.(-logp[keep])
end

# =====================================================================
#  R² helpers
# =====================================================================
r_squared(true_y::AbstractVector{<:Real}, pred_y::AbstractVector{<:Real}) = begin
    ss_res = sum((true_y .- pred_y) .^ 2)
    ss_tot = sum((true_y .- mean(true_y)) .^ 2)
    1 - ss_res / ss_tot
end

tail_cuts(true_y, pred_y) = begin
    resid = abs.(true_y .- pred_y)
    q99   = quantile(resid, 0.99)
    q90   = quantile(resid, 0.90)
    (resid .<= q99, resid .<= q90)
end

# =====================================================================
#  Main
# =====================================================================
function main(; iters_fewstep_per_epoch = 4000,
              fewstep_last_epoch_multiplier = 5,
              iters_ebm      = 20000,
              batch          = 4096,
              n_teacher      = 50000,
              n_eval         = 2000,
              n_samples_plot = 5000,
              gen_steps      = T.(0:0.005:1),
              aux_weight     = T(1.0),
              aux_cadence    = 5,
              aux_batch      = 1024,
              warmup_iters   = 50,
              outdir         = @__DIR__,
              fewstep_path   = joinpath(@__DIR__, "fewstep_map_model.bin"),
              ebm_raw_path   = joinpath(@__DIR__, "ff_reg_ebm_raw_model.bin"),
              ebm_reg_path   = joinpath(@__DIR__, "ff_reg_ebm_reg_model.bin"))
    # ─── 1. fewstep teacher ───
    local fewstep_model
    if isfile(fewstep_path)
        @info "loading saved fewstep model (delete to retrain)" fewstep_path
        loaded = deserialize(fewstep_path)
        fewstep_model = loaded.model |> DEV
    else
        @info "training fewstep model from scratch" iters_fewstep_per_epoch fewstep_last_epoch_multiplier
        fewstep_model = TwoTimeMap(embeddim = 256, nlayers = 3) |> DEV
        hist_fs = train!(fewstep_model;
                         iters_per_epoch = iters_fewstep_per_epoch,
                         last_epoch_multiplier = fewstep_last_epoch_multiplier)
        serialize(fewstep_path, (model = fewstep_model |> Flux.cpu, hist = hist_fs))
    end
    fewstep_cpu = fewstep_model |> Flux.cpu

    # ─── 2. teacher dataset ───
    x1_teach_cpu, neg_logp_teach_cpu = generate_teacher_dataset(fewstep_cpu, n_teacher)
    @info "teacher dataset" n_kept=size(x1_teach_cpu, 2) n_total=n_teacher
    teacher_x1       = x1_teach_cpu       |> DEV
    teacher_neg_logp = neg_logp_teach_cpu |> DEV

    # ─── 3. raw Flowfusion EBM ───
    local ebm_raw, raw_hist
    if isfile(ebm_raw_path)
        @info "loading saved raw FF-EBM (delete to retrain)" ebm_raw_path
        loaded   = deserialize(ebm_raw_path)
        ebm_raw  = loaded.model |> DEV
        raw_hist = loaded.hist
    else
        @info "training raw FF-EBM from scratch" iters_ebm
        ebm_raw  = EBMFlowWrapper(EBM(embeddim = 256, nlayers = 3)) |> DEV
        raw_hist = train_ebm!(ebm_raw; iters = iters_ebm, batch = batch,
                              desc = "training raw FF-EBM")
        serialize(ebm_raw_path, (model = ebm_raw |> Flux.cpu, hist = raw_hist))
    end

    # ─── 4. regularised Flowfusion EBM ───
    local ebm_reg, reg_hist
    if isfile(ebm_reg_path)
        @info "loading saved reg FF-EBM (delete to retrain)" ebm_reg_path
        loaded   = deserialize(ebm_reg_path)
        ebm_reg  = loaded.model |> DEV
        reg_hist = loaded.hist
    else
        @info "training regularised FF-EBM" iters_ebm aux_weight aux_cadence warmup_iters
        ebm_reg  = EBMFlowWrapper(EBM(embeddim = 256, nlayers = 3)) |> DEV
        reg_hist = train_ebm!(ebm_reg;
                              iters = iters_ebm, batch = batch,
                              teacher_x1 = teacher_x1,
                              teacher_neg_logp = teacher_neg_logp,
                              aux_weight = aux_weight,
                              aux_cadence = aux_cadence,
                              aux_batch = aux_batch,
                              warmup_iters = warmup_iters,
                              desc = "training reg FF-EBM")
        serialize(ebm_reg_path, (model = ebm_reg |> Flux.cpu, hist = reg_hist))
    end

    # ─── 5. likelihood-scatter evaluation ───
    x0_eval = sampleX0_cpu(n_eval)
    x_eval_cpu, _ = sample_fewstep_with_logpdf(fewstep_cpu, x0_eval; nsteps = 4)
    true_logp  = literal_cat_logpdf(x_eval_cpu; sigma = Float64(CAT_SIGMA))
    keep       = isfinite.(true_logp)
    x_eval_cpu = x_eval_cpu[:, keep]
    true_logp  = true_logp[keep]

    ebm_raw_cpu = (ebm_raw |> Flux.cpu).ebm
    ebm_reg_cpu = (ebm_reg |> Flux.cpu).ebm
    t0          = zeros(T, length(true_logp))
    E_raw = Float64.(Array(ebm_raw_cpu(x_eval_cpu, t0)))
    E_reg = Float64.(Array(ebm_reg_cpu(x_eval_cpu, t0)))

    pred_raw = -E_raw .+ (mean(true_logp) - mean(-E_raw))
    pred_reg = -E_reg .+ (mean(true_logp) - mean(-E_reg))

    function scatter_with_r2(tru, prd, title_str)
        r2 = r_squared(tru, prd)
        lo = min(minimum(tru), minimum(prd))
        hi = max(maximum(tru), maximum(prd))
        p  = scatter(tru, prd; msw = 0, ms = 2, alpha = 0.55, color = "black",
                     xlabel = "true log p(x)",
                     ylabel = "predicted log p(x) = -E_θ(x,0) + c",
                     title  = "$(title_str)   R² = $(round(r2; digits = 3))",
                     label  = "samples", legend = :topleft)
        plot!(p, [lo, hi], [lo, hi]; color = "red", lw = 2, label = "y = x")
        return p, r2
    end

    (m99_r, m90_r) = tail_cuts(true_logp, pred_raw)
    (m99_g, m90_g) = tail_cuts(true_logp, pred_reg)

    p_r_full, r2_raw_full = scatter_with_r2(true_logp,        pred_raw,        "raw FF-EBM (full)")
    p_r_99,   r2_raw_99   = scatter_with_r2(true_logp[m99_r], pred_raw[m99_r], "raw FF-EBM (99%-cut)")
    p_r_90,   r2_raw_90   = scatter_with_r2(true_logp[m90_r], pred_raw[m90_r], "raw FF-EBM (90%-cut)")
    p_g_full, r2_reg_full = scatter_with_r2(true_logp,        pred_reg,        "reg FF-EBM (full)")
    p_g_99,   r2_reg_99   = scatter_with_r2(true_logp[m99_g], pred_reg[m99_g], "reg FF-EBM (99%-cut)")
    p_g_90,   r2_reg_90   = scatter_with_r2(true_logp[m90_g], pred_reg[m90_g], "reg FF-EBM (90%-cut)")

    p_all = plot(p_r_full, p_r_99, p_r_90,
                 p_g_full, p_g_99, p_g_90;
                 layout = (2, 3), size = (1800, 1100))
    savefig(p_all, joinpath(outdir, "ff_reg_ebm_likelihood_comparison.svg"))
    savefig(p_all, joinpath(outdir, "ff_reg_ebm_likelihood_comparison.png"))

    header = "R² — predicted vs true log-likelihood on few-step-generated eval points (Flowfusion EBM)\n" *
             "                     full       99%-cut    90%-cut\n" *
             "  raw FF-EBM       : $(rpad(round(r2_raw_full; digits=4), 10))" *
                                   "$(rpad(round(r2_raw_99;   digits=4), 10))" *
                                   "$(rpad(round(r2_raw_90;   digits=4), 10))\n" *
             "  regularised FF-EBM: $(rpad(round(r2_reg_full; digits=4), 10))" *
                                   "$(rpad(round(r2_reg_99;   digits=4), 10))" *
                                   "$(rpad(round(r2_reg_90;   digits=4), 10))"
    println(stdout, header)
    open(joinpath(outdir, "ff_reg_ebm_r2.txt"), "w") do io
        println(io, header)
    end

    # ─── 6. training-loss plots ───
    function loss_panel(hist, title_str)
        xs = 1:length(hist.fm_hist)
        p = plot(xs, hist.fm_hist; yscale = :log10, lw = 1, color = :steelblue,
                 xlabel = "iter", ylabel = "loss", title = title_str,
                 label = "floss", legend = :topright)
        if any(hist.aux_active_hist)
            idx = findall(hist.aux_active_hist)
            scatter!(p, idx, hist.aux_hist[idx]; ms = 1, msw = 0, color = :darkorange,
                     alpha = 0.7, label = "aux (on fire iters)")
        end
        return p
    end
    p_losses = plot(loss_panel(raw_hist, "raw FF-EBM"),
                    loss_panel(reg_hist, "regularised FF-EBM");
                    layout = (1, 2), size = (1200, 450))
    savefig(p_losses, joinpath(outdir, "ff_reg_ebm_loss_comparison.svg"))
    savefig(p_losses, joinpath(outdir, "ff_reg_ebm_loss_comparison.png"))

    # ─── 7. sample plots (Flowfusion `gen` from the real prior) ───
    P        = BrownianMotion(BM_SIGMA)
    X0_inf   = ContinuousState(sampleX0(n_samples_plot))
    raw_samps = Array(tensor(gen(P, X0_inf, ebm_raw, gen_steps)))
    reg_samps = Array(tensor(gen(P, X0_inf, ebm_reg, gen_steps)))
    x1_true   = Array(sampleX1(n_samples_plot))
    x0c       = Array(tensor(X0_inf))

    function sample_panel(samps, title_str)
        p = scatter(x1_true[1, :], x1_true[2, :]; msw = 0, ms = 1, color = "orange",
                    alpha = 0.5, label = "X1 (true)", size = (600, 600),
                    title = title_str, legend = :topleft)
        scatter!(p, x0c[1, :], x0c[2, :]; msw = 0, ms = 1, color = "blue",
                 alpha = 0.25, label = "X0 (prior)")
        scatter!(p, samps[1, :], samps[2, :]; msw = 0, ms = 1, color = "green",
                 alpha = 0.45, label = "X1 (EBM)")
        return p
    end
    p_samples = plot(sample_panel(raw_samps, "raw FF-EBM"),
                     sample_panel(reg_samps, "regularised FF-EBM");
                     layout = (1, 2), size = (1200, 600))
    savefig(p_samples, joinpath(outdir, "ff_reg_ebm_samples_comparison.svg"))
    savefig(p_samples, joinpath(outdir, "ff_reg_ebm_samples_comparison.png"))

    return (; ebm_raw, ebm_reg, fewstep_model, raw_hist, reg_hist,
            r2_raw = (full = r2_raw_full, p99 = r2_raw_99, p90 = r2_raw_90),
            r2_reg = (full = r2_reg_full, p99 = r2_reg_99, p90 = r2_reg_90))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
