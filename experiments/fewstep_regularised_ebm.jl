# Reuses the TwoTimeMap / few-step logpdf machinery defined in fewstep_map.jl
# (include is safe: fewstep_map.jl's `main()` is gated on `abspath(PROGRAM_FILE)`).
include(joinpath(@__DIR__, "fewstep_map.jl"))

using Jester: grad_fd

# =====================================================================
#  Regularised EBM vs raw EBM comparison.
#
#  1. Train / load a few-step TwoTimeMap (fewstep_map_model.bin if present).
#  2. Use it to produce a frozen "teacher" dataset (x1, −log p̂(x1)) via its
#     forward sample+logpdf pass.
#  3. Train a *raw* EBM by pure denoising score matching.
#  4. Train a *regularised* EBM with the same DSM loss plus an auxiliary
#     offset-invariant MSE between E_θ(x, 0) and the teacher's −log p̂ — fired
#     every `aux_cadence` iters, after a `warmup_iters` pure-DSM warmup.
#  5. Scatter both EBMs' predicted log-density against the ground-truth
#     literal-cat log-density on few-step-generated eval points, report R² at
#     full / 99-pct / 90-pct residual cuts.
# =====================================================================

# Variance-exploding noise schedule for the EBM (same as experiments/ebm.jl).
const SIGMA_MIN = T(0.01)
const SIGMA_MAX = T(2.0)
sigma_at(t) = SIGMA_MIN .* (SIGMA_MAX / SIGMA_MIN) .^ t

# ---------- EBM architecture ----------
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
    # Tuple, not Vector — Jester.grad_fd's rrule recurses through Tuple/NamedTuple
    # tangents but not Vector-of-NamedTuple tangents.
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

# ---------- losses ----------
function score_loss(model, x1, t, ε_noise; fd_eps = T(1e-3))
    σv  = reshape(sigma_at(t), 1, :)
    x_t = x1 .+ σv .* ε_noise
    f(x_, m_) = sum(m_(x_, t))
    gradx = grad_fd(f, x_t, model; ε = fd_eps)
    mean(sum(abs2.(ε_noise .- σv .* gradx); dims = 1))
end

# Offset-invariant MSE: both the model's energy head and the teacher's −log p̂ are
# centered per-batch, so the constant identifiability of an EBM's absolute energy
# does not corrupt the signal — we only constrain the *shape*.
function aux_loss(model, x1_aux, t_aux, neg_logp_teacher)
    E   = model(x1_aux, t_aux)
    E_c = E                  .- mean(E)
    g_c = neg_logp_teacher   .- mean(neg_logp_teacher)
    mean((E_c .- g_c) .^ 2)
end

# ---------- trainer ----------
"""
    train_ebm!(model; ...)

Pure DSM if `teacher_x1 === nothing`. Otherwise also applies the auxiliary
energy-vs-teacher-logp loss on a minibatch of teacher samples once every
`aux_cadence` iterations, after a `warmup_iters` pure-DSM warmup.
"""
function train_ebm!(model;
                    iters = 20000, batch = 4096, eta = 1f-3,
                    fd_eps = T(1e-3),
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
    η_type = typeof(float(eta))
    η_base = η_type(eta)
    η      = η_base
    opt    = Flux.setup(AdamW(eta = η), model)
    tot_hist = Float32[]; sm_hist = Float32[]; aux_hist = Float32[]
    eta_hist = Float32[]; aux_active_hist = Bool[]
    cooldown_start  = iters - floor(Int, cooldown_frac * iters)
    cooldown_length = max(iters - cooldown_start, 1)
    final_factor    = η_type(cooldown_final_factor)
    prog_desc = desc === nothing ?
                (regularised ? "training reg EBM" : "training raw EBM") : desc
    prog = Progress(iters; desc = prog_desc, dt = 0.5, showspeed = true)
    for iter in 1:iters
        x1    = sampleX1(batch)
        t     = rand(T, batch) |> DEV
        ε_nse = randn!(similar(x1))
        use_aux = regularised && iter > warmup_iters && (iter % aux_cadence == 0)
        if use_aux
            idx      = rand(1:n_teacher, aux_batch)
            x1_aux   = teacher_x1[:, idx]
            nlp_aux  = teacher_neg_logp[idx]
            t0_aux   = fill!(similar(x1_aux, aux_batch), T(0))
        end
        val, g = Flux.withgradient(model) do m
            sm = score_loss(m, x1, t, ε_nse; fd_eps = fd_eps)
            if use_aux
                a     = aux_loss(m, x1_aux, t0_aux, nlp_aux)
                total = sm + aux_weight * a
                (total, sm, a)
            else
                (sm, sm, zero(sm))
            end
        end
        total_val, sm_val, aux_val = val
        Flux.update!(opt, model, g[1])
        push!(tot_hist, Float32(total_val))
        push!(sm_hist,  Float32(sm_val))
        push!(aux_hist, Float32(aux_val))
        push!(aux_active_hist, use_aux)
        if iter > cooldown_start
            progress = η_type((iter - cooldown_start) / cooldown_length)
            η = η_base * (one(η_type) - progress * (one(η_type) - final_factor))
            Optimisers.adjust!(opt, η)
        end
        push!(eta_hist, Float32(η))
        next!(prog; showvalues = [(:iter, iter),
                                  (:sm,   round(sm_val; digits = 4)),
                                  (:aux,  round(aux_val; digits = 4)),
                                  (:tot,  round(total_val; digits = 4)),
                                  (:eta,  Float32(η))])
    end
    return (; tot_hist, sm_hist, aux_hist, eta_hist, aux_active_hist,
            regularised, aux_weight, aux_cadence, warmup_iters)
end

# ---------- sampling ----------
# Annealed Langevin dynamics, matching experiments/ebm.jl.
function annealed_langevin(model, x0;
                           n_sigmas = 20,
                           n_steps_per_sigma = 80,
                           eps = T(2f-5),
                           fd_eps = T(1e-3))
    sigmas = T.(SIGMA_MAX .* (SIGMA_MIN / SIGMA_MAX) .^ range(0, 1, length = n_sigmas))
    x = copy(x0)
    for σi in sigmas
        t_val = T(log(σi / SIGMA_MIN) / log(SIGMA_MAX / SIGMA_MIN))
        tvec  = similar(x, size(x, 2)); fill!(tvec, t_val)
        α     = eps * (σi / SIGMA_MIN)^2
        for _ in 1:n_steps_per_sigma
            f_(x_, m_) = sum(m_(x_, tvec))
            g     = grad_fd(f_, x, model; ε = fd_eps)
            score = -g
            noise = randn!(similar(x))
            x = x .+ (α / T(2)) .* score .+ sqrt(α) .* noise
        end
    end
    return x
end

const LANGEVIN_INIT_CENTER = T.([5.0, 5.0])
const LANGEVIN_INIT_STD    = T(0.5)
langevin_init(n) = LANGEVIN_INIT_CENTER .+ LANGEVIN_INIT_STD .* randn(T, SPACEDIM, n)

# ---------- teacher dataset ----------
"""
    generate_teacher_dataset(fewstep_model_cpu, n; nsteps = 4)

Runs `fewstep_model_cpu`'s forward sample+logpdf pass from n uniform base
samples and returns `(x1, -logp)` filtered to finite entries. Runs on CPU
(that's what `sample_fewstep_with_logpdf` supports) — the returned arrays
can be moved to DEV by the caller.
"""
function generate_teacher_dataset(fewstep_model_cpu, n; nsteps = 4, ε = T(1e-3))
    @info "generating teacher dataset" n nsteps
    x0 = sampleX0_cpu(n)
    x1, logp = sample_fewstep_with_logpdf(fewstep_model_cpu, x0; nsteps = nsteps, ε = ε)
    keep = isfinite.(logp)
    return x1[:, keep], T.(-logp[keep])
end

# ---------- evaluation ----------
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

# ---------- main ----------
function main(; iters_fewstep_per_epoch = 4000,
              fewstep_last_epoch_multiplier = 5,
              iters_ebm      = 20000,
              batch          = 4096,
              n_teacher      = 50000,
              n_eval         = 2000,
              aux_weight     = T(0.1),
              aux_cadence    = 5,
              aux_batch      = 1024,
              warmup_iters   = 50,
              outdir         = @__DIR__,
              fewstep_path   = joinpath(@__DIR__, "fewstep_map_model.bin"),
              ebm_raw_path   = joinpath(@__DIR__, "reg_ebm_raw_model.bin"),
              ebm_reg_path   = joinpath(@__DIR__, "reg_ebm_reg_model.bin"))
    # ─── 1. fewstep teacher ───
    local fewstep_model
    if isfile(fewstep_path)
        @info "loading saved fewstep model (delete to retrain)" fewstep_path
        loaded = deserialize(fewstep_path)
        fewstep_model = loaded.model |> DEV
    else
        @info "training fewstep model from scratch" iters_fewstep_per_epoch fewstep_last_epoch_multiplier
        fewstep_model = TwoTimeMap(embeddim = 256, nlayers = 3) |> DEV
        # `train!` here is fewstep_map.jl's TwoTimeMap trainer, brought in by include.
        hist_fs = train!(fewstep_model;
                         iters_per_epoch = iters_fewstep_per_epoch,
                         last_epoch_multiplier = fewstep_last_epoch_multiplier)
        serialize(fewstep_path, (model = fewstep_model |> Flux.cpu, hist = hist_fs))
        @info "saved fewstep model" fewstep_path
    end
    fewstep_cpu = fewstep_model |> Flux.cpu

    # ─── 2. teacher dataset ───
    x1_teach_cpu, neg_logp_teach_cpu = generate_teacher_dataset(fewstep_cpu, n_teacher)
    @info "teacher dataset" n_kept=size(x1_teach_cpu, 2) n_total=n_teacher
    teacher_x1       = x1_teach_cpu  |> DEV
    teacher_neg_logp = neg_logp_teach_cpu |> DEV

    # ─── 3. raw EBM ───
    local ebm_raw, raw_hist
    if isfile(ebm_raw_path)
        @info "loading saved raw EBM (delete to retrain)" ebm_raw_path
        loaded   = deserialize(ebm_raw_path)
        ebm_raw  = loaded.model |> DEV
        raw_hist = loaded.hist
    else
        @info "training raw EBM from scratch" iters_ebm
        ebm_raw  = EBM(embeddim = 256, nlayers = 3) |> DEV
        raw_hist = train_ebm!(ebm_raw; iters = iters_ebm, batch = batch,
                              desc = "training raw EBM")
        serialize(ebm_raw_path, (model = ebm_raw |> Flux.cpu, hist = raw_hist))
    end

    # ─── 4. regularised EBM ───
    local ebm_reg, reg_hist
    if isfile(ebm_reg_path)
        @info "loading saved reg EBM (delete to retrain)" ebm_reg_path
        loaded   = deserialize(ebm_reg_path)
        ebm_reg  = loaded.model |> DEV
        reg_hist = loaded.hist
    else
        @info "training regularised EBM" iters_ebm aux_weight aux_cadence warmup_iters
        ebm_reg  = EBM(embeddim = 256, nlayers = 3) |> DEV
        reg_hist = train_ebm!(ebm_reg;
                              iters = iters_ebm, batch = batch,
                              teacher_x1 = teacher_x1,
                              teacher_neg_logp = teacher_neg_logp,
                              aux_weight = aux_weight,
                              aux_cadence = aux_cadence,
                              aux_batch = aux_batch,
                              warmup_iters = warmup_iters,
                              desc = "training reg EBM")
        serialize(ebm_reg_path, (model = ebm_reg |> Flux.cpu, hist = reg_hist))
    end

    # ─── 5. evaluation ───
    # Use fewstep-generated points as the eval set (matches the distribution
    # fewstep_map.jl's scatter uses).
    x0_eval = sampleX0_cpu(n_eval)
    x_eval_cpu, _ = sample_fewstep_with_logpdf(fewstep_cpu, x0_eval; nsteps = 4)
    true_logp = literal_cat_logpdf(x_eval_cpu; sigma = Float64(CAT_SIGMA))
    keep       = isfinite.(true_logp)
    x_eval_cpu = x_eval_cpu[:, keep]
    true_logp  = true_logp[keep]

    ebm_raw_cpu = ebm_raw |> Flux.cpu
    ebm_reg_cpu = ebm_reg |> Flux.cpu
    t0          = zeros(T, length(true_logp))
    E_raw = Float64.(Array(ebm_raw_cpu(x_eval_cpu, t0)))
    E_reg = Float64.(Array(ebm_reg_cpu(x_eval_cpu, t0)))

    # Predicted log-density: -E + c, with c chosen so means agree with the truth.
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

    p_r_full, r2_raw_full = scatter_with_r2(true_logp,        pred_raw,        "raw EBM (full)")
    p_r_99,   r2_raw_99   = scatter_with_r2(true_logp[m99_r], pred_raw[m99_r], "raw EBM (99%-cut)")
    p_r_90,   r2_raw_90   = scatter_with_r2(true_logp[m90_r], pred_raw[m90_r], "raw EBM (90%-cut)")
    p_g_full, r2_reg_full = scatter_with_r2(true_logp,        pred_reg,        "reg EBM (full)")
    p_g_99,   r2_reg_99   = scatter_with_r2(true_logp[m99_g], pred_reg[m99_g], "reg EBM (99%-cut)")
    p_g_90,   r2_reg_90   = scatter_with_r2(true_logp[m90_g], pred_reg[m90_g], "reg EBM (90%-cut)")

    p_all = plot(p_r_full, p_r_99, p_r_90,
                 p_g_full, p_g_99, p_g_90;
                 layout = (2, 3), size = (1800, 1100))
    savefig(p_all, joinpath(outdir, "reg_ebm_likelihood_comparison.svg"))
    savefig(p_all, joinpath(outdir, "reg_ebm_likelihood_comparison.png"))

    # Report.
    header = "R² — predicted vs true log-likelihood on few-step-generated eval points\n" *
             "                    full       99%-cut    90%-cut\n" *
             "  raw EBM         : $(rpad(round(r2_raw_full; digits=4), 10))" *
                                  "$(rpad(round(r2_raw_99;   digits=4), 10))" *
                                  "$(rpad(round(r2_raw_90;   digits=4), 10))\n" *
             "  regularised EBM : $(rpad(round(r2_reg_full; digits=4), 10))" *
                                  "$(rpad(round(r2_reg_99;   digits=4), 10))" *
                                  "$(rpad(round(r2_reg_90;   digits=4), 10))"
    println(stdout, header)
    open(joinpath(outdir, "reg_ebm_r2.txt"), "w") do io
        println(io, header)
    end

    # ─── 6. training-loss plots ───
    function loss_panel(hist, title_str)
        xs = 1:length(hist.sm_hist)
        p = plot(xs, hist.sm_hist; yscale = :log10, lw = 1, color = :steelblue,
                 xlabel = "iter", ylabel = "loss", title = title_str,
                 label = "score-matching", legend = :topright)
        if any(hist.aux_active_hist)
            idx = findall(hist.aux_active_hist)
            scatter!(p, idx, hist.aux_hist[idx]; ms = 1, msw = 0, color = :darkorange,
                     alpha = 0.7, label = "aux (on fire iters)")
        end
        return p
    end
    p_loss_raw = loss_panel(raw_hist, "raw EBM")
    p_loss_reg = loss_panel(reg_hist, "regularised EBM")
    p_losses   = plot(p_loss_raw, p_loss_reg; layout = (1, 2), size = (1200, 450))
    savefig(p_losses, joinpath(outdir, "reg_ebm_loss_comparison.svg"))
    savefig(p_losses, joinpath(outdir, "reg_ebm_loss_comparison.png"))

    # ─── 7. sample-quality plots (annealed Langevin) ───
    n_inf   = 5000
    x0_init = langevin_init(n_inf) |> DEV
    raw_samps = Array(annealed_langevin(ebm_raw, x0_init))
    reg_samps = Array(annealed_langevin(ebm_reg, x0_init))
    x1_true   = Array(sampleX1(n_inf))
    init_c    = Array(x0_init)

    function sample_panel(samps, title_str)
        p = scatter(x1_true[1, :], x1_true[2, :]; msw = 0, ms = 1, color = "orange",
                    alpha = 0.5, label = "X1 (true)", size = (600, 600),
                    title = title_str, legend = :topleft)
        scatter!(p, init_c[1, :], init_c[2, :]; msw = 0, ms = 1, color = "blue",
                 alpha = 0.25, label = "Langevin init")
        scatter!(p, samps[1, :], samps[2, :]; msw = 0, ms = 1, color = "green",
                 alpha = 0.45, label = "X1 (EBM)")
        return p
    end
    p_samp_raw = sample_panel(raw_samps, "raw EBM")
    p_samp_reg = sample_panel(reg_samps, "regularised EBM")
    p_samples  = plot(p_samp_raw, p_samp_reg; layout = (1, 2), size = (1200, 600))
    savefig(p_samples, joinpath(outdir, "reg_ebm_samples_comparison.svg"))
    savefig(p_samples, joinpath(outdir, "reg_ebm_samples_comparison.png"))

    return (; ebm_raw, ebm_reg, fewstep_model, raw_hist, reg_hist,
            r2_raw = (full = r2_raw_full, p99 = r2_raw_99, p90 = r2_raw_90),
            r2_reg = (full = r2_reg_full, p99 = r2_reg_99, p90 = r2_reg_90))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
