# ProteinEBM-style (Roney, Ou & Ovchinnikov, bioRxiv 2025.12.09.693073)
# rewrite of what was a Flowfusion-`floss` / X̂₁-regression EBM. The old
# version trained a scalar energy head by wrapping it in an X̂₁ predictor
# `x + (1.05 − t)·(−∇E)` and minimizing `floss` on a Brownian-motion
# bridge. That is an X₁-loss, not score matching: its minimizer has no
# reason to satisfy `E_θ(·, 0) ∝ −log p_{data}`, so the scalar head is
# not a bona fide energy for X₁.
#
# ProteinEBM's training language (§3 / Appendix A) fixes this by:
#
#   1. Forward process (VP):
#          x_t = √β̄_t · x_0 + √(1 − β̄_t) · ε,     ε ~ N(0, I)
#      with β̄_t = exp[−½ (b₁ t + b₂ t²)],  b₁ = 0.1, b₂ = 19.99
#      (Yim et al. 2023 / paper App. A).
#
#   2. Explicitly conservative score:  s_θ(x_t, t) := −∇_x E_θ(x_t, t).
#
#   3. Denoising score matching loss (paper §3):
#          L_DSM = ‖ −∇_x E_θ(x_t, t) − (x_t − √β̄_t x_0)/(1 − β̄_t) ‖²
#      which, by Vincent's identity, drives s_θ toward the marginal
#      score ∇ log p_t, and thus E_θ(·, t) → −log p_t + C_t.
#      We use the σ-weighted ("ε-prediction") form
#          ‖ √(1 − β̄_t) · ∇_x E_θ − ε ‖²
#      which has identical minimizer but stays finite as β̄_t → 1.
#
# Convention (matches the paper, *not* Flowfusion's bridge indexing):
#   `x_0` ≡ clean data sample — in this codebase, that is `sampleX1`.
#   t = 0 is clean; t = 1 is (near-)pure N(0, I) noise.
#
# Practical caveat on "t = 0": at t = 0 the DSM signal on ∇E vanishes
# (√(1 − β̄_0) = 0), so the model has *no* gradient information there.
# Both aux-loss matching and likelihood evaluation are therefore done at
# a small-but-nonzero `T_EVAL` — exactly the knob ProteinEBM uses when
# they rank decoys at t = 0.1 instead of t = 0. For this 2-D cat toy the
# cat modes have width CAT_SIGMA ≈ 0.05, so T_EVAL is chosen to keep the
# marginal noise std well below that.
#
# Raw-vs-regularised comparison is structurally the same as
# `fewstep_regularised_ebm.jl`: aux loss fires every `aux_cadence`
# iterations after warmup, offset-invariantly matching E_θ(x, T_EVAL) to
# the few-step teacher's −log p̂(x).

include(joinpath(@__DIR__, "fewstep_map.jl"))

using CannotWaitForTheseOptimisers
using Jester: grad_fd

# =====================================================================
# VP noise schedule  (Roney et al. App. A; Yim et al. 2023)
#     β̄_t = exp[ −½ (b₁ t + b₂ t²) ],   b₁ = 0.1, b₂ = 19.99
# β̄_0 = 1 (clean), β̄_1 ≈ 6.4e-3 (essentially pure N(0, I) noise).
# =====================================================================
const VP_B1 = T(0.1)
const VP_B2 = T(19.99)

alpha_bar(t) = exp.(T(-0.5) .* (VP_B1 .* t .+ VP_B2 .* t .^ 2))

# Small-but-nonzero evaluation time (see caveat above).
const T_EVAL = T(2f-3)
const FD_EPS = T(1e-3)

# =====================================================================
# EBM  —  scalar-energy head, identical architecture to
# `fewstep_regularised_ebm.jl`. The score used for DSM and Langevin is
# always −∇_x of this scalar output, so it is conservative by construction.
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

# =====================================================================
# Losses
# =====================================================================
# DSM on the VP process, σ-weighted so it is finite at t → 0.
# Target is ε directly (the noise draw), which — by Tweedie / Vincent —
# expands out to matching −∇E to the conditional score of p(x_t | x_0).
function dsm_loss(model::EBM, x0_data, t; fd_eps = FD_EPS)
    ab     = reshape(alpha_bar(t), 1, :)
    sq_ab  = sqrt.(ab)
    sig    = sqrt.(one(T) .- ab)                 # = √(1 − β̄_t)
    eps_n  = randn!(similar(x0_data))
    x_t    = sq_ab .* x0_data .+ sig .* eps_n
    f(x_, m_) = sum(m_(x_, t))
    gE     = grad_fd(f, x_t, model; ε = fd_eps)
    mean(sum(abs2.(sig .* gE .- eps_n); dims = 1))
end

# Offset-invariant MSE: both the EBM's energy head and the teacher's
# −log p̂ are centered per-batch, so the EBM's free additive constant
# does not corrupt the signal — we only constrain the *shape*.
function aux_loss(model::EBM, x0_aux, t_aux, neg_logp_teacher)
    E   = model(x0_aux, t_aux)
    E_c = E                .- mean(E)
    g_c = neg_logp_teacher .- mean(neg_logp_teacher)
    mean((E_c .- g_c) .^ 2)
end

# =====================================================================
# Trainer  —  raw or regularised VP-EBM
# =====================================================================
"""
    train_ebm!(model; ...)

Raw if `teacher_x1 === nothing`: pure DSM on the VP process. Otherwise
adds the energy-vs-teacher-logp auxiliary loss on a random teacher
minibatch every `aux_cadence` iters after `warmup_iters` of warmup.
"""
function train_ebm!(model::EBM;
                    iters = 20000, batch = 4096, eta = 1f-3,
                    fd_eps = FD_EPS,
                    teacher_x1 = nothing, teacher_neg_logp = nothing,
                    aux_weight = T(0.1),
                    aux_cadence = 5,
                    aux_batch = 1024,
                    warmup_iters = 50,
                    cooldown_frac = 0.2,
                    cooldown_final_factor = 0.01,
                    t_min = T_EVAL,
                    t_max = T(1f0),
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
    opt    = Flux.setup(Muon(eta = η), model)
    tot_hist = Float32[]; dsm_hist = Float32[]; aux_hist = Float32[]
    eta_hist = Float32[]; aux_active_hist = Bool[]
    cooldown_start  = iters - floor(Int, cooldown_frac * iters)
    cooldown_length = max(iters - cooldown_start, 1)
    final_factor    = η_type(cooldown_final_factor)
    prog_desc = desc === nothing ?
                (regularised ? "training reg VP-EBM" : "training raw VP-EBM") : desc
    prog = Progress(iters; desc = prog_desc, dt = 0.5, showspeed = true)
    for iter in 1:iters
        # `sampleX1` is the clean-data sampler; in paper notation that's x_0.
        x0_data = sampleX1(batch)
        t       = (t_min .+ (t_max - t_min) .* rand(T, batch)) |> DEV
        use_aux = regularised && iter > warmup_iters && (iter % aux_cadence == 0)
        if use_aux
            idx     = rand(1:n_teacher, aux_batch)
            x0_aux  = teacher_x1[:, idx]
            nlp_aux = teacher_neg_logp[idx]
            t_aux   = fill!(similar(x0_aux, aux_batch), T_EVAL)
        end
        val, g = Flux.withgradient(model) do m
            sm = dsm_loss(m, x0_data, t; fd_eps = fd_eps)
            if use_aux
                a     = aux_loss(m, x0_aux, t_aux, nlp_aux)
                total = sm + aux_weight * a
                (total, sm, a)
            else
                (sm, sm, zero(sm))
            end
        end
        total_val, dsm_val, aux_val = val
        Flux.update!(opt, model, g[1])
        push!(tot_hist, Float32(total_val))
        push!(dsm_hist, Float32(dsm_val))
        push!(aux_hist, Float32(aux_val))
        push!(aux_active_hist, use_aux)
        if iter > cooldown_start
            progress = η_type((iter - cooldown_start) / cooldown_length)
            η = η_base * (one(η_type) - progress * (one(η_type) - final_factor))
            Optimisers.adjust!(opt, η)
        end
        push!(eta_hist, Float32(η))
        next!(prog; showvalues = [(:iter, iter),
                                  (:dsm,   round(dsm_val;   digits = 4)),
                                  (:aux,   round(aux_val;   digits = 4)),
                                  (:tot,   round(total_val; digits = 4)),
                                  (:eta,   Float32(η))])
    end
    return (; tot_hist, dsm_hist, aux_hist, eta_hist, aux_active_hist,
            regularised, aux_weight, aux_cadence, warmup_iters)
end

# =====================================================================
# Teacher dataset  (unchanged from the prior version)
# =====================================================================
function generate_teacher_dataset(fewstep_model_cpu, n; nsteps = 4, ε = T(1e-3))
    @info "generating teacher dataset" n nsteps
    x0 = sampleX0_cpu(n)
    x1, logp = sample_fewstep_with_logpdf(fewstep_model_cpu, x0; nsteps = nsteps, ε = ε)
    keep = isfinite.(logp)
    return x1[:, keep], T.(-logp[keep])
end

# =====================================================================
# Sampling — annealed overdamped Langevin down the VP noise ladder
# (ProteinEBM Appendix B.1). For each noise level τ we run
#   x ← x − α_τ ∇_x E_θ(x, τ) + √(2 α_τ)·ξ ,
# with α_τ = eps · σ²(τ) / σ²(t_lo). This is the (σ_τ/σ_min)² scaling
# used by `fewstep_regularised_ebm.jl` / Song & Ermon 2019 — it makes
# the effective Langevin drift α·‖∇E‖ and diffusion √α scale with the
# conditional variance at level τ, so steps at high-noise levels are
# large enough to traverse the data manifold while low-noise levels
# only do local refinement. Without this scaling α stays O(eps)
# across the whole ladder and samples never leave the N(0, I) init.
# Start from N(0, I) — the VP prior at t = 1.
# =====================================================================
function annealed_langevin(model::EBM, x_init;
                           n_levels = 20,
                           n_steps_per_level = 80,
                           eps = T(2f-5),
                           fd_eps = FD_EPS,
                           t_hi = T(1f0),
                           t_lo = T_EVAL)
    levels  = T.(range(t_hi, t_lo; length = n_levels))
    sig2_lo = one(T) - exp(T(-0.5) * (VP_B1 * t_lo + VP_B2 * t_lo^2))
    x = copy(x_init)
    for t_val in levels
        ab    = exp(T(-0.5) * (VP_B1 * t_val + VP_B2 * t_val^2))
        sig2  = one(T) - ab
        α     = eps * sig2 / sig2_lo
        tvec  = similar(x, size(x, 2)); fill!(tvec, t_val)
        for _ in 1:n_steps_per_level
            f_(x_, m_) = sum(m_(x_, tvec))
            gE    = grad_fd(f_, x, model; ε = fd_eps)
            x     = x .- α .* gE .+ sqrt(T(2) * α) .* randn!(similar(x))
        end
    end
    return x
end

# VP prior init: pure N(0, I).
langevin_init(n) = randn(T, SPACEDIM, n)

# =====================================================================
# Evaluation helpers
# =====================================================================
# R² under the intercept-only linear model  true = b₀ + pred  (slope fixed
# at 1). LS solution is b₀ = mean(true) − mean(pred). The EBM's free
# additive constant in −log p gets absorbed into b₀ — so this is the
# right R² for "does the shape of −E track the shape of the true log p".
function intercept_ls_r2(true_y::AbstractVector{<:Real},
                         pred_y::AbstractVector{<:Real})
    b0     = mean(true_y) - mean(pred_y)
    fitted = pred_y .+ b0
    ss_res = sum((true_y .- fitted) .^ 2)
    ss_tot = sum((true_y .- mean(true_y)) .^ 2)
    1 - ss_res / ss_tot
end

# Spearman = Pearson on ranks. `invperm(sortperm(·))` gives the rank of
# each element; ties are broken arbitrarily (fine for continuous log-p
# values where exact ties have measure zero).
function spearman_rank(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    @assert length(x) == length(y)
    rx = Float64.(invperm(sortperm(x)))
    ry = Float64.(invperm(sortperm(y)))
    mx = mean(rx); my = mean(ry)
    num = sum((rx .- mx) .* (ry .- my))
    den = sqrt(sum((rx .- mx) .^ 2) * sum((ry .- my) .^ 2))
    num / den
end

# =====================================================================
# Main
# =====================================================================
function main(; iters_fewstep_per_epoch = 4000,
              fewstep_last_epoch_multiplier = 5,
              iters_ebm      = 20000,
              batch          = 4096,
              n_teacher      = 50000,
              n_eval         = 2000,
              n_samples_plot = 5000,
              aux_weight     = T(1.0),
              aux_cadence    = 5,
              aux_batch      = 1024,
              warmup_iters   = 50,
              outdir         = @__DIR__,
              fewstep_path   = joinpath(@__DIR__, "fewstep_map_model.bin"),
              ebm_raw_path   = joinpath(@__DIR__, "vp_ebm_raw_model.bin"),
              ebm_reg_path   = joinpath(@__DIR__, "vp_ebm_reg_model.bin"))
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

    # ─── 3. raw VP-EBM ───
    local ebm_raw, raw_hist
    if isfile(ebm_raw_path)
        @info "loading saved raw VP-EBM (delete to retrain)" ebm_raw_path
        loaded   = deserialize(ebm_raw_path)
        ebm_raw  = loaded.model |> DEV
        raw_hist = loaded.hist
    else
        @info "training raw VP-EBM from scratch" iters_ebm
        ebm_raw  = EBM(embeddim = 256, nlayers = 3) |> DEV
        raw_hist = train_ebm!(ebm_raw; iters = iters_ebm, batch = batch,
                              desc = "training raw VP-EBM")
        serialize(ebm_raw_path, (model = ebm_raw |> Flux.cpu, hist = raw_hist))
    end

    # ─── 4. regularised VP-EBM ───
    local ebm_reg, reg_hist
    if isfile(ebm_reg_path)
        @info "loading saved reg VP-EBM (delete to retrain)" ebm_reg_path
        loaded   = deserialize(ebm_reg_path)
        ebm_reg  = loaded.model |> DEV
        reg_hist = loaded.hist
    else
        @info "training regularised VP-EBM" iters_ebm aux_weight aux_cadence warmup_iters
        ebm_reg  = EBM(embeddim = 256, nlayers = 3) |> DEV
        reg_hist = train_ebm!(ebm_reg;
                              iters = iters_ebm, batch = batch,
                              teacher_x1 = teacher_x1,
                              teacher_neg_logp = teacher_neg_logp,
                              aux_weight = aux_weight,
                              aux_cadence = aux_cadence,
                              aux_batch = aux_batch,
                              warmup_iters = warmup_iters,
                              desc = "training reg VP-EBM")
        serialize(ebm_reg_path, (model = ebm_reg |> Flux.cpu, hist = reg_hist))
    end

    # ─── 5. likelihood-scatter evaluation (eval at t = T_EVAL) ───
    x0_eval = sampleX0_cpu(n_eval)
    x_eval_cpu, _ = sample_fewstep_with_logpdf(fewstep_cpu, x0_eval; nsteps = 4)
    true_logp  = literal_cat_logpdf(x_eval_cpu; sigma = Float64(CAT_SIGMA))
    keep       = isfinite.(true_logp)
    x_eval_cpu = x_eval_cpu[:, keep]
    true_logp  = true_logp[keep]

    ebm_raw_cpu = ebm_raw |> Flux.cpu
    ebm_reg_cpu = ebm_reg |> Flux.cpu
    t_eval      = fill(T_EVAL, length(true_logp))
    E_raw = Float64.(Array(ebm_raw_cpu(x_eval_cpu, t_eval)))
    E_reg = Float64.(Array(ebm_reg_cpu(x_eval_cpu, t_eval)))

    # Predicted log p = -E. The intercept is handled by `intercept_ls_r2`;
    # we only shift here so the scatter plot sits near y = x visually.
    pred_raw_raw = -E_raw
    pred_reg_raw = -E_reg
    pred_raw     = pred_raw_raw .+ (mean(true_logp) - mean(pred_raw_raw))
    pred_reg     = pred_reg_raw .+ (mean(true_logp) - mean(pred_reg_raw))

    function scatter_with_metrics(tru, prd_raw, prd_shift, title_str)
        r2  = intercept_ls_r2(tru, prd_raw)   # invariant under shifts
        ρ   = spearman_rank(tru, prd_raw)
        lo  = min(minimum(tru), minimum(prd_shift))
        hi  = max(maximum(tru), maximum(prd_shift))
        p   = scatter(tru, prd_shift; msw = 0, ms = 2, alpha = 0.55, color = "black",
                      xlabel = "true log p(x)",
                      ylabel = "predicted log p(x) = -E_θ(x, T_EVAL) + b₀",
                      title  = "$(title_str)   R² = $(round(r2; digits = 3))   ρ = $(round(ρ; digits = 3))",
                      label  = "samples", legend = :topleft)
        plot!(p, [lo, hi], [lo, hi]; color = "red", lw = 2, label = "y = x")
        return p, r2, ρ
    end

    # Tail cuts on residuals of the intercept-only fit.
    resid_raw = true_logp .- (pred_raw_raw .+ (mean(true_logp) - mean(pred_raw_raw)))
    resid_reg = true_logp .- (pred_reg_raw .+ (mean(true_logp) - mean(pred_reg_raw)))
    abs_r = abs.(resid_raw); abs_g = abs.(resid_reg)
    m99_r = abs_r .<= quantile(abs_r, 0.99); m90_r = abs_r .<= quantile(abs_r, 0.90)
    m99_g = abs_g .<= quantile(abs_g, 0.99); m90_g = abs_g .<= quantile(abs_g, 0.90)

    p_r_full, r2_raw_full, ρ_raw_full = scatter_with_metrics(true_logp,        pred_raw_raw,        pred_raw,        "raw VP-EBM (full)")
    p_r_99,   r2_raw_99,   ρ_raw_99   = scatter_with_metrics(true_logp[m99_r], pred_raw_raw[m99_r], pred_raw[m99_r], "raw VP-EBM (99%-cut)")
    p_r_90,   r2_raw_90,   ρ_raw_90   = scatter_with_metrics(true_logp[m90_r], pred_raw_raw[m90_r], pred_raw[m90_r], "raw VP-EBM (90%-cut)")
    p_g_full, r2_reg_full, ρ_reg_full = scatter_with_metrics(true_logp,        pred_reg_raw,        pred_reg,        "reg VP-EBM (full)")
    p_g_99,   r2_reg_99,   ρ_reg_99   = scatter_with_metrics(true_logp[m99_g], pred_reg_raw[m99_g], pred_reg[m99_g], "reg VP-EBM (99%-cut)")
    p_g_90,   r2_reg_90,   ρ_reg_90   = scatter_with_metrics(true_logp[m90_g], pred_reg_raw[m90_g], pred_reg[m90_g], "reg VP-EBM (90%-cut)")

    p_all = plot(p_r_full, p_r_99, p_r_90,
                 p_g_full, p_g_99, p_g_90;
                 layout = (2, 3), size = (1800, 1100))
    savefig(p_all, joinpath(outdir, "vp_ebm_likelihood_comparison.svg"))
    savefig(p_all, joinpath(outdir, "vp_ebm_likelihood_comparison.png"))

    fmt(x) = rpad(round(x; digits = 4), 10)
    header = "Intercept-only LS R² and Spearman ρ — predicted vs true log-likelihood\n" *
             "(predicted = −E_θ(x, T_EVAL); intercept b₀ estimated per cut via LS)\n" *
             "                             full       99%-cut    90%-cut\n" *
             "  raw VP-EBM          R²  : $(fmt(r2_raw_full))$(fmt(r2_raw_99))$(fmt(r2_raw_90))\n" *
             "  raw VP-EBM          ρ   : $(fmt(ρ_raw_full))$(fmt(ρ_raw_99))$(fmt(ρ_raw_90))\n" *
             "  regularised VP-EBM  R²  : $(fmt(r2_reg_full))$(fmt(r2_reg_99))$(fmt(r2_reg_90))\n" *
             "  regularised VP-EBM  ρ   : $(fmt(ρ_reg_full))$(fmt(ρ_reg_99))$(fmt(ρ_reg_90))"
    println(stdout, header)
    open(joinpath(outdir, "vp_ebm_r2.txt"), "w") do io
        println(io, header)
    end

    # ─── 6. training-loss plots ───
    function loss_panel(hist, title_str)
        xs = 1:length(hist.dsm_hist)
        p = plot(xs, hist.dsm_hist; yscale = :log10, lw = 1, color = :steelblue,
                 xlabel = "iter", ylabel = "loss", title = title_str,
                 label = "DSM", legend = :topright)
        if any(hist.aux_active_hist)
            idx = findall(hist.aux_active_hist)
            scatter!(p, idx, hist.aux_hist[idx]; ms = 1, msw = 0, color = :darkorange,
                     alpha = 0.7, label = "aux (on fire iters)")
        end
        return p
    end
    p_losses = plot(loss_panel(raw_hist, "raw VP-EBM"),
                    loss_panel(reg_hist, "regularised VP-EBM");
                    layout = (1, 2), size = (1200, 450))
    savefig(p_losses, joinpath(outdir, "vp_ebm_loss_comparison.svg"))
    savefig(p_losses, joinpath(outdir, "vp_ebm_loss_comparison.png"))

    # ─── 7. sample plots (annealed Langevin from the VP prior N(0, I)) ───
    x_init    = langevin_init(n_samples_plot) |> DEV
    raw_samps = Array(annealed_langevin(ebm_raw, x_init))
    reg_samps = Array(annealed_langevin(ebm_reg, x_init))
    x1_true   = Array(sampleX1(n_samples_plot))
    init_c    = Array(x_init)

    function sample_panel(samps, title_str)
        p = scatter(x1_true[1, :], x1_true[2, :]; msw = 0, ms = 1, color = "orange",
                    alpha = 0.5, label = "X1 (true)", size = (600, 600),
                    title = title_str, legend = :topleft)
        scatter!(p, init_c[1, :], init_c[2, :]; msw = 0, ms = 1, color = "blue",
                 alpha = 0.25, label = "Langevin init  (VP prior, N(0, I))")
        scatter!(p, samps[1, :], samps[2, :]; msw = 0, ms = 1, color = "green",
                 alpha = 0.45, label = "X1 (EBM)")
        return p
    end
    p_samples = plot(sample_panel(raw_samps, "raw VP-EBM"),
                     sample_panel(reg_samps, "regularised VP-EBM");
                     layout = (1, 2), size = (1200, 600))
    savefig(p_samples, joinpath(outdir, "vp_ebm_samples_comparison.svg"))
    savefig(p_samples, joinpath(outdir, "vp_ebm_samples_comparison.png"))

    return (; ebm_raw, ebm_reg, fewstep_model, raw_hist, reg_hist,
            r2_raw       = (full = r2_raw_full, p99 = r2_raw_99, p90 = r2_raw_90),
            r2_reg       = (full = r2_reg_full, p99 = r2_reg_99, p90 = r2_reg_90),
            spearman_raw = (full = ρ_raw_full,  p99 = ρ_raw_99,  p90 = ρ_raw_90),
            spearman_reg = (full = ρ_reg_full,  p99 = ρ_reg_99,  p90 = ρ_reg_90))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
