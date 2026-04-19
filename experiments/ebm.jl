using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "Flowfusion.jl"))
Pkg.develop(path=joinpath(@__DIR__, "..", "Jester.jl"))
Pkg.instantiate()

import Flowfusion
using Flowfusion: random_literal_cat
using Flux, RandomFeatureMaps, Optimisers, Plots, Statistics, Random
using Jester: grad_fd
using CUDA, cuDNN
using ProgressMeter
using Serialization

const DEV = CUDA.functional() ? Flux.gpu : Flux.cpu
const ARR = CUDA.functional() ? CUDA.CuArray : Array
@info "Device" cuda_functional=CUDA.functional()

# =====================================================================
#   A simple energy-based model trained by denoising score matching.
#
#   Variance-exploding schedule:  σ(t) = σ_min · (σ_max/σ_min)^t,  t ∈ [0,1]
#   Noised sample:  x_t = x1 + σ(t) · ε,   ε ~ N(0, I)
#   Conditional score:   ∇_{x_t} log p(x_t | x1) = -ε / σ(t)
#   Model score:         s_θ(x, t) = -∇_x E_θ(x, t)
#   σ²-weighted loss (NCSN/EDM):
#       L = E ‖ ε − σ(t) · ∇_x E_θ(x_t, t) ‖²
#
#   The gradient ∇_x E_θ is produced by `Jester.grad_fd`, whose custom
#   rrule propagates ∂L/∂θ through the inner differentiation via a
#   finite-difference probe — standard Flux.withgradient then works.
#
#   Sampling: annealed Langevin dynamics over the σ schedule.
#   Energy diagnostic: compare E_θ(x, t≈0) to -log p_cat(x) (the model
#   only identifies energy up to an additive constant — we fit that
#   offset before plotting).
# =====================================================================

const T = Float32
const SPACEDIM = 2
const CAT_SIGMA = T(0.05)
const SIGMA_MIN = T(0.01)
const SIGMA_MAX = T(2.0)

sigma_at(t) = SIGMA_MIN .* (SIGMA_MAX / SIGMA_MIN) .^ t

# ---------- data ----------
sampleX1_cpu(n) = random_literal_cat(n; sigma = CAT_SIGMA)
sampleX1(n)     = sampleX1_cpu(n) |> DEV

# ---------- architecture ----------
# x : (SPACEDIM, batch), t : (batch,)  →  scalar energy per sample, (batch,)
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
    # Tuple (not Vector) so Zygote's gradient for this field is a Tuple of
    # NamedTuples — Jester.grad_fd's custom rrule recurses through Tuples and
    # NamedTuples but not Vectors of NamedTuples.
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

# ---------- loss ----------
# `grad_fd(f, x, m)` returns ∇_x sum(E_θ(x, t))  — because the per-sample
# energies decouple across the batch, this is exactly the stacked per-sample
# score. The outer Zygote hits grad_fd's rrule (FD in parameter space).
function score_loss(model, x1, t, ε_noise; fd_eps = T(1e-3))
    σv  = reshape(sigma_at(t), 1, :)
    x_t = x1 .+ σv .* ε_noise
    f(x_, m_) = sum(m_(x_, t))
    gradx = grad_fd(f, x_t, model; ε = fd_eps)
    mean(sum(abs2.(ε_noise .- σv .* gradx); dims = 1))
end

# ---------- training ----------
function train!(model;
                iters = 20000, batch = 4096, eta = 1f-3,
                fd_eps = T(1e-3),
                cooldown_frac = 0.2,
                cooldown_final_factor = 0.01,
                seed = 0)
    Random.seed!(seed)
    η_type = typeof(float(eta))
    η_base = η_type(eta)
    η      = η_base
    opt    = Flux.setup(AdamW(eta = η), model)
    loss_hist = Float32[]; eta_hist = Float32[]
    cooldown_start  = iters - floor(Int, cooldown_frac * iters)
    cooldown_length = max(iters - cooldown_start, 1)
    final_factor    = η_type(cooldown_final_factor)
    prog = Progress(iters; desc = "training EBM", dt = 0.5, showspeed = true)
    for iter in 1:iters
        x1    = sampleX1(batch)
        t     = rand(T, batch) |> DEV
        ε_nse = randn!(similar(x1))
        val, g = Flux.withgradient(model) do m
            score_loss(m, x1, t, ε_nse; fd_eps = fd_eps)
        end
        Flux.update!(opt, model, g[1])
        push!(loss_hist, Float32(val))
        if iter > cooldown_start
            progress = η_type((iter - cooldown_start) / cooldown_length)
            η = η_base * (one(η_type) - progress * (one(η_type) - final_factor))
            Optimisers.adjust!(opt, η)
        end
        push!(eta_hist, Float32(η))
        next!(prog; showvalues = [(:iter, iter),
                                  (:loss, round(val; digits = 4)),
                                  (:eta,  Float32(η))])
    end
    return (; loss_hist, eta_hist)
end

# ---------- sampling ----------
# Annealed Langevin dynamics (NCSN-style).  For each noise level σᵢ we run
# `n_steps_per_sigma` Langevin steps with step size αᵢ = eps · (σᵢ/σ_min)².
function annealed_langevin(model, x0;
                           n_sigmas = 20,
                           n_steps_per_sigma = 80,
                           eps = T(2f-5),
                           fd_eps = T(1e-3))
    sigmas = T.(SIGMA_MAX .* (SIGMA_MIN / SIGMA_MAX) .^ range(0, 1, length = n_sigmas))
    x = copy(x0)
    for σi in sigmas
        t_val = T(log(σi / SIGMA_MIN) / log(SIGMA_MAX / SIGMA_MIN))
        tvec = similar(x, size(x, 2)); fill!(tvec, t_val)
        α    = eps * (σi / SIGMA_MIN)^2
        for _ in 1:n_steps_per_sigma
            f_(x_, m_) = sum(m_(x_, tvec))
            g     = grad_fd(f_, x, model; ε = fd_eps)   # ∇_x E_θ
            score = -g
            noise = randn!(similar(x))
            x = x .+ (α / T(2)) .* score .+ sqrt(α) .* noise
        end
    end
    return x
end

# Init: tight Gaussian up-and-right of the cat so the starting blob doesn't
# overlap the target in the samples plot.
const INIT_CENTER = T.([5.0, 5.0])
const INIT_STD    = T(0.5)
langevin_init(n) = INIT_CENTER .+ INIT_STD .* randn(T, SPACEDIM, n)

# ---------- reference cat log-density (matches fewstep_map.jl) ----------
function literal_cat_logpdf(x; sigma = Float64(CAT_SIGMA), n_angles::Int = 4096)
    θs = range(0.0; stop = 2π, length = n_angles + 1)[1:end-1]
    centers = Matrix{Float64}(undef, SPACEDIM, n_angles)
    for (i, θ) in enumerate(θs)
        centers[:, i] .= Flowfusion.cat_shape(θ) ./ 200
    end
    x64 = Float64.(x)
    x2 = reshape(sum(abs2, x64; dims = 1), :, 1)
    c2 = reshape(sum(abs2, centers; dims = 1), 1, :)
    sqdist = max.(x2 .+ c2 .- 2 .* (transpose(x64) * centers), 0.0)
    logconst = -log(2π * sigma^2)
    logterms = @. logconst - sqdist / (2 * sigma^2)
    rowmax = maximum(logterms; dims = 2)
    return vec(rowmax .+ log.(sum(exp.(logterms .- rowmax); dims = 2) ./ n_angles))
end

# ---------- main ----------
function main(; iters = 20000,
              batch = 4096,
              n_inf = 5000,
              n_scatter = 2000,
              outdir = @__DIR__,
              model_path = joinpath(@__DIR__, "ebm_model.bin"))
    if isfile(model_path)
        @info "loading saved model (delete file to retrain)" model_path
        loaded = deserialize(model_path)
        model  = loaded.model |> DEV
        hist   = loaded.hist
    else
        @info "no saved model at $(model_path) — training from scratch"
        model = EBM(embeddim = 256, nlayers = 3) |> DEV
        hist  = train!(model; iters = iters, batch = batch)
        serialize(model_path, (model = model |> Flux.cpu, hist = hist))
        @info "saved trained model" model_path
    end

    # --- loss curve ---
    xs = 1:length(hist.loss_hist)
    p_loss = plot(xs, hist.loss_hist; yscale = :log10, xlabel = "iter",
                  ylabel = "score-matching loss", label = false, lw = 1,
                  title = "EBM training loss")
    p_eta  = plot(xs, hist.eta_hist; xlabel = "iter", ylabel = "η",
                  label = false, lw = 1, title = "learning rate")
    p_tr   = plot(p_loss, p_eta; layout = (1, 2), size = (1200, 400))
    savefig(p_tr, joinpath(outdir, "ebm_loss.svg"))
    savefig(p_tr, joinpath(outdir, "ebm_loss.png"))

    # --- representative samples via annealed Langevin ---
    x0_init  = langevin_init(n_inf) |> DEV
    samples  = annealed_langevin(model, x0_init)
    x1_true  = sampleX1(n_inf)
    samp_c   = Array(samples); true_c = Array(x1_true); init_c = Array(x0_init)

    p_samp = scatter(true_c[1, :], true_c[2, :]; msw = 0, ms = 1, color = "orange",
                     alpha = 0.5, label = "X1 (true)", size = (500, 500),
                     title = "EBM annealed Langevin samples", legend = :topleft)
    scatter!(p_samp, init_c[1, :], init_c[2, :]; msw = 0, ms = 1, color = "blue",
             alpha = 0.2, label = "Langevin init")
    scatter!(p_samp, samp_c[1, :], samp_c[2, :]; msw = 0, ms = 1, color = "green",
             alpha = 0.5, label = "X1 (EBM)")
    savefig(p_samp, joinpath(outdir, "ebm_samples.svg"))
    savefig(p_samp, joinpath(outdir, "ebm_samples.png"))

    # --- estimated energy vs. true energy ---
    # Compare E_θ(x, t≈0) against -log p_cat(x) at (a) clean cat samples,
    # (b) model samples. The constant offset is fit so that the clean-sample
    # means coincide.
    eval_cpu  = model |> Flux.cpu
    x_data    = sampleX1_cpu(n_scatter)
    x_model   = Array(annealed_langevin(model, langevin_init(n_scatter) |> DEV))
    t0        = zeros(T, n_scatter)

    E_data    = Array(eval_cpu(x_data,  t0))
    E_model   = Array(eval_cpu(x_model, t0))
    true_data  = -literal_cat_logpdf(x_data;  sigma = Float64(CAT_SIGMA))
    true_model = -literal_cat_logpdf(x_model; sigma = Float64(CAT_SIGMA))

    keep_d = isfinite.(true_data)  .& isfinite.(E_data)
    keep_m = isfinite.(true_model) .& isfinite.(E_model)

    # Fit the additive constant using the clean-data scatter only (well-defined
    # support; model samples can stray into low-density regions where -log p
    # balloons).
    c = mean(Float64.(true_data[keep_d]) .- Float64.(E_data[keep_d]))
    est_d = Float64.(E_data[keep_d])  .+ c
    est_m = Float64.(E_model[keep_m]) .+ c
    tru_d = Float64.(true_data[keep_d])
    tru_m = Float64.(true_model[keep_m])

    # Tail cuts mirror the likelihood scatter in fewstep_map.jl.
    function tail_masks(true_e, est_e)
        diff = abs.(true_e .- est_e)
        (diff .<= quantile(diff, 0.99), diff .<= quantile(diff, 0.90))
    end

    function scatter_energy(true_e, est_e, title_str)
        lo = min(minimum(true_e), minimum(est_e))
        hi = max(maximum(true_e), maximum(est_e))
        p = scatter(true_e, est_e; msw = 0, ms = 2, alpha = 0.55, color = "black",
                    xlabel = "true energy  -log p(x)",
                    ylabel = "model energy  E_θ(x, 0) + c",
                    title = title_str, label = "samples", legend = :topleft)
        plot!(p, [lo, hi], [lo, hi]; color = "red", lw = 2, label = "y = x")
        return p
    end

    m99_d, m90_d = tail_masks(tru_d, est_d)
    m99_m, m90_m = tail_masks(tru_m, est_m)

    p_ed_full = scatter_energy(tru_d,        est_d,        "cat samples (full)")
    p_ed_99   = scatter_energy(tru_d[m99_d], est_d[m99_d], "cat samples (99th-pct cut)")
    p_ed_90   = scatter_energy(tru_d[m90_d], est_d[m90_d], "cat samples (90th-pct cut)")
    p_em_full = scatter_energy(tru_m,        est_m,        "model samples (full)")
    p_em_99   = scatter_energy(tru_m[m99_m], est_m[m99_m], "model samples (99th-pct cut)")
    p_em_90   = scatter_energy(tru_m[m90_m], est_m[m90_m], "model samples (90th-pct cut)")
    p_e = plot(p_ed_full, p_ed_99, p_ed_90,
               p_em_full, p_em_99, p_em_90;
               layout = (2, 3), size = (1800, 1100))
    savefig(p_e, joinpath(outdir, "ebm_energy_scatter.svg"))
    savefig(p_e, joinpath(outdir, "ebm_energy_scatter.png"))

    return (; model, hist)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
