using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "Flowfusion.jl"))
Pkg.develop(path=joinpath(@__DIR__, "..", "Jester.jl"))
Pkg.instantiate()

using ForwardBackward, Flowfusion
using Flowfusion: random_literal_cat, cat_shape

# -----------------------------------------------------------------
# GPU-friendly override. Upstream `endpoint_conditioned_sample` uses
# `randn(eltype(μ), size(μ))` (CPU), which breaks when μ is a CuArray.
# `randn!(similar(μ))` is device-preserving and works for both.
# -----------------------------------------------------------------
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
using Flux, RandomFeatureMaps, Optimisers, Plots, Statistics, Random
using Jester: grad_fd
using CUDA, cuDNN
using ProgressMeter
using Serialization

const DEV = CUDA.functional() ? Flux.gpu : Flux.cpu
const ARR = CUDA.functional() ? CUDA.CuArray : Array
@info "Device" cuda_functional=CUDA.functional()

# =====================================================================
#   Flowfusion-style EBM.
#
#   The Flowfusion continuous.jl recipe:
#       X₀ ~ U[2,3]² (prior)     X₁ ~ cat (target)
#       Xₜ = bridge(P, X₀, X₁, t)    with P = BrownianMotion(σ_bm)
#   trains a predictor
#       X̂₁_θ(t, Xₜ)          floss = MSE(X̂₁_θ, X₁)
#   and samples via
#       gen(P, X₀, model, steps).
#
#   Here we parameterise X̂₁ by an energy head:
#       X̂₁_θ(t, Xₜ) = Xₜ + (−∇_x E_θ(Xₜ, t)) · (1.05 − t)
#   mirroring the (1.05 − t) scaling in continuous.jl. The gradient is
#   produced by `Jester.grad_fd`, whose custom rrule propagates ∂L/∂θ
#   through the inner differentiation.
#
#   Training runs on the Flowfusion bridge (an explicit X₀ → X₁ transport
#   with process-driven noise); sampling runs the process from the real
#   prior X₀ via `gen` — no MCMC, no arbitrary chain init.
# =====================================================================

const T = Float32
const SPACEDIM = 2
const BASE_LO = T(2)
const BASE_HI = T(3)
const CAT_SIGMA = T(0.05)
const BM_SIGMA = T(0.15)     # matches Flowfusion continuous.jl

# ---------- data ----------
sampleX0_cpu(n) = rand(T, SPACEDIM, n) .+ BASE_LO
sampleX1_cpu(n) = random_literal_cat(n; sigma = CAT_SIGMA)
sampleX0(n)     = sampleX0_cpu(n) |> DEV
sampleX1(n)     = sampleX1_cpu(n) |> DEV

# ---------- reference cat log-density ----------
function literal_cat_logpdf(x; sigma = Float64(CAT_SIGMA), n_angles::Int = 4096)
    θs = range(0.0; stop = 2π, length = n_angles + 1)[1:end-1]
    centers = Matrix{Float64}(undef, SPACEDIM, n_angles)
    for (i, θ) in enumerate(θs)
        centers[:, i] .= cat_shape(θ) ./ 200
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

# =====================================================================
#  EBM scalar energy head
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
    # Tuple: Jester.grad_fd's rrule recurses through Tuple/NamedTuple tangents.
    ffs         = Tuple(Dense(embeddim => embeddim, swish) for _ in 1:nlayers)
    decode      = Dense(embeddim => 1)
    EBM((; embed_t, embed_state, project, ffs, decode))
end

# x : (SPACEDIM, batch), t : (batch,) — returns a scalar energy per sample.
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
#  Flowfusion wrapper:  model(t, Xt) → X̂₁ (ContinuousState)
# =====================================================================
struct EBMFlowWrapper{M}
    ebm::M
end
Flux.@layer EBMFlowWrapper

const FD_EPS = T(1e-3)

function (w::EBMFlowWrapper)(t, Xt)
    x     = tensor(Xt)
    batch = size(x, 2)
    # `t` is a Vector during training (batch-sized) and a scalar during gen.
    tv = if t isa AbstractArray
        reshape(t, :)
    else
        buf = similar(x, batch); fill!(buf, T(t)); buf
    end
    f(x_, m_) = sum(m_(x_, tv))
    score  = -grad_fd(f, x, w.ebm; ε = FD_EPS)              # −∇_x E_θ
    scale  = reshape(T(1.05) .- tv, 1, :)
    x_hat  = x .+ score .* scale
    return ContinuousState(x_hat)
end

# =====================================================================
#  Training — Flowfusion bridge + floss, EBM wrapped as X̂₁ predictor
# =====================================================================
function train!(model::EBMFlowWrapper;
                iters = 20000, batch = 4096, eta = 1f-3,
                cooldown_frac = 0.2,
                cooldown_final_factor = 0.01,
                seed = 0)
    Random.seed!(seed)
    P      = BrownianMotion(BM_SIGMA)
    η_type = typeof(float(eta))
    η_base = η_type(eta)
    η      = η_base
    opt    = Flux.setup(AdamW(eta = η), model)
    loss_hist = Float32[]; eta_hist = Float32[]
    cooldown_start  = iters - floor(Int, cooldown_frac * iters)
    cooldown_length = max(iters - cooldown_start, 1)
    final_factor    = η_type(cooldown_final_factor)
    prog = Progress(iters; desc = "training Flowfusion EBM", dt = 0.5, showspeed = true)
    for iter in 1:iters
        X0 = ContinuousState(sampleX0(batch))
        X1 = ContinuousState(sampleX1(batch))
        t  = rand(T, batch) |> DEV
        Xt = bridge(P, X0, X1, t)
        val, g = Flux.withgradient(model) do m
            floss(P, m(t, Xt), X1, scalefloss(P, t))
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

# =====================================================================
#  Main
# =====================================================================
function main(; iters = 20000, batch = 4096,
              n_inf      = 5000,
              n_scatter  = 2000,
              gen_steps  = T.(0:0.005:1),
              outdir     = @__DIR__,
              model_path = joinpath(@__DIR__, "flowfusion_ebm_model.bin"))
    local model, hist
    if isfile(model_path)
        @info "loading saved Flowfusion EBM (delete to retrain)" model_path
        loaded = deserialize(model_path)
        model  = loaded.model |> DEV
        hist   = loaded.hist
    else
        @info "training Flowfusion EBM from scratch" iters
        ebm   = EBM(embeddim = 256, nlayers = 3) |> DEV
        model = EBMFlowWrapper(ebm)
        hist  = train!(model; iters = iters, batch = batch)
        serialize(model_path, (model = model |> Flux.cpu, hist = hist))
        @info "saved model" model_path
    end

    # ─── sampling via Flowfusion's gen (transport from the real prior) ───
    P        = BrownianMotion(BM_SIGMA)
    X0_inf   = ContinuousState(sampleX0(n_inf))
    samples  = gen(P, X0_inf, model, gen_steps)
    x0c      = Array(tensor(X0_inf))
    sampc    = Array(tensor(samples))
    x1_true  = Array(sampleX1(n_inf))

    p_samp = scatter(x1_true[1, :], x1_true[2, :]; msw = 0, ms = 1, color = "orange",
                     alpha = 0.5, label = "X1 (true)", size = (500, 500),
                     title = "Flowfusion EBM samples", legend = :topleft)
    scatter!(p_samp, x0c[1, :], x0c[2, :]; msw = 0, ms = 1, color = "blue",
             alpha = 0.3, label = "X0 (prior)")
    scatter!(p_samp, sampc[1, :], sampc[2, :]; msw = 0, ms = 1, color = "green",
             alpha = 0.5, label = "X1 (EBM)")
    savefig(p_samp, joinpath(outdir, "flowfusion_ebm_samples.svg"))
    savefig(p_samp, joinpath(outdir, "flowfusion_ebm_samples.png"))

    # ─── loss curve ───
    xs  = 1:length(hist.loss_hist)
    p_l = plot(xs, hist.loss_hist; yscale = :log10, xlabel = "iter", ylabel = "loss",
               title = "training loss", label = false, lw = 1)
    p_e = plot(xs, hist.eta_hist;  xlabel = "iter", ylabel = "η",
               title = "learning rate", label = false, lw = 1)
    p_tr = plot(p_l, p_e; layout = (1, 2), size = (1200, 400))
    savefig(p_tr, joinpath(outdir, "flowfusion_ebm_loss.svg"))
    savefig(p_tr, joinpath(outdir, "flowfusion_ebm_loss.png"))

    # ─── energy-vs-true-log-p scatter (analogue of ebm.jl's plot) ───
    model_cpu = model |> Flux.cpu
    ebm_cpu   = model_cpu.ebm
    x_data    = sampleX1_cpu(n_scatter)
    # Take first n_scatter generated samples (or fewer if we generated less).
    n_g       = min(n_scatter, size(sampc, 2))
    x_gen     = sampc[:, 1:n_g]
    t0_data   = zeros(T, n_scatter)
    t0_gen    = zeros(T, n_g)
    E_data    = Float64.(Array(ebm_cpu(x_data, t0_data)))
    E_gen     = Float64.(Array(ebm_cpu(x_gen,  t0_gen)))
    true_data = -literal_cat_logpdf(x_data;  sigma = Float64(CAT_SIGMA))
    true_gen  = -literal_cat_logpdf(x_gen;   sigma = Float64(CAT_SIGMA))

    keep_d = isfinite.(true_data) .& isfinite.(E_data)
    keep_g = isfinite.(true_gen)  .& isfinite.(E_gen)
    # Fit the additive constant on clean-data points only.
    c_shift = mean(Float64.(true_data[keep_d]) .- Float64.(E_data[keep_d]))
    est_d   = Float64.(E_data[keep_d]) .+ c_shift
    est_g   = Float64.(E_gen[keep_g])  .+ c_shift
    tru_d   = Float64.(true_data[keep_d])
    tru_g   = Float64.(true_gen[keep_g])

    tail_masks(tru, est) = begin
        d = abs.(tru .- est)
        (d .<= quantile(d, 0.99), d .<= quantile(d, 0.90))
    end
    function scatter_energy(tru, est, title_str)
        lo = min(minimum(tru), minimum(est))
        hi = max(maximum(tru), maximum(est))
        p = scatter(tru, est; msw = 0, ms = 2, alpha = 0.55, color = "black",
                    xlabel = "true energy  -log p(x)",
                    ylabel = "model energy  E_θ(x, 0) + c",
                    title = title_str, label = "samples", legend = :topleft)
        plot!(p, [lo, hi], [lo, hi]; color = "red", lw = 2, label = "y = x")
        return p
    end

    m99_d, m90_d = tail_masks(tru_d, est_d)
    m99_g, m90_g = tail_masks(tru_g, est_g)
    p_ed_full = scatter_energy(tru_d,        est_d,        "cat samples (full)")
    p_ed_99   = scatter_energy(tru_d[m99_d], est_d[m99_d], "cat samples (99%-cut)")
    p_ed_90   = scatter_energy(tru_d[m90_d], est_d[m90_d], "cat samples (90%-cut)")
    p_eg_full = scatter_energy(tru_g,        est_g,        "generated (full)")
    p_eg_99   = scatter_energy(tru_g[m99_g], est_g[m99_g], "generated (99%-cut)")
    p_eg_90   = scatter_energy(tru_g[m90_g], est_g[m90_g], "generated (90%-cut)")
    p_e = plot(p_ed_full, p_ed_99, p_ed_90,
               p_eg_full, p_eg_99, p_eg_90;
               layout = (2, 3), size = (1800, 1100))
    savefig(p_e, joinpath(outdir, "flowfusion_ebm_energy_scatter.svg"))
    savefig(p_e, joinpath(outdir, "flowfusion_ebm_energy_scatter.png"))

    return (; model, hist)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
