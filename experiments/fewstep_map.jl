using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "Flowfusion.jl"))
Pkg.develop(path=joinpath(@__DIR__, "..", "Jester.jl"))
Pkg.instantiate()

import Flowfusion
using Flowfusion: random_literal_cat
using Flux, RandomFeatureMaps, Optimisers, Plots, Statistics, Random, CannotWaitForTheseOptimisers
using Jester: jvp
using Zygote: @ignore_derivatives
using CUDA, cuDNN
using ProgressMeter
using Serialization

const DEV = CUDA.functional() ? Flux.gpu : Flux.cpu
const ARR = CUDA.functional() ? CUDA.CuArray : Array
@info "Device" cuda_functional=CUDA.functional()

# =====================================================================
# Stage 1: two-time deterministic few-step flow map on the cat toy
#
#   X_ψ(x, s, t) = x + (t-s) * h_ψ(x, s, t)         (§3, eq. 2 of consistent_energy.tex)
#   Interpolant (deterministic, γ_t = 0):
#       I_t = (1-t) x0 + t x1,   I_t˙ = x1 - x0
#
#   Losses trained jointly:
#     L_map = E‖ X_ψ(I_s, s, t) − I_t ‖²               (map-MSE target loss)
#     L_lag = E‖ ∂_t X_ψ(X_ψ(I_t, t, s), s, t) − I_t˙ ‖²   (eq. 3, Lagrangian)
#
#   (s,t) ∼ U[0,1]² are sampled independently — s can exceed t, so the map
#   is trained to step backwards as well as forwards.
# =====================================================================

const T = Float32
const SPACEDIM = 2
const BASE_LO = T(2)
const BASE_HI = T(3)
const CAT_SIGMA = T(0.05)

# ---------- architecture ----------
# Mirrors Flowfusion.jl/examples/continuous.jl FModel, but with two times (s, t)
# and the map parameterized so X_ψ(x, s, s) = x by construction.

struct TwoTimeMap{A}
    layers::A
end
Flux.@layer TwoTimeMap

function TwoTimeMap(; embeddim=128, spacedim=SPACEDIM, nlayers=3)
    embed_s     = Chain(RandomFourierFeatures(1 => embeddim, 1f0),        Dense(embeddim => embeddim, swish))
    embed_t     = Chain(RandomFourierFeatures(1 => embeddim, 1f0),        Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(spacedim => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    # Project concat( raw x ; raw s ; raw t ; embed_s ; embed_t ; embed_state ) -> embeddim
    project_in  = spacedim + 2 + 3 * embeddim
    project     = Dense(project_in => embeddim, swish)
    ffs    = [Dense(embeddim => embeddim, swish) for _ in 1:nlayers]
    decode = Dense(embeddim => spacedim)
    TwoTimeMap((; embed_s, embed_t, embed_state, project, ffs, decode))
end

# x : (spacedim, batch);  s, t : (1, batch) or Vector(batch)
function (m::TwoTimeMap)(x, s, t)
    l  = m.layers
    sv = reshape(s, 1, :)
    tv = reshape(t, 1, :)
    es = l.embed_s(sv)
    et = l.embed_t(tv)
    ex = l.embed_state(x)
    # Dense sees raw values alongside the (random-Fourier) embeddings.
    h  = l.project(vcat(x, sv, tv, es, et, ex))
    for ff in l.ffs
        h = h .+ ff(h)
    end
    v = l.decode(h)
    return x .+ (tv .- sv) .* v   # identity at s == t
end

# ---------- data ----------
# CPU samplers — training loop moves batches to device explicitly.
sampleX0_cpu(n) = rand(T, SPACEDIM, n) .+ BASE_LO              # base: uniform([2,3]^2)
sampleX1_cpu(n) = random_literal_cat(n; sigma = CAT_SIGMA)     # cat target

sampleX0(n) = sampleX0_cpu(n) |> DEV
sampleX1(n) = sampleX1_cpu(n) |> DEV

# ---------- loss ----------
# Returns (total, L_map, L_lag). Only `total` is differentiated.
function losses(m::TwoTimeMap, x0, x1, s, t; lag_weight = 0.2)
    sv = reshape(s, 1, :)
    tv = reshape(t, 1, :)
    Is = (T(1) .- sv) .* x0 .+ sv .* x1
    It = (T(1) .- tv) .* x0 .+ tv .* x1
    dI = x1 .- x0                                # ẋ is constant along a linear interpolant

    # Shared inner pullback  y := X_ψ(I_t, t, s)   (≈ I_s when the map is self-consistent)
    y = m(It, t, s)

    # Map-MSE term of FMM (Boffi+2024, eq. 3.17):  X_ψ(y, s, t) ≈ I_t
    Xrec  = m(y, s, t)
    L_map = mean(sum(abs2.(Xrec .- It); dims=1))

    # Lagrangian term of FMM:  ∂_t X_ψ(y, s, t) ≈ İ_t
    f_t(tval, model_) = model_(y, s, tval)
    one_vec = @ignore_derivatives ARR(ones(T, length(t)))
    _, ∂tX  = jvp(f_t, t, m, one_vec; ε = T(1e-3))
    L_lag   = mean(sum(abs2.(∂tX .- dI); dims=1))

    return L_map + lag_weight * L_lag, L_map, L_lag
end

# ---------- training ----------
"""
    sample_times(batch, window) -> (s, t)

t ~ U[0,1];  s | t ~ U[max(0, t-window), min(1, t+window)].  |s-t| ≤ window with sign
unconstrained — bidirectional stepping.
"""
function sample_times(batch::Int, window::T = T(0.125))
    tc = rand(T, batch)
    u  = rand(T, batch)
    lo = max.(T(0), tc .- window)
    hi = min.(T(1), tc .+ window)
    sc = lo .+ (hi .- lo) .* u
    return sc, tc
end

"""
Curriculum training: `n_epochs` each of `iters_per_epoch` steps (the final epoch
runs for `iters_per_epoch * last_epoch_multiplier` iters). Default schedule
progressively widens the time window from 0.125 to 1.0 in five equal steps.

LR schedule is **per-epoch**: at the start of every epoch `η` is reset to `eta_peak(epoch)`
(defaults to the constant base `eta`). The linear warmdown (from `η_peak` to
`η_peak * cooldown_final_factor` over the last `cooldown_frac` of the epoch) is applied
only to the final epoch.
"""
function train!(model;
                iters_per_epoch = 4000,
                time_window_schedule = Float32.(collect(range(0.125, 1.0, length=5))),
                batch = 4096, eta = 1e-3, seed = 0,
                eta_peak = nothing,        # Union{Nothing, Function, AbstractVector}
                cooldown_frac = 0.2,
                cooldown_final_factor = 0.01,
                last_epoch_multiplier::Int = 4)
    # Keep eta in its original numeric type so Optimisers.adjust! can write it back into
    # the Leaf (AdamW{typeof(eta),…}) without a Float64↔Float32 convert error.
    η_base  = float(eta)
    η_type  = typeof(η_base)
    n_epochs       = length(time_window_schedule)
    iters_schedule = fill(iters_per_epoch, n_epochs)
    iters_schedule[end] *= last_epoch_multiplier
    iters_total    = sum(iters_schedule)
    # Per-epoch peak LR. Default = constant base eta. Accepts a function `epoch -> η`
    # or an explicit per-epoch vector.
    peak_fn = if eta_peak === nothing
        epoch -> η_base
    elseif eta_peak isa AbstractVector
        epoch -> η_type(eta_peak[epoch])
    else
        epoch -> η_type(eta_peak(epoch))
    end
    final_iters             = iters_schedule[end]
    cooldown_start_in_epoch = final_iters - floor(Int, cooldown_frac * final_iters)
    cooldown_length         = final_iters - cooldown_start_in_epoch
    final_factor            = η_type(cooldown_final_factor)

    η_peak_epoch = peak_fn(1)
    η   = η_peak_epoch
    opt = Flux.setup(Muon(eta=η), model)
    tot_hist = Float32[]; map_hist = Float32[]; lag_hist = Float32[]
    eta_hist = Float32[]; win_hist = Float32[]
    prog = Progress(iters_total; desc="training", dt=0.5, showspeed=true)
    for epoch in 1:n_epochs
        if epoch > 1
            η_peak_epoch = peak_fn(epoch)
            η = η_peak_epoch
            Optimisers.adjust!(opt, η)
        end
        win        = T(time_window_schedule[epoch])
        this_iters = iters_schedule[epoch]
        for iter_in_epoch in 1:this_iters
            x0  = sampleX0(batch)
            x1  = sampleX1(batch)
            s_cpu, t_cpu = sample_times(batch, win)
            s   = s_cpu |> DEV
            t   = t_cpu |> DEV
            (vals, g) = Flux.withgradient(model) do m
                losses(m, x0, x1, s, t)
            end
            Flux.update!(opt, model, g[1])
            push!(tot_hist, Float32(vals[1]))
            push!(map_hist, Float32(vals[2]))
            push!(lag_hist, Float32(vals[3]))
            # Linear warmdown tail on the final epoch only.
            if epoch == n_epochs && iter_in_epoch > cooldown_start_in_epoch
                progress = η_type((iter_in_epoch - cooldown_start_in_epoch) / cooldown_length)
                η = η_peak_epoch * (one(η_type) - progress * (one(η_type) - final_factor))
                Optimisers.adjust!(opt, η)
            end
            push!(eta_hist, Float32(η))
            push!(win_hist, Float32(win))
            next!(prog; showvalues = [(:epoch,  epoch),
                                      (:window, Float32(win)),
                                      (:total,  round(vals[1]; digits=4)),
                                      (:map,    round(vals[2]; digits=4)),
                                      (:lag,    round(vals[3]; digits=4)),
                                      (:eta,    Float32(η))])
        end
    end
    return (; tot_hist, map_hist, lag_hist, eta_hist, win_hist,
            iters_schedule, n_epochs)
end

# ---------- sampling ----------
step_grid(nsteps::Int) = T.(collect(range(0; stop=1, length=nsteps+1)))

function time_vector(x, tval)
    tv = similar(x, size(x, 2))
    fill!(tv, eltype(x)(tval))
    return tv
end

# Few-step sampling from time 0 to time 1 over a grid of K steps.
function sample_fewstep(model, x0; nsteps::Int = 4)
    grid = step_grid(nsteps)
    x = x0
    for k in 1:nsteps
        s = time_vector(x, grid[k])
        t = time_vector(x, grid[k+1])
        x = model(x, s, t)
    end
    return x
end

sample_onestep(model, x0) = sample_fewstep(model, x0; nsteps = 1)

function base_logpdf(x)
    inside = (x[1, :] .>= BASE_LO) .& (x[1, :] .<= BASE_HI) .&
             (x[2, :] .>= BASE_LO) .& (x[2, :] .<= BASE_HI)
    logp = fill(-Inf, size(x, 2))
    logp[inside] .= 0.0
    return logp
end

function step_logabsdet(model, x, s, t; ε = T(1e-3))
    e1 = zero(x); e1[1, :] .= one(eltype(x))
    e2 = zero(x); e2[2, :] .= one(eltype(x))
    f_x(xval, model_) = model_(xval, s, t)
    _, col1 = jvp(f_x, x, model, e1; ε)
    _, col2 = jvp(f_x, x, model, e2; ε)
    detj = Float64.(col1[1, :] .* col2[2, :] .- col1[2, :] .* col2[1, :])
    return log.(abs.(detj))
end

function sample_fewstep_with_logpdf(model, x0; nsteps::Int = 4, ε = T(1e-3))
    grid = step_grid(nsteps)
    x = copy(x0)
    logp = base_logpdf(x0)
    for k in 1:nsteps
        s = time_vector(x, grid[k])
        t = time_vector(x, grid[k+1])
        logp .-= step_logabsdet(model, x, s, t; ε)
        x = model(x, s, t)
    end
    return x, logp
end

function literal_cat_logpdf(x; sigma = Float64(CAT_SIGMA), n_angles::Int = 4096)
    θs = range(0.0; stop = 2pi, length = n_angles + 1)[1:end-1]
    centers = Matrix{Float64}(undef, SPACEDIM, n_angles)
    for (i, θ) in enumerate(θs)
        centers[:, i] .= Flowfusion.cat_shape(θ) ./ 200
    end
    x64 = Float64.(x)
    x2 = reshape(sum(abs2, x64; dims=1), :, 1)
    c2 = reshape(sum(abs2, centers; dims=1), 1, :)
    sqdist = max.(x2 .+ c2 .- 2 .* (transpose(x64) * centers), 0.0)
    logconst = -log(2pi * sigma^2)
    logterms = @. logconst - sqdist / (2 * sigma^2)
    rowmax = maximum(logterms; dims=2)
    return vec(rowmax .+ log.(sum(exp.(logterms .- rowmax); dims=2) ./ n_angles))
end

# ---------- main ----------
function main(; iters_per_epoch = 6000,
              time_window_schedule = Float32.(collect(range(0.125, 1.0, length=5))),
              n_inf = 5000,
              n_like = 1024,
              like_steps = 4,
              like_angles = 4096,
              outdir = @__DIR__,
              model_path = joinpath(@__DIR__, "fewstep_map_model.bin"))
    if isfile(model_path)
        @info "loading saved model (delete file to retrain)" model_path
        loaded = deserialize(model_path)
        model  = loaded.model |> DEV
        hist   = loaded.hist
    else
        @info "no saved model at $(model_path) — training from scratch"
        model = TwoTimeMap(embeddim=256, nlayers=3) |> DEV
        hist  = train!(model; iters_per_epoch, time_window_schedule)
        serialize(model_path, (model = model |> Flux.cpu, hist = hist))
        @info "saved trained model" model_path
    end

    # --- samples for plotting (pull back to CPU for Plots) ---
    x0_inf    = sampleX0(n_inf)
    x1_true   = sampleX1(n_inf)
    one_samps = sample_onestep(model, x0_inf)
    few_samps = sample_fewstep(model, x0_inf; nsteps=4)

    x0c  = Array(x0_inf); x1c = Array(x1_true)
    ones_c = Array(one_samps); fews_c = Array(few_samps)

    # --- plot 1: one-step ---
    p1 = scatter(x1c[1,:], x1c[2,:], msw=0, ms=1, color="orange",
                 alpha=0.5, label="X1 (true)", size=(500,500),
                 title="one-step  X_ψ(x0, 0, 1)", legend=:topleft)
    scatter!(p1, x0c[1,:], x0c[2,:], msw=0, ms=1, color="blue",
             alpha=0.35, label="X0")
    scatter!(p1, ones_c[1,:], ones_c[2,:], msw=0, ms=1, color="green",
             alpha=0.5, label="X1 (generated, 1 step)")
    savefig(p1, joinpath(outdir, "fewstep_onestep.svg")); savefig(p1, joinpath(outdir, "fewstep_onestep.png"))

    # --- plot 2: few-step (4 steps) vs true ---
    p2 = scatter(x1c[1,:], x1c[2,:], msw=0, ms=1, color="orange",
                 alpha=0.5, label="X1 (true)", size=(500,500),
                 title="few-step  X_ψ iterated (4 steps)", legend=:topleft)
    scatter!(p2, x0c[1,:], x0c[2,:], msw=0, ms=1, color="blue",
             alpha=0.35, label="X0")
    scatter!(p2, fews_c[1,:], fews_c[2,:], msw=0, ms=1, color="green",
             alpha=0.5, label="X1 (generated, 4 steps)")
    savefig(p2, joinpath(outdir, "fewstep_multistep.svg")); savefig(p2, joinpath(outdir, "fewstep_multistep.png"))

    # --- plot 3: total / Lagrangian / map / eta, with epoch boundaries ---
    xs     = 1:length(hist.tot_hist)
    bounds = collect(cumsum(hist.iters_schedule))[1:end-1]
    add_bounds!(p) = isempty(bounds) ? p :
        vline!(p, bounds; ls=:dot, color=:gray, alpha=0.6, label=false)
    p_tot = add_bounds!(plot(xs, hist.tot_hist, yscale=:log10, xlabel="iter", ylabel="L_total",
                 title="total", label=false, lw=1))
    p_lag = add_bounds!(plot(xs, hist.lag_hist, yscale=:log10, xlabel="iter", ylabel="L_lag",
                 title="Lagrangian", label=false, lw=1))
    p_map = add_bounds!(plot(xs, hist.map_hist, yscale=:log10, xlabel="iter", ylabel="L_map",
                 title="map (MSE)", label=false, lw=1))
    p_eta = add_bounds!(plot(xs, hist.eta_hist, xlabel="iter", ylabel="η",
                 title="learning rate (per-epoch warmdown)", label=false, lw=1))
    p3 = plot(p_tot, p_lag, p_map, p_eta; layout=(1,4), size=(1600,350))
    savefig(p3, joinpath(outdir, "fewstep_losses.svg")); savefig(p3, joinpath(outdir, "fewstep_losses.png"))

    # --- plot 4: reference vs model log-likelihood on generated samples ---
    like_model = model |> Flux.cpu
    x0_like = sampleX0_cpu(n_like)
    like_samps, model_logp = sample_fewstep_with_logpdf(like_model, x0_like; nsteps=like_steps)
    ref_logp = literal_cat_logpdf(like_samps; sigma=Float64(CAT_SIGMA), n_angles=like_angles)
    keep = isfinite.(ref_logp) .& isfinite.(model_logp)
    ref_keep = ref_logp[keep]
    model_keep = model_logp[keep]

    # Absolute model/truth disagreement — used to drop the tail outliers.
    disagreement = abs.(ref_keep .- model_keep)
    q99 = quantile(disagreement, 0.99)
    q90 = quantile(disagreement, 0.90)
    mask99 = disagreement .<= q99
    mask90 = disagreement .<= q90

    function likelihood_scatter(ref, mdl, title_str)
        lo = min(minimum(ref), minimum(mdl))
        hi = max(maximum(ref), maximum(mdl))
        p = scatter(ref, mdl, msw=0, ms=2, alpha=0.55, color="black",
                    xlabel="reference log p(x)", ylabel="model log p(x)",
                    title=title_str, label="samples", legend=:topleft)
        plot!(p, [lo, hi], [lo, hi], color="red", lw=2, label="y = x")
        return p
    end

    p4a = likelihood_scatter(ref_keep[mask99], model_keep[mask99],
                             "99th-pct cut ($(like_steps) steps)")
    p4b = likelihood_scatter(ref_keep[mask90], model_keep[mask90],
                             "90th-pct cut ($(like_steps) steps)")
    p4c = likelihood_scatter(ref_keep, model_keep,
                             "full ($(like_steps) steps)")
    p4 = plot(p4a, p4b, p4c; layout=(1,3), size=(1800,600))
    savefig(p4, joinpath(outdir, "fewstep_likelihood_scatter.svg"))
    savefig(p4, joinpath(outdir, "fewstep_likelihood_scatter.png"))

    # --- plot 5: side-by-side histograms of true vs estimated log-likelihood ---
    p5a = histogram(ref_keep; bins=50, xlabel="reference log p(x)", ylabel="count",
                    title="true log-likelihood", label=false, color=:steelblue)
    p5b = histogram(model_keep; bins=50, xlabel="model log p(x)", ylabel="count",
                    title="estimated log-likelihood", label=false, color=:seagreen)
    p5 = plot(p5a, p5b; layout=(1,2), size=(1200,500))
    savefig(p5, joinpath(outdir, "fewstep_likelihood_hist.svg"))
    savefig(p5, joinpath(outdir, "fewstep_likelihood_hist.png"))

    return (; model, hist)
end

# Run when executed as a script:
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
