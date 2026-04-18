using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "Flowfusion.jl"))
Pkg.develop(path=joinpath(@__DIR__, "..", "Jester.jl"))
Pkg.instantiate()

using Flowfusion: random_literal_cat
using Flux, RandomFeatureMaps, Optimisers, Plots, Statistics, Random
using Jester: jvp
using Zygote: @ignore_derivatives
using CUDA, cuDNN
using ProgressMeter

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
sampleX0_cpu(n) = rand(T, SPACEDIM, n) .+ T(2)                 # base: uniform([2,3]^2)
sampleX1_cpu(n) = random_literal_cat(n; sigma = T(0.05))       # cat target

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
Curriculum training: `n_epochs` each of `iters_per_epoch` steps; the time-sampling
window is scheduled across epochs from a tight band up to the uniform regime
(window ≥ 1 ⇒ s ∼ U[0,1] given t ∼ U[0,1]).

LR schedule is **per-epoch**: at the start of every epoch `η` is reset to `eta_peak(epoch)`
(defaults to the constant base `eta`), then during the last `cooldown_frac` of that epoch
it is multiplied by `cooldown_rate` every `cooldown_every` steps via `Optimisers.adjust!`.
"""
function train!(model;
                iters_per_epoch = 8000,
                time_window_schedule = Float32.(range(0.125, 1.0; length = 5)),
                batch = 2048, eta = 1e-3, seed = 0,
                eta_peak = nothing,        # Union{Nothing, Function, AbstractVector}
                cooldown_frac = 0.5, cooldown_rate = 0.975, cooldown_every = 10)
    # Keep eta in its original numeric type so Optimisers.adjust! can write it back into
    # the Leaf (AdamW{typeof(eta),…}) without a Float64↔Float32 convert error.
    η_base  = float(eta)
    η_type  = typeof(η_base)
    rate    = η_type(cooldown_rate)
    n_epochs    = length(time_window_schedule)
    iters_total = iters_per_epoch * n_epochs
    # Per-epoch peak LR. Default = constant base eta. Accepts a function `epoch -> η`
    # or an explicit per-epoch vector.
    peak_fn = if eta_peak === nothing
        epoch -> η_base
    elseif eta_peak isa AbstractVector
        epoch -> η_type(eta_peak[epoch])
    else
        epoch -> η_type(eta_peak(epoch))
    end
    cooldown_start_in_epoch = iters_per_epoch - floor(Int, cooldown_frac * iters_per_epoch)

    η   = peak_fn(1)
    opt = Flux.setup(AdamW(eta=η), model)
    tot_hist = Float32[]; map_hist = Float32[]; lag_hist = Float32[]
    eta_hist = Float32[]; win_hist = Float32[]
    prog = Progress(iters_total; desc="training", dt=0.5, showspeed=true)
    for i in 1:iters_total
        iter_in_epoch = (i - 1) % iters_per_epoch + 1   # 1..iters_per_epoch
        epoch         = (i - 1) ÷ iters_per_epoch + 1
        # Start-of-epoch LR reset (no reset at i=1 since we already set η above).
        if iter_in_epoch == 1 && epoch > 1
            η = peak_fn(epoch)
            Optimisers.adjust!(opt, η)
        end
        win = T(time_window_schedule[epoch])
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
        # Within-epoch cooldown tail.
        if iter_in_epoch > cooldown_start_in_epoch && iter_in_epoch % cooldown_every == 0
            η *= rate
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
    return (; tot_hist, map_hist, lag_hist, eta_hist, win_hist,
            iters_per_epoch, n_epochs)
end

# ---------- sampling ----------
# Few-step sampling from time 0 to time 1 over a grid of K steps.
function sample_fewstep(model, x0; nsteps::Int = 4)
    grid = T.(collect(range(0; stop=1, length=nsteps+1)))   # length nsteps+1
    x = x0
    for k in 1:nsteps
        s = ARR(fill(grid[k],   size(x, 2)))
        t = ARR(fill(grid[k+1], size(x, 2)))
        x = model(x, s, t)
    end
    return x
end

sample_onestep(model, x0) = sample_fewstep(model, x0; nsteps = 1)

# ---------- main ----------
function main(; iters_per_epoch = 8000,
              time_window_schedule = Float32.(range(0.125, 1.0; length = 5)),
              n_inf = 5000, outdir = @__DIR__)
    model = TwoTimeMap(embeddim=128, nlayers=3) |> DEV
    hist  = train!(model; iters_per_epoch, time_window_schedule)

    # --- samples for plotting (pull back to CPU for Plots) ---
    x0_inf    = sampleX0(n_inf)""
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
    bounds = [hist.iters_per_epoch * k for k in 1:(hist.n_epochs - 1)]
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

    return (; model, hist)
end

# Run when executed as a script:
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
