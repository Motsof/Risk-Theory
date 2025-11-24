### Risk Theory final project


## Packages

using Distributions, Plots, StatsBase, GLM, DataFrames
include("02probestim.jl")


# ==============================================================================
#  S = Y(1) + ⋯ + Y(N) 
#  where the frequency is a counting random variable N ∈ {0,1,…}
#  and the random variables {Y(1), Y(2), ...} represent weekly 
#  claim severities, but conditionally independendient given {N = n}
#  with conditional distribution:
#  Y | N = n ~ LogNormal(μ(n), σ(n)) where:
#  μ(n) = g1(n) + ε1  with ε1 ~ Normal(0, ω1)
#  σ(n) = g2(n) + ε2  with ε2 ~ Normal(0, ω2)
# ==============================================================================

## Read and process data

countlines("proyecto\\siniestros.csv")

begin
    f = open("proyecto\\siniestros.csv")
    contenido = readlines(f)
    close(f)
    contenido
end

begin
    m = length(contenido)
    sev = fill(Float64[], m) # [Y1, Y2, YN]
    frec = zeros(Int, m) # N
    reclam = zeros(m) # S
    for i ∈ 1:m
        v = parse.(Float64, split(contenido[i], ','))
        if v == [0.0]
            sev[i] = []
        else
            sev[i] = v
            frec[i] = length(v)
            reclam[i] = sum(v)
        end
    end
end

##Frequency Model

EN = mean(frec)
VarN = var(frec, corrected = false)
println("Since the variance is greater than the mean, we fit a Negative Binomial distribution for frequency.")
p = EN / VarN
r = EN*p / (1 - p)
NB = NegativeBinomial(r, p)
println("Parameters:")
println("r = ", round(r,;digits=3))
println("p = ", round(p,;digits=3))

# Fitting the frequency distribution
Npos = masaprob(frec[frec .> 0]) # conditional pmf N|N>0
PN0 = sum(frec .== 0) / length(frec) # P(N=0)
pmfN(n::Integer) = Npos.fmp(n)*(1-PN0) + PN0*(n==0)

begin
    nn = collect(0:maximum(frec))
    pnn = pmfN.(nn)
    bar(nn, pnn, xlabel = "Frequency N", ylabel = "P(N = n)", label = "Empirical")
    scatter!(nn, pdf(NB, nn), label = "Negative Binomial", ms = 2)
end


## Exploring the data
  
# Severity
begin
    all_freqs = Int[]
    all_sevs = Float64[]
    for i in 1:length(frec)
        if frec[i] > 0
            append!(all_freqs, fill(frec[i], frec[i]))
            append!(all_sevs, sev[i])
        end
    end
    c = round(corspearman(all_freqs, all_sevs), digits = 4)

    scatter(all_freqs, all_sevs, legend = false, title = "Severity of claims\\n (Spearman correlation: $c)",color = :gray)
    xlabel!("Number of claims per period")
    ylabel!("Claim amount")
end

# Log-Severity
begin
    scatter(all_freqs, log.(all_sevs), legend = false, title = "Log-severity of claims\\n (Spearman correlation: $c)",color = :gray)
    xlabel!("Number of claims per period")
    ylabel!("Log-severity")
    unique_freqs = sort(unique(all_freqs))
    grupos = [log.(all_sevs[all_freqs .== k]) for k in unique_freqs]

    medias = mean.(grupos)
    desvst = std.(grupos, corrected = false)

    plot!(unique_freqs, medias, color=:red, lw=3)
    plot!(unique_freqs, medias .+ 2 .* desvst, color=:blue, lw=3)
    plot!(unique_freqs, medias .- 2 .* desvst, color=:blue, lw=3)
end

# Mean and Std Dev of log-severity vs frequency
begin
    plot(unique_freqs, medias, color = :red, lw = 3, legend = false)
    title!("Mean log-severity conditional on frequency")
    xaxis!("frequency")
    p1 = yaxis!("mean")
    # savefig("05logmeaunique_freqs[valid_idx]everity.png")
    plot(unique_freqs, desvst, color = :blue, lw = 3, legend = false)
    title!("Standard deviation of conditional log-severity")
    xaxis!("frequency")
    p2 = yaxis!("standard deviation")
    # savefig("06logsevstddev.png")
    plot(p1, p2, layout = (2,1), size = (600,600))
end

## Dependence modeles for severity parameters on frequency
# Create a DataFrame to facilitate modeling with GLM
valid_idx = length.(grupos) .> 1

df_summary = DataFrame(n = unique_freqs[valid_idx], mean_log = mean.(grupos[valid_idx]), std_log = std.(grupos[valid_idx],corrected = false))

# Fitting model for μ(n)
# We adjust a linear model for the mean of log-severity vs frequency
#g1(n) = α0 + α1 * n
model_g1 = lm(@formula(mean_log ~ n), df_summary)
alpha0 = coef(model_g1)[1]
alpha1 = coef(model_g1)[2]

# Omega_1 is defined as the MSE of g1(n)
res_g1 = residuals(model_g1)
omega_1 = sqrt(mean(res_g1.^2))
e_1 = Normal(0, omega_1)

println("\n--- Model μ(n) = g1(n) + ε1 with ε1 ~ N(0, omega_1) ---")
println("Estimated equation: μ(n) = $(round(alpha0, digits=3)) + $(round(alpha1, digits=3)) * n + ε1 with ε1 ~ N(0, $(round(omega_1, digits=5)))")

# Fitting model for σ(n)
# We adjust a quadratic model for the std dev of log-severity vs frequency
# g2(n) = β0 + β1*n + β2*n^2
model_g2 = lm(@formula(std_log ~ n + n^2), df_summary)
beta0 = coef(model_g2)[1]
beta1 = coef(model_g2)[2]
beta2 = coef(model_g2)[3]

# Omega_2 is defined as the MSE of g2(n)
res_g2 = residuals(model_g2)
omega_2 = sqrt(mean(res_g2.^2))
e_2 = Normal(0, omega_2)

println("\n--- Model σ(n) = g2(n) + ε2 with ε2 ~ N(0, omega_2) ---")
println("Estimated equation: σ(n) = $(round(beta0, digits=3)) + $(round(beta1, digits=3)) * n + $(round(beta2, digits=3)) * n^2 + ε2 with ε2 ~ N(0, $(round(omega_2, digits=5)))")

# predicctions for plotting
begin
    n_range = range(minimum(unique_freqs[valid_idx]), maximum(unique_freqs[valid_idx]), length=100)
    df_pred = DataFrame(n = n_range)
    df_pred.pred_g1 = predict(model_g1, df_pred)
    df_pred.pred_g2 = predict(model_g2, df_pred)
end

# Plot 1: empirical mean vs model g1
begin
    p1 = plot(df_summary.n, df_summary.mean_log, label="Empiric", title="Mean log-severity conditional on frequency",legend=:bottomright,color=:red,linewidth=3)
    xlabel!("frequency")
    ylabel!("mean")
    plot!(df_pred.n, df_pred.pred_g1, label="Model g1(n) (Linear)", color=:blue, linewidth=2)
end

# Plot 2: standard deviation vs model g2
begin
    p2 = plot(df_summary.n, df_summary.std_log,label="Empiric",title="Standard deviation of conditional log-severity", legend=:bottomright, color=:blue, linewidth=3)
    xlabel!("frequency")
    ylabel!("standard deviation")
    plot!(df_pred.n, df_pred.pred_g2, label="Model g2(n) (Quadratic)", color=:orange, linewidth=2)
end

# combined plot
final_plot = plot(p1, p2, layout=(2, 1), size=(600, 600))

### Simulate Collective risk model
NB = NegativeBinomial(r, p)

g1(n) = alpha0 + alpha1 * n
g2(n) = beta0 + beta1 * n + beta2 * n^2
e_1 = Normal(0, omega_1)
e_2 = Normal(0, omega_2)

function simulateCRM(m = 10_000)
    S = zeros(m)
    N = rand(NB, m)
    iN = findall(N .> 0)
    Y_all = Float64[]
    
    for i in iN
        n = N[i]
        mu_n = g1(n) + rand(e_1)
        sigma_n = g2(n) + rand(e_2)
        
        sigma_n = max(sigma_n, 1e-6)
        
        d = LogNormal(mu_n, sigma_n)
        
        Yi = rand(d, n)
        S[i] = sum(Yi)
        append!(Y_all, Yi)
    end
    
    PS0 = 1 - length(iN)/m
    
    return (ES = mean(S), MS = median(S), VS = var(S), PS0 = PS0,
            VaRS = quantile(S, 0.995), S = S, N = N, Y = Y_all
    )
end

# Run simulation
results = simulateCRM(100_000)
println("Observed versus simulated")
println("-------------------------")
println("E(S)    = ", (mean(reclam), results.ES))
println("M(S)    = ", (median(reclam), results.MS))
println("VaR(S)  = ", (quantile(reclam, 0.995), results.VaRS))
println("P(S=0)  = ", (mean(reclam .== 0), results.PS0))

# ==============================================================================
# Code in Julia a discrete-time risk process using as initial capital de Mexican regulatory solvency capital requirement,
# considering an annual low risk interest rate 7% and a ROE of 13%.
# ==============================================================================

# Parameters
begin
    i_rate = 0.07 
    r_rate = 0.13  

    # BEL = E(S)
    BEL_0 = results.ES
    SCR_0 = results.VaRS - results.ES
    RM_0 = (r_rate - i_rate) * SCR_0

    # Initial Capital according to Mexican regulation C_0 = (1 - (r - i)) * SCR
    C_0 = (1 - (r_rate - i_rate)) * SCR_0

    println("\n--- Solvency Parameters ---")
    println("BEL (Best Estimate): ", round(BEL_0, digits=2))
    println("SCR (Solvency Capital): ", round(SCR_0, digits=2))
    println("RM (Risk Margin): ", round(RM_0, digits=2))
    println("Initial Capital (C_0): ", round(C_0, digits=2))
    println("Total Assets (A = BEL + RM + C_0): ", round(BEL_0 + RM_0 + C_0, digits=2))
end

function simulate_risk_process(m_sims=10_000)
    i_daily = (1 + i_rate)^(1/365) - 1
    
    C_end = zeros(m_sims)
    
    for k in 1:m_sims
        n = rand(NB)
        
        if n > 0
            mu_n = g1(n) + rand(e_1)
            sigma_n = g2(n) + rand(e_2)
            sigma_n = max(sigma_n, 1e-6)
            claims = rand(LogNormal(mu_n, sigma_n), n)
            days = rand(1:365, n)
            daily_claims = zeros(365)
            
            for j in 1:n
                daily_claims[days[j]] += claims[j]
            end
        else
            daily_claims = zeros(365)
        end

        BEL = BEL_0
        RM = RM_0
        SCR = SCR_0
        
        A = BEL + RM + C_0
        
        for t in 1:365
            C_t = daily_claims[t]
            
            A = A * (1 + i_daily) - C_t
            
            prev_BEL = BEL
            prev_RM = RM
            prev_SCR = SCR
            
            BEL = max(0.0, prev_BEL - C_t)
            
            excess_claim_BEL = max(0.0, C_t - prev_BEL)
            RM = max(0.0, prev_RM - excess_claim_BEL)
            
            excess_claim_RM = max(0.0, excess_claim_BEL - prev_RM)
            SCR = max(0.0, prev_SCR - excess_claim_RM)
        end
        
        C_end[k] = A
    end
    
    return C_end
end

# Run simulation
C_end = simulate_risk_process(50_000)

# Calculate ROE
begin
    ROE = (C_end .- C_0) ./ C_0

    println("Expected ROE: ", round(mean(ROE) * 100, digits=2), "%")
    println("Prob(ROE >= 13%): ", round(mean(ROE .>= 0.13), digits=4))
    println("Prob(Insolvency): ", round(mean(C_end .< 0), digits=4))

    histogram(ROE, label="Simulated ROE", xlabel="ROE", ylabel="Frequency", legend=:topleft)
    vline!([0.13], label="Target 13%", lw=2)
end

# ==============================================================================
# Calculate the probability of ruin and a point estimation of the ruin severity 
# over the following time horizons: 1, 5 and 10 years. 
# Illustrate with appropriate graphs.
# ==============================================================================

function simulate_ruin_probability(years, num_sims)
    # Premium calculation: Pi = E(S) + (r - i) * SCR
    # We assume parameters are constant over time
    Pi = BEL_0 + (r_rate - i_rate) * SCR_0
    
    ruin_counts = zeros(Int, years)
    ruin_severities = [Float64[] for _ in 1:years]
    
    # Store capitals at specific years for distribution plots
    capitals_snapshots = Dict{Int, Vector{Float64}}()
    target_years = [1, 5, 10]
    for y in target_years
        if y <= years
            capitals_snapshots[y] = zeros(num_sims)
        end
    end
    
    # For plotting paths
    paths = zeros(years + 1, min(num_sims, 100))
    paths[1, :] .= C_0
    
    for k in 1:num_sims
        C = C_0
        is_ruined = false
        
        for t in 1:years
            # Generate S_t
            n = rand(NB)
            if n > 0
                mu_n = g1(n) + rand(e_1)
                sigma_n = g2(n) + rand(e_2)
                sigma_n = max(sigma_n, 1e-6)
                d = LogNormal(mu_n, sigma_n)
                S_t = sum(rand(d, n))
            else
                S_t = 0.0
            end
            
            # Update Capital
            C = C + Pi - S_t
            
            if k <= 100
                paths[t+1, k] = C
            end
            
            if t in keys(capitals_snapshots)
                capitals_snapshots[t][k] = C
            end
            
            if !is_ruined && C < 0
                is_ruined = true
                current_severity = -C
                
                # Record ruin for this year and all subsequent years
                # (Cumulative probability of ruin)
                # Severity is recorded as the deficit at the FIRST time of ruin
                for future_t in t:years
                    ruin_counts[future_t] += 1
                    push!(ruin_severities[future_t], current_severity)
                end
            end
        end
    end
    
    probs = ruin_counts ./ num_sims
    mean_severities = [isempty(x) ? 0.0 : mean(x) for x in ruin_severities]
    
    return probs, mean_severities, paths, ruin_severities, capitals_snapshots
end

# Run simulation
T_max = 10
probs, severities, paths, all_severities, capitals_snapshots = simulate_ruin_probability(T_max, 50_000)

println("\n--- Ruin Probability and Severity ---")
horizons = [1, 5, 10]
for t in horizons
    println("Horizon $t years:")
    println("  Probability of Ruin: ", round(probs[t], digits=4))
    println("  Expected Severity:   ", round(severities[t], digits=2))
end

# Theoretical 1-year severity check
# E(-C1 | C1 < 0) = E(S1 | S1 > VaR) - VaR
S_sim = results.S
VaR_S = results.VaRS
tail_S = S_sim[S_sim .> VaR_S]
theoretical_sev_1 = mean(tail_S) - VaR_S
println("\nTheoretical 1-year Severity (from formula): ", round(theoretical_sev_1, digits=2))

# Graphs
begin
    # Helper to create path plot
    function plot_paths(limit_t)
        p = plot(0:limit_t, paths[1:limit_t+1, :], 
            title="Paths (0-$limit_t yrs)", 
            xlabel="Time", ylabel="Capital", 
            legend=false, alpha=0.3, color=:blue)
        hline!(p, [0], color=:red, lw=2)
        return p
    end

    # Helper to create distribution plot
    function plot_dist(data, t)
        pr = round(probs[t] * 100, digits=2)
        es = round(severities[t], digits=2)
        p = histogram(data, 
            title="Dist. Year $t\nProb Ruin: $pr%, Exp Sev: $es", 
            xlabel="Capital", ylabel="Freq", 
            legend=false, normalize=:pdf, alpha=0.6, color=:green, titlefontsize=9)
        vline!(p, [0], color=:red, lw=2)
        return p
    end

    plots_list = []
    for t in [1, 5, 10]
        push!(plots_list, plot_paths(t))
        push!(plots_list, plot_dist(capitals_snapshots[t], t))
    end
    
    plot(plots_list..., layout=(3,2), size=(1000, 1200))
end
