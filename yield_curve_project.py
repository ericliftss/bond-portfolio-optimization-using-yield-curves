"""Historical NSS / PCA bond portfolio experiment. 

Run: python yield_curve_project.py --data FedYCdb.csv --output results
Read README.md before interpreting results. Rates inside calculations are decimals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog, minimize

PARAMETERS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
#BETA0 Represents the long-term interest rate level as maturity goes to infinity.
#BETA1 Represents the short-term slope component (difference between short and long rates).
#BETA2 First curvature/hump, BETA3 Second curvature/hump (Svensson extension)
#TAU1 Decay rate for BETA1, TAU2 Decay rate for BETA3
MATURITIES = np.array([1., 2., 3., 5., 7., 10.])
# Include half-year nodes so all semiannual cash flows lie inside the risk grid.
RISK_GRID = np.arange(0.5, 10.01, 0.5)
STRATEGIES = ["PCA", "Duration", "Equal weight"]


@dataclass(frozen=True)
class Bond:
    """Hypothetical fixed-rate, option-free bond, purchased on a coupon date."""
    maturity: float
    coupon: float = 0.04
    face: float = 100.0
    frequency: int = 2

    def cashflows(self):
        count = int(round(self.maturity * self.frequency))
        if count <= 0 or not np.isclose(count / self.frequency, self.maturity):
            raise ValueError("Maturity must be a positive whole number of coupon periods.")
        times = np.arange(1, count + 1, dtype=float) / self.frequency
        cash = np.full(count, self.face * self.coupon / self.frequency)
        cash[-1] += self.face
        return times, cash


def nss_zero_rate(times, parameters):
    """Fed NSS convention: betas in percent; taus in YEARS (not inverse taus).

    z(t) = b0 + b1*L(t/tau1) + b2*[L(t/tau1)-exp(-t/tau1)]
           + b3*[L(t/tau2)-exp(-t/tau2)]. Returns decimal continuous rates.
    """
    #where L = (1-exp(-t/tau1))
    times = np.asarray(times, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if p.shape != (6,) or not np.all(np.isfinite(p)) or min(p[4:]) <= 0:
        raise ValueError("Expected six finite NSS parameters and positive taus.")
    if np.any(times < 0):
        raise ValueError("Maturities cannot be negative.")
    def loading(tau):
        x = times / tau
        return np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)
    l1, l2 = loading(p[4]), loading(p[5])
    return (p[0] + p[1]*l1 + p[2]*(l1-np.exp(-times/p[4]))
            + p[3]*(l2-np.exp(-times/p[5]))) / 100.0


def price_bond(bond, parameters, elapsed=0.0, parallel_shift=0.0):
    """Full/dirty value of REMAINING payments; elapsed is years since purchase."""
    times, cash = bond.cashflows()
    remaining = times - elapsed
    keep = remaining > 1e-10
    remaining, cash = remaining[keep], cash[keep]
    return float(np.sum(cash * np.exp(-(nss_zero_rate(remaining, parameters)
                                      + parallel_shift) * remaining)))


def holding_return(bond, start_parameters, end_parameters, elapsed):
    """Total return: aged dirty price plus coupons/principal paid during holding.

    Cash receipts earn zero interest until the next rebalance. No future cash
    flows enter the current price before their scheduled payment.
    """
    times, cash = bond.cashflows()
    receipts = float(cash[times <= elapsed + 1e-10].sum())
    return (price_bond(bond, end_parameters, elapsed) + receipts) / price_bond(
        bond, start_parameters) - 1.0


def yield_to_maturity(bond, price):
    """Nominal annual YTM with semiannual compounding (same convention for all)."""
    times, cash = bond.cashflows()
    m = bond.frequency
    def gap(y):
        return np.sum(cash / (1+y/m)**(m*times)) - price
    return float(brentq(gap, -0.5, 1.0))


def sensitivities(bonds, parameters):
    """Analytical normalized price gradient wrt zero-rate grid, and parallel D/C.

    At purchase every cash flow lies exactly on the half-year risk grid.
    Gradient is negative. Duration = -sum(gradient); convexity is wrt a
    parallel shift in continuous zero rates, not conventional modified YTM D.
    """
    jacobian, durations, convexities = [], [], []
    for bond in bonds:
        times, cash = bond.cashflows()
        pv = cash * np.exp(-nss_zero_rate(times, parameters) * times)
        price = pv.sum()
        j = np.zeros(len(RISK_GRID))
        indices = np.rint(times*2).astype(int)-1
        np.add.at(j, indices, -times*pv/price)
        jacobian.append(j)
        durations.append(np.sum(times*pv)/price)
        convexities.append(np.sum(times**2*pv)/price)
    return np.array(jacobian), np.array(durations), np.array(convexities)


def load_data(path):
    """Accept the Fed's CSV with its explanatory preamble or a clean CSV."""
    with open(path, encoding="utf-8-sig") as stream:
        header = next((i for i, line in enumerate(stream)
                       if line.strip().startswith("Date,BETA0")), None)
    if header is None:
        raise ValueError("Cannot find Date,BETA0 header in Fed CSV.")
    data = pd.read_csv(path, skiprows=header, parse_dates=["Date"])
    required = PARAMETERS + [f"SVENY{i:02}" for i in range(1, 11)]
    missing = set(required)-set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data[required] = data[required].apply(pd.to_numeric, errors="coerce")
    data = data.set_index("Date").sort_index()
    if data.index.has_duplicates or data.index.isna().any():
        raise ValueError("Duplicate or invalid observation dates.")
    valid = np.isfinite(data[required]).all(axis=1) & (data.TAU1 > 0) & (data.TAU2 > 0)
    # Early observations without valid Svensson parameters are excluded explicitly.
    return data.loc[valid].copy(), int((~valid).sum())


def curve_panel(data):
    return pd.DataFrame([nss_zero_rate(RISK_GRID, row)
                         for row in data[PARAMETERS].to_numpy()],
                        index=data.index, columns=RISK_GRID)


def risk_models(history, jacobian, durations, factors=3):
    """PCA on demeaned DAILY changes, using covariance (not correlation).

    Preserve leading factors plus a diagonal discarded-factor residual so
    optimization cannot treat all omitted curve risk as zero.
    Duration benchmark assumes a uniform shift equal to mean grid change.
    """
    covariance = np.cov(history, rowvar=False, ddof=1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    leading = vectors[:, :factors]
    explained = values[:factors] / values.sum()
    exposures = jacobian @ leading
    factor_risk = (exposures * values[:factors]) @ exposures.T
    residual = np.maximum(np.diag(covariance - (leading*values[:factors])@leading.T), 0)
    pca_risk = factor_risk + (jacobian * residual) @ jacobian.T
    shift_variance = np.var(history.mean(axis=1), ddof=1)
    duration_risk = np.outer(durations, durations) * shift_variance
    # One common tiny ridge fixes singularity/ties without privileging a model.
    ridge = max(np.trace(jacobian@covariance@jacobian.T)/len(durations), 1e-12)*1e-6
    return (pca_risk + ridge*np.eye(len(durations)),
            duration_risk + ridge*np.eye(len(durations)),
            values, vectors, explained, exposures)


def optimize(risk, yields, target, cap=0.40):
    """Minimum variance; exact yield equality for a fair paired comparison."""
    n = len(yields)
    if cap*n < 1:
        raise ValueError("Position cap makes a fully invested portfolio infeasible.")
    # Scaling avoids SLSQP considering tiny daily variances already converged.
    q = risk / max(float(np.max(np.diag(risk))), 1e-15)
    yield_scale = max(float(np.ptp(yields)), 1e-4)
    centered = (yields-target) / yield_scale
    constraints = [{"type":"eq", "fun":lambda w: w.sum()-1,
                    "jac":lambda w: np.ones(n)}]
    if np.ptp(yields) > 1e-10:
        constraints.append({"type":"eq", "fun":lambda w: centered@w,
                            "jac":lambda w: centered})
    elif abs(target-yields[0]) > 1e-9:
        raise ValueError("Yield target infeasible for identical asset yields.")
    result = minimize(lambda w: w@q@w, np.ones(n)/n,
                      jac=lambda w: 2*q@w, method="SLSQP",
                      bounds=[(0, cap)]*n, constraints=constraints,
                      options={"ftol":1e-11,"maxiter":1000})
    w = result.x
    if (not result.success or abs(w.sum()-1)>1e-7
        or abs(w@yields-target)>1e-7 or np.min(w)<-1e-8
        or np.max(w)>cap+1e-8):
        raise RuntimeError(f"Optimization failed: {result.message}")
    return w


def performance(returns):
    """Monthly-rebalance returns; annualization uses actual elapsed years."""
    result = []
    for name in STRATEGIES:
        r = returns[name].to_numpy()
        wealth = np.r_[1., np.cumprod(1+r)]
        elapsed = returns.attrs["elapsed_years"]
        rate = len(r)/elapsed
        vol = r.std(ddof=1)*np.sqrt(rate)
        result.append({"strategy":name, "total_return":wealth[-1]-1,
                       "annualized_return":wealth[-1]**(1/elapsed)-1,
                       "annualized_volatility":vol,
                       "maximum_drawdown":np.min(wealth/np.maximum.accumulate(wealth)-1),
                       "worst_period_return":r.min(), "periods":len(r)})
    return pd.DataFrame(result).set_index("strategy")


def backtest(data, curves, start="2005-01-01", end="2025-12-31", lookback=756, cap=.4):
    bonds = [Bond(m) for m in MATURITIES]
    changes = curves.diff()
    available = data.loc[start:end]
    # First available observation in every calendar month. Close-to-close ideal execution.
    dates = available.groupby(available.index.to_period("M")).head(1).index
    if len(dates)<3:
        raise ValueError("Need at least three rebalancing dates.")
    # Include the final requested observation, so December is not silently omitted.
    if dates[-1] != available.index[-1]:
        dates = dates.append(pd.DatetimeIndex([available.index[-1]]))
    records, weight_records, bond_records = [], [], []
    earliest = None
    last_snapshot = None
    for date, next_date in zip(dates[:-1], dates[1:]):
        history = changes.loc[changes.index < date].dropna().tail(lookback)
        if len(history)<lookback:
            raise ValueError(f"Insufficient training history at {date.date()}; move --start later.")
        if earliest is None:
            earliest = date
        params = data.loc[date, PARAMETERS].to_numpy(dtype=float)
        end_params = data.loc[next_date, PARAMETERS].to_numpy(dtype=float)
        prices = np.array([price_bond(b, params) for b in bonds])
        yields = np.array([yield_to_maturity(b, p) for b,p in zip(bonds, prices)])
        jac, durations, convexities = sensitivities(bonds, params)
        pca, duration, values, vectors, explained, exposures = risk_models(
            history.to_numpy(), jac, durations)
        target = yields.mean()  # Equal-weight portfolio satisfies exactly the same target.
        allocations = {"PCA":optimize(pca,yields,target,cap),
                       "Duration":optimize(duration,yields,target,cap),
                       "Equal weight":np.ones(len(bonds))/len(bonds)}
        elapsed = (next_date-date).days/365.25
        asset_returns = np.array([holding_return(b,params,end_params,elapsed) for b in bonds])
        period = {"date":next_date,"start_date":date,"target_yield":target,
                  "pca_explained_variance":explained.sum(),
                  "training_end":history.index[-1]}
        for name,w in allocations.items():
            period[name] = float(w@asset_returns)
            weight_records.append({"date":date,"strategy":name,"target_yield":target,
                "portfolio_yield":float(w@yields),"parallel_duration":float(w@durations),
                "parallel_convexity":float(w@convexities),
                **{f"weight_{int(m)}y":float(a) for m,a in zip(MATURITIES,w)}})
        for i,b in enumerate(bonds):
            bond_records.append({"date":date,"maturity":b.maturity,"price":prices[i],
                "ytm":yields[i],"parallel_duration":durations[i],
                "parallel_convexity":convexities[i],"holding_return":asset_returns[i],
                **{f"PC{k+1}_exposure":exposures[i,k] for k in range(3)}})
        records.append(period)
        last_snapshot = (date,pca,duration,yields,values,vectors)
    returns = pd.DataFrame(records).set_index("date")
    returns.attrs["elapsed_years"] = (dates[-1]-earliest).days/365.25
    return returns, pd.DataFrame(weight_records), pd.DataFrame(bond_records), last_snapshot


def frontier(snapshot, cap):
    date,pca,duration,yields,_,_ = snapshot
    bounds = [(0,cap)]*len(yields)
    low = linprog(yields,A_eq=[np.ones(len(yields))],b_eq=[1],bounds=bounds,method="highs")
    high = linprog(-yields,A_eq=[np.ones(len(yields))],b_eq=[1],bounds=bounds,method="highs")
    if not low.success or not high.success:
        raise RuntimeError("Cannot determine feasible yield range.")
    records=[]
    # Avoid degenerate extreme-bound solutions by a tiny inward margin.
    for target in np.linspace(low.fun+1e-8,-high.fun-1e-8,25):
        for name,risk in [("PCA",pca),("Duration",duration)]:
            w=optimize(risk,yields,target,cap)
            records.append({"date":date,"strategy":name,"target_yield":target,
                "pca_model_annual_volatility":np.sqrt(w@pca@w*252),
                "own_model_annual_volatility":np.sqrt(w@risk@w*252)})
    return pd.DataFrame(records)


def save_plots(data,returns,snapshot,frontier_data,output):
    plt.rcParams.update({"figure.dpi":140,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(9,4.8))
    base_date=pd.Timestamp(returns.start_date.iloc[0])
    for name in STRATEGIES:
        index=pd.DatetimeIndex([base_date]).append(returns.index)
        ax.plot(index,np.r_[100,100*(1+returns[name]).cumprod()],label=name)
    ax.set(title="Historical model-priced bond portfolios",ylabel="Portfolio value (start = 100)")
    ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(output/"portfolio_growth.png")
    plt.show()
    plt.close(fig)
    date,_,_,_,values,vectors=snapshot
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for k in range(3):axes[0].plot(RISK_GRID,vectors[:,k],label=f"PC{k+1}")
    axes[0].set(title=f"PCA directions: training before {date.date()}",xlabel="Maturity (years)",ylabel="Unit eigenvector loading")
    axes[0].legend();axes[1].bar(np.arange(1,7),100*values[:6]/values.sum())
    axes[1].set(title="Variance explained",xlabel="Principal component",ylabel="Share (%)")
    fig.tight_layout();fig.savefig(output/"pca_factors.png")
    plt.show()
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for name,group in frontier_data.groupby("strategy"):
        ax.plot(group.pca_model_annual_volatility*100,group.target_yield*100,label=name)
    ax.set(title=f"Yield versus PCA-model risk: {date.date()}",xlabel="Estimated annual volatility (%)",ylabel="Weighted-average YTM (%)")
    ax.legend();fig.tight_layout();fig.savefig(output/"yield_risk_frontier.png")
    plt.show()
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    times=np.linspace(.05,10,200)
    for target in ["2007-07-01","2020-03-01","2022-10-01",str(date.date())]:
        position=data.index.searchsorted(pd.Timestamp(target))
        if position<len(data):
            day=data.index[position]
            ax.plot(times,nss_zero_rate(times,data.loc[day,PARAMETERS].to_numpy())*100,label=str(day.date()))
    ax.set(title="NSS zero-coupon yield curves",xlabel="Maturity (years)",ylabel="Continuous zero yield (%)")
    ax.legend();fig.tight_layout();fig.savefig(output/"yield_curves.png")
    plt.show()
    plt.close(fig)


def run(args):
    data,dropped=load_data(args.data)
    # Keep the curve convention honest: published integer yields are rounded.
    errors=[]
    for _,row in data.iloc[::100].iterrows():
        observed=row[[f"SVENY{i:02}" for i in range(1,11)]].to_numpy(float)/100
        errors.append(np.max(np.abs(nss_zero_rate(np.arange(1,11),row[PARAMETERS].to_numpy(float))-observed)))
    max_error=float(max(errors))
    if max_error>1e-5:
        raise ValueError(f"NSS reconstruction does not match published yields: {max_error}")
    # Retain sufficient pre-start history; discard post-end data completely.
    data=data.loc[:args.end]
    curves=curve_panel(data)
    returns,weights,bonds,snapshot=backtest(data,curves,args.start,args.end,args.lookback,args.cap)
    stats=performance(returns)
    frontier_data=frontier(snapshot,args.cap)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    returns.to_csv(output/"period_returns.csv")
    weights.to_csv(output/"portfolio_weights.csv",index=False)
    bonds.to_csv(output/"bond_diagnostics.csv",index=False)
    stats.to_csv(output/"performance_summary.csv")
    frontier_data.to_csv(output/"yield_risk_frontier.csv",index=False)
    save_plots(data,returns,snapshot,frontier_data,output)
    source_hash=hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    manifest={"data_sha256":source_hash,"start":args.start,"end":args.end,
              "lookback_daily_changes":args.lookback,"position_cap":args.cap,
              "maturities":MATURITIES.tolist(),"coupon_rate":.04,"pca_factors":3,
              "max_nss_reconstruction_error_decimal":max_error,"invalid_rows_excluded":dropped,
              "mean_pca_variance_explained":float(returns.pca_explained_variance.mean()),
              "numpy":np.__version__,"pandas":pd.__version__,
              "limitations":["Current-vintage revised Fed curves, not real-time vintages.",
                 "Hypothetical bonds and NSS model prices, not executable security prices.",
                 "Same-day curve-price execution without slippage or transaction costs.",
                 "Monthly sale and replacement at fixed maturities; not actual issuance calendars.",
                 "Cash receipts earn zero interest; yield targets are not expected returns.",
                 "PCA on daily changes; three factors plus diagonal residual risk.",
                 "Convexity enters full historical repricing, not the quadratic optimization objective.",
                 "Near-zero maturities extrapolate the fitted curve."]}
    (output/"run_config.json").write_text(json.dumps(manifest,indent=2))
    print(stats.to_string(float_format=lambda v:f"{v:.4f}"))
    print(f"\nMean variance explained by 3 PCs: {manifest['mean_pca_variance_explained']:.2%}")
    print(f"NSS reconstruction max error: {max_error*10000:.4f} basis points")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("FedYCdb.csv"))
    parser.add_argument("--output",default="results")
    parser.add_argument("--start",default="2005-01-01")
    parser.add_argument("--end",default="2025-12-31")
    parser.add_argument("--lookback",type=int,default=756,help="Prior daily changes (about 3 years)")
    parser.add_argument("--cap",type=float,default=.4,help="Maximum market-value weight")
    options=parser.parse_args()
    if options.lookback<30 or not (1/6<=options.cap<=1):
        parser.error("Require lookback >=30 and 1/6 <= cap <=1.")
    run(options)

