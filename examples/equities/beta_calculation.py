import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import date, datetime
from fixedIncome.src.portfolio.portfolio_optimization import PortfolioOptimizer

start_date = date(2012, 1, 1)
end_date   = date(2026, 3, 21)


schd_df = yf.download('SCHD', str(start_date), str(end_date))
spy_df  = yf.download('SPY', str(start_date), str(end_date))
ms_df   = yf.download('MS', str(start_date), str(end_date))
lqd_df  = yf.download('LQD', str(start_date), str(end_date))


schd_df[('Daily Return', 'SCHD')] = schd_df[('Close', 'SCHD')].diff(1)/schd_df[('Close', 'SCHD')]
spy_df[('Daily Return', 'SPY')]   = spy_df[('Close', 'SPY')].diff(1)/spy_df[('Close', 'SPY')]
ms_df[('Daily Return', 'MS')]     = ms_df[('Close', 'MS')].diff(1)/ms_df[('Close', 'MS')]
lqd_df[('Daily Return', 'LQD')]     = lqd_df[('Close', 'LQD')].diff(1)/lqd_df[('Close', 'LQD')]

# ---- Plot prices ---- #

fig = plt.figure(figsize=(10, 6))
plt.plot(spy_df[('Close', 'SPY')]/spy_df[('Close', 'SPY')].iloc[0])
plt.plot(schd_df[('Close', 'SCHD')]/schd_df[('Close', 'SCHD')].iloc[0])
plt.plot(ms_df[('Close', 'MS')]/ms_df[('Close', 'MS')].iloc[0])
plt.plot(lqd_df[('Close', 'LQD')]/lqd_df[('Close', 'LQD')].iloc[0])
plt.legend(['SPY', 'SCHD', 'MS', 'LQD'], frameon=False)
plt.title(f'SPY, SCHD, MS, and LQD Normalized Prices from {start_date} to {end_date}\nWith Dividend and Split Adjustments')

# --- Calculate beta and idio --- #

schd_lm_model = LinearRegression()
ms_lm_model   = LinearRegression()
lqd_lm_model  = LinearRegression()

market_returns = spy_df[('Daily Return', 'SPY')].values.reshape(-1, 1)[1:, :]
schd_returns   = schd_df[('Daily Return', 'SCHD')].values.reshape((-1, 1))[1:, :]
ms_returns     = ms_df[('Daily Return', 'MS')].values.reshape((-1, 1))[1:, :]
lqd_returns    = lqd_df[('Daily Return', 'LQD')].values.reshape((-1, 1))[1:, :]

schd_lm_model.fit(market_returns, schd_returns)
ms_lm_model.fit(market_returns, ms_returns)
lqd_lm_model.fit(market_returns, lqd_returns)

schd_beta = float(schd_lm_model.coef_)
schd_alpha = float(schd_lm_model.intercept_)

ms_beta = float(ms_lm_model.coef_)
ms_alpha = float(ms_lm_model.intercept_)

lqd_beta = float(lqd_lm_model.coef_)
lqd_alpha = float(lqd_lm_model.intercept_)

schd_market_return_part = schd_beta * market_returns
schd_idio_return_part = schd_returns - schd_market_return_part

ms_market_return = ms_beta * market_returns
ms_idio_return = ms_returns - ms_market_return

lqd_market_return = lqd_beta * market_returns
lqd_idio_return   = lqd_returns - lqd_market_return



plt.figure(figsize=(10, 6))
plt.plot(schd_df.index.values[1:], schd_idio_return_part.flatten())
plt.title(f'SCHD Idio Return from {start_date} to {end_date}')

plt.figure(figsize=(10, 6))
plt.plot(ms_df.index.values[1:], ms_idio_return.flatten())
plt.title(f'MS Idio Return from {start_date} to {end_date}')

plt.figure(figsize=(10, 6))
plt.plot(lqd_df.index.values[1:], lqd_idio_return.flatten())
plt.title(f'LQD Idio Return from {start_date} to {end_date}')





plt.figure(figsize=(10, 6))
plt.plot(spy_df.index.values[1:], market_returns.flatten())
plt.plot(schd_df.index.values[1:], schd_market_return_part.flatten())
plt.title(f'SPY Returns and SCHD Market Return from {start_date} to {end_date}')
plt.legend(['SPY', f'SCHD Market Return\nBeta {round(schd_beta, 2)}'], frameon=False)

plt.figure(figsize=(10, 6))
plt.plot(ms_df.index.values[1:], ms_market_return.flatten(), color='tab:green')
plt.plot(spy_df.index.values[1:], market_returns.flatten(), color='tab:blue')
plt.title(f'SPY Returns and MS Market Return from {start_date} to {end_date}')
plt.legend([f'MS Market Return\nBeta {round(ms_beta, 2)}', 'SPY'], frameon=False)

plt.figure(figsize=(10, 6))
plt.plot(spy_df.index.values[1:], market_returns.flatten(), color='tab:blue')
plt.plot(lqd_df.index.values[1:], lqd_market_return.flatten(), color='tab:red')
plt.title(f'SPY Returns and LQD Market Return from {start_date} to {end_date}')
plt.legend(['SPY', f'LQD Market Return\nBeta {round(lqd_beta, 2)}'], frameon=False)



# Risk decomposition
reconstructed_sched_risk = (schd_market_return_part.std()**2 + schd_idio_return_part.std()**2)**0.5

schd_risk = schd_returns.std()

# --- Mean-Variance Portfolio Analysis --- #
num_weights = 200_000
num_assets = 4

weights = np.random.random((num_weights, num_assets))
weight_means = weights.sum(axis=1)
normalized_weights = (weights.T / weight_means).T

all_returns = [market_returns, schd_returns, ms_returns, lqd_returns]
returns = np.concatenate([asset_return for index, asset_return in enumerate(all_returns) if index < num_assets], axis=1)

mean_returns = returns.mean(axis=0) * 252
cov_returns = np.cov(returns.T) * 252



RISK_FREE_RATE = 0.0335

optimizer = PortfolioOptimizer(mean_vector=mean_returns, covariance_mat=cov_returns)
returns, volatilities, weights = optimizer.calculate_efficient_frontier(return_increment=0.0001)

sharpe_ratios = optimizer.sharpe_ratio(returns=returns, volatilities=volatilities, risk_free_rate=RISK_FREE_RATE)
tangency_portfolio_index = np.argmax(sharpe_ratios)
tangent_portfolio_return = returns[tangency_portfolio_index]
tangent_portfolio_vol = volatilities[tangency_portfolio_index]
tangent_portfolio_weights = weights[tangency_portfolio_index]
tangent_portfolio_sharpe = (tangent_portfolio_return - RISK_FREE_RATE) / tangent_portfolio_vol

# now find weights which maximize the Utility function for various param values

agg_weights = optimizer.find_maximum_utility_portfolio(initial_wealth=10, param=0.1)
agg_utility_return = mean_returns.dot(agg_weights)
agg_cov = agg_weights.dot(cov_returns.dot(agg_weights))**0.5

cons_weights = optimizer.find_maximum_utility_portfolio(initial_wealth=10, param=0.5)
cons_utility_return = mean_returns.dot(cons_weights)
cons_cov = cons_weights.dot(cov_returns.dot(cons_weights))**0.5

# plot efficient frontier with tangent and Utility max portfolios
plt.figure(figsize=(10, 6))
plt.scatter(volatilities, returns, marker='.', c= sharpe_ratios)
plt.plot(tangent_portfolio_vol, tangent_portfolio_return, marker=r'$\star$', markersize=15, color='gold')
plt.plot(agg_cov, agg_utility_return, marker='^', markersize=10, color='black')
plt.plot(cons_cov, cons_utility_return, marker='v', markersize=10, color='black')

plt.plot([0.0, tangent_portfolio_vol], [RISK_FREE_RATE, tangent_portfolio_return])
plt.colorbar(label='Sharpe Ratio')
plt.legend(['Efficient Frontier', 'Tangency Portfolio',
            r'Utility Max. Portfolio $\lambda$ = 0.1',
            r'Utility Max. Portfolio $\lambda$ = 0.5',
            'Capital Market Line'], frameon=False, loc='lower right')

plt.grid(alpha=0.1)
plt.xlim(0, max(volatilities)*1.1)
plt.ylim(0, max(returns)*1.1)
plt.ylabel('Annualized Return')
plt.xlabel('Annualized Volatility')
plt.suptitle(f'Efficient Frontier and Tangency Portfolio for a Portfolios Consisting of SPY, SCHD, MS, and LQD\n'
             f'With a Risk-free Rate of {RISK_FREE_RATE*100}%')
plt.text(0.01, 0.11,  f"Tangency Portfolio\n  SPY {round(tangent_portfolio_weights[0], 2)}\n"
                     f"  SCHD {round(tangent_portfolio_weights[1], 2)}\n"
                     f"  MS {round(tangent_portfolio_weights[2], 2)}\n"
                     f"  LQD {round(tangent_portfolio_weights[3], 2)}\n")

#----------------------------------------------------
# Estimates of the uncertainty for Risk-Free Portfolio
# by Bootstrapping

NUM_BOOT_SUBSAMPLES = 100
boot_sharpe_ratios = []
actual_sharpe_ratios = []
mean_shrinkage = 0.1

for i in range(NUM_BOOT_SUBSAMPLES):
    np.random.seed(i)
    bootstrap_indices = np.random.randint(len(returns), size=len(returns))
    boot_samples_returns = returns[bootstrap_indices, :]

    boot_mean_return = boot_samples_returns.mean(axis=0) * 252
    boot_mean = np.mean(boot_mean_return)
    boot_mean_return_with_shrinkage = (1-mean_shrinkage) * boot_mean_return + mean_shrinkage * boot_mean

    boot_cov = np.cov(boot_samples_returns.T) * 252

    boot_optimizer = PortfolioOptimizer(mean_vector=boot_mean_return_with_shrinkage, covariance_mat=boot_cov)
    boot_ef_returns, boot_ef_volatilities, boot_ef_weights = boot_optimizer.calculate_efficient_frontier(
        return_increment=0.0001
    )

    boot_ef_sharpe_ratios = boot_optimizer.sharpe_ratio(
        returns=boot_ef_returns, volatilities=boot_ef_volatilities, risk_free_rate=RISK_FREE_RATE
    )

    boot_tangency_portfolio_index = np.argmax(boot_ef_sharpe_ratios)
    boot_tan_portfolio_return = boot_ef_returns[boot_tangency_portfolio_index]
    boot_tan_portfolio_vol = boot_ef_volatilities[boot_tangency_portfolio_index]
    boot_tan_portfolio_weights = boot_ef_weights[boot_tangency_portfolio_index]
    boot_tan_sharpe_ratio = boot_ef_sharpe_ratios[boot_tangency_portfolio_index]

    boot_sharpe_ratios.append(boot_tan_sharpe_ratio)

    actual_return_ratio = mean_returns.dot(boot_tan_portfolio_weights)
    actual_vol = (boot_tan_portfolio_weights.dot(cov_returns.dot(boot_tan_portfolio_weights)))**0.5
    actual_sharpe_ratio = (actual_return_ratio - RISK_FREE_RATE) / actual_vol
    actual_sharpe_ratios.append(actual_sharpe_ratio)

    print(i)


plt.boxplot(np.array([actual_sharpe_ratios, boot_sharpe_ratios]).T)
plt.title(f"Actual and Estimated Sharpe Ratios From\n{NUM_BOOT_SUBSAMPLES} Bootstrapped Sub-Samplings of Returns")
plt.axhline(y=tangent_portfolio_sharpe, color='r', linestyle='--', label='Tangent Portfolio Sharpe')
plt.xticks(ticks=[1, 2], labels=['Actual', 'Estimated'])
plt.show()


