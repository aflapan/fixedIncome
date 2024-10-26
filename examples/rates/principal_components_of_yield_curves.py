"""
Data comes from Gurkaynak, Sack, Wright (2006)
https://www.federalreserve.gov/econres/feds/the-us-treasury-yield-curve-1961-to-the-present.htm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------------------------------
# load data
data = pd.read_csv('../../fixedIncome/data/rates/feds200628.csv', skiprows=9)
data.set_index('Date', inplace=True)

model_coeff_cols = ['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']

def yield_curve(t, coeffs):
    intercept = coeffs['BETA0']
    term_one = coeffs['BETA1'] * (1 - np.exp(-t/coeffs['TAU1'])) / (t/coeffs['TAU1'])
    term_two = coeffs['BETA2'] * ((1 - np.exp(-t / coeffs['TAU1']) ) / (t / coeffs['TAU1']) - np.exp(t / coeffs['TAU1']))
    term_three = coeffs['BETA3'] * ((1 - np.exp(-t / coeffs['TAU2']) ) / (t / coeffs['TAU2']) - np.exp(t / coeffs['TAU2']))
    return intercept + term_one + term_two + term_three



one_year_forward_rates = [
    'SVEN1F01', 'SVEN1F04', 'SVEN1F09'
]

instantaneous_forward_rates = [
    'SVENF01', 'SVENF02', 'SVENF03', 'SVENF04', 'SVENF05', 'SVENF06',
    'SVENF07', 'SVENF08', 'SVENF09', 'SVENF10', 'SVENF11', 'SVENF12',
    'SVENF13', 'SVENF14', 'SVENF15', 'SVENF16', 'SVENF17', 'SVENF18',
    'SVENF19', 'SVENF20', 'SVENF21', 'SVENF22', 'SVENF23', 'SVENF24',
    'SVENF25', 'SVENF26', 'SVENF27', 'SVENF28', 'SVENF29', 'SVENF30'
]

par_yields = [
    'SVENPY01', 'SVENPY02', 'SVENPY03', 'SVENPY04', 'SVENPY05', 'SVENPY06',
    'SVENPY07', 'SVENPY08', 'SVENPY09', 'SVENPY10', 'SVENPY11', 'SVENPY12',
    'SVENPY13', 'SVENPY14', 'SVENPY15', 'SVENPY16', 'SVENPY17', 'SVENPY18',
    'SVENPY19', 'SVENPY20', 'SVENPY21', 'SVENPY22', 'SVENPY23', 'SVENPY24',
    'SVENPY25', 'SVENPY26', 'SVENPY27', 'SVENPY28', 'SVENPY29', 'SVENPY30'
]

zero_coupon_yields = [
    'SVENY01', 'SVENY02', 'SVENY03', 'SVENY04', 'SVENY05',
    'SVENY06', 'SVENY07', 'SVENY08', 'SVENY09', 'SVENY10',
    'SVENY11', 'SVENY12', 'SVENY13', 'SVENY14', 'SVENY15',
    'SVENY16', 'SVENY17', 'SVENY18', 'SVENY19', 'SVENY20',
    'SVENY21', 'SVENY22', 'SVENY23', 'SVENY24', 'SVENY25',
    'SVENY26', 'SVENY27', 'SVENY28', 'SVENY29', 'SVENY30'
]


# Principal components of the yield curve
# DAILY CHANGES
yield_curve_change = data[zero_coupon_yields].diff().dropna()  # data from 1985-11-26. Each row represents the change in yield from t-1 to t.
centered_yield_curve_change = yield_curve_change - yield_curve_change.mean()

U, s, Vt = np.linalg.svd(centered_yield_curve_change.values)

level = Vt.T[:, 0]
slope = Vt.T[:, 1]
curvature = Vt.T[:, 2]

plt.figure(figsize=(12, 6))
plt.plot(range(1, 31), level)
plt.plot(range(1, 31), slope)
plt.plot(range(1, 31), curvature)
plt.legend(['Level', 'Slope', 'Curvature'], frameon=False)
plt.grid(alpha=0.25)
plt.title('First Three Principal Components of Daily Changes in the Yield Curve from 1985-11-26 to 2023-11-22')
plt.xlabel('Maturity (Years)')
plt.show()


# DAILY Levels
yield_curve_levels = data[zero_coupon_yields].dropna()  # data from 1985-11-26. Each row represents the change in yield from t-1 to t.
level_means = yield_curve_levels.mean()
centered_yield_curve_levels = yield_curve_levels - level_means

U, s, Vt = np.linalg.svd(centered_yield_curve_levels.values)

pc1_level = Vt.T[:, 0]
pc2_level = Vt.T[:, 1]
pc3_level = Vt.T[:, 2]
pc4_level = Vt.T[:, 3]
if sum(pc1_level) < 0:
    pc1_level = -pc1_level


plt.figure(figsize=(12, 6))
plt.plot(range(1, 31), pc1_level)
plt.plot(range(1, 31), pc2_level)
plt.plot(range(1, 31), pc3_level)
plt.plot(range(1, 31), pc4_level)
plt.legend(['Level', 'Slope', 'Curvature', 'pc4', 'pc5'], frameon=False)
plt.grid(alpha=0.25)
plt.suptitle('First Four Principal Components of Daily Levels in the Yield Curve from 1985-11-25 to 2023-11-22',
             y=0.95)
plt.title('Data Source: Gurkaynak, Sack, and Wright (2006)', size=7, x=0.175)
plt.xlabel('Maturity (Years)')
plt.show()


#-------------------------------------------------------------
# time series of first principal components


principal_components_levels = {'pc1': (yield_curve_levels - level_means) @ pc1_level,
                               'pc2': (yield_curve_levels - level_means) @ pc2_level,
                               'pc3': (yield_curve_levels - level_means) @ pc3_level,
                               'pc4': (yield_curve_levels - level_means) @ pc4_level
                               }

principal_components_levels = pd.DataFrame(principal_components_levels)
principal_components_levels.plot(figsize=(12, 7))
plt.suptitle('First Four Principal Components (Levels)')
plt.title('Data Source: Gurkaynak, Sack, and Wright (2006)', size=7, x=0.175)
plt.legend(frameon=False)
plt.grid(alpha=0.5, axis='y')




#------------------------------------------------------------
# PCA Reconstruction

yield_date = '2023-11-22'
sample_yield_curve = yield_curve_levels.loc[yield_date]
pc1 = pc1_level @ (sample_yield_curve - level_means)
pc2 = pc2_level @ (sample_yield_curve - level_means)
pc3 = pc3_level @ (sample_yield_curve - level_means)
pc4 = pc4_level @ (sample_yield_curve - level_means)

projected_yield_curve = pc1 * pc1_level + pc2 * pc2_level + pc3 * pc3_level + pc4 * pc4_level + level_means


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.plot(range(1, 31), sample_yield_curve, label='Yield Curve', color='black')
plt.plot(range(1, 31), projected_yield_curve, label='Reconstructed Yield Curve', color='black', linestyle='--')
plt.legend(frameon=False, loc='lower right')
plt.suptitle(f'Yield Curve and Reconstructed Yield Curve Using the First Four Principal Components\n'
             f'for Yield Levels on {yield_date}')
plt.title('Data Source: Gurkaynak, Sack, and Wright (2006)', size=7, x=0.175)
plt.grid(alpha=0.5, axis='y')
plt.ylabel('Yield (%)')
plt.xlabel('Years to Maturity')
plt.tight_layout()
plt.show()

plt.savefig('../../docs/images/rates/yield_curve_and_reconstructed_yield_curve.png')



#--------------------------------------------------------------------------#
#----------------------------- Inflation Data -----------------------------#
#--------------------------------------------------------------------------#

inflation_data = pd.read_csv('../../fixedIncome/data/rates/inflation_feds200805.csv', skiprows=18)
inflation_data.set_index('Date', inplace=True)

breakeven_cols = [
    'BKEVENPY02', 'BKEVENPY03', 'BKEVENPY04', 'BKEVENPY05', 'BKEVENPY06', 'BKEVENPY07', 'BKEVENPY08', 'BKEVENPY09',
    'BKEVENPY10', 'BKEVENPY11', 'BKEVENPY12', 'BKEVENPY13', 'BKEVENPY14', 'BKEVENPY15', 'BKEVENPY16', 'BKEVENPY17',
    'BKEVENPY18', 'BKEVENPY19', 'BKEVENPY20'
]


tips_yields = [
    'TIPSY02', 'TIPSY03', 'TIPSY04', 'TIPSY05', 'TIPSY06',
    'TIPSY07', 'TIPSY08', 'TIPSY09', 'TIPSY10', 'TIPSY11',
    'TIPSY12', 'TIPSY13', 'TIPSY14', 'TIPSY15', 'TIPSY16',
    'TIPSY17', 'TIPSY18', 'TIPSY19', 'TIPSY20'
]

breakeven_data = inflation_data[breakeven_cols].dropna()
breakeven_means = breakeven_data.mean()
centered_breakeven = breakeven_data - breakeven_means
U_be, s_be, Vt_be = np.linalg.svd(centered_breakeven.values)

breakeven_pc1 = Vt_be.T[:, 0]
breakeven_pc2 = Vt_be.T[:, 1]
breakeven_pc3 = Vt_be.T[:, 2]

plt.plot(figsize=(12, 5))
plt.plot(breakeven_pc1)
plt.plot(breakeven_pc2)
plt.plot(breakeven_pc3)
plt.show()


(centered_breakeven @ breakeven_pc1).plot(figsize=(13, 5))
