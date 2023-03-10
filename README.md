# fixedIncome
This is a library for fixed income quant analytics, including yield curve calibration, DV01 and convexity calculations, and key rate calculations for hedging. The package source code can be found under fixedIncome/src/, and the unit tests can be found under fixedIncome/tests. 

The below plot represents some of the current capabilities of the package, and it is one of the plots produced by the main.py script. 

![Thirty Year Bond PV](https://github.com/aflapan/fixedIncome/blob/master/docs/images/thrity_year_pv.png)


This project is currently under construction, and future work will include (in order of immediacy):

1. Functionality to download data from the U.S. Treasury Direct API and calibrate a yield curve automatically on real data.
2. Implementations of KeyRate and PCA hedging strategies and simulations. 
3. Alternative forms of yield curve calibration, including interpolations in different spaces and least-squares fitting for various basis function expansions.
4. Different asset classes, include outright SOFR and Fed Fund interest rate swaps with their corresponding curves. 
5. If time permits, a global solver for interest rate curves.
