Hi

Please run run.py

It will output three CSV's containing the paths, unhedged and hedged IRRs for all paths. A png for the IRR's will also be outputted

For Q3 and Q5 the output will be in your terminal

Just incase

Q3 Answer - expectation of the maximum of GBPUSD price path: 1.42012

Q5 Answer - European Put Option Value 1.28065x100k = £8048787.28

In terms of the project, we can see that in the hedged portfolio the returns to the upside are capped but downside is limited (lowest return is 9.7% vs 3.8%) this is observed in our histogram where the left tail is cut off. In reducing the downside IRRs it also shifts the distribution to be negatively skewed as well as reducing best case (24.8% vs 26.8%). I've included a notebook containing this analysis.

In terms of the validity of this model, due to the returns being geometrically symmetric this results in an upward drift in prices and hence for more accurate estimation would be to use log-returns 


This project took me 3hrs to complete
