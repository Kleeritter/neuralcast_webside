import itertools

forecast_horizonts=[2,4,6,12,15,18,24,32,48,60,72,84,96,192]
window_sizes=[16*24*7,8*7*24,4*7*24,2*7*24,7*24,6*24,5*24,4*24,3*24,2*24,24,12,6,3]

permutations = list(itertools.product(forecast_horizonts, window_sizes))

print(permutations)