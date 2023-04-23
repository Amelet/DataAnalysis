import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.optimize import curve_fit


def draw_sample_from_chisquare(sample_size):
    samples = (np.random.chisquare(df, sample_size))
    smpl_mean = (np.mean(samples))
    return smpl_mean


def gaus(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def find_stats(collected_samples):
    count, bin_edges = np.histogram(
        collected_samples,
        bins=50,
        density=True)
    bin_middles = np.round(0.5 * (bin_edges[1:] + bin_edges[:-1]), 0)
    x = bin_middles
    y = count
    mean = sum(x * y) / sum(y)  # estimate mean of a histogram
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y)) # estimate sigma
    peak = max(count) # find peak
    return x, y, mean, sigma, peak


print("""
Using central limit theorem evaluate a mean of a distribution.
Distribution to evaluate: chi square.
Project is done in steps:
1. Plot chi square distribution
2. Sample N-values from the chi square distribution (N is chosen by user) and take the average of this sample
3. Repeat step 2 1000-times and plot histogram of these values
4. Fit Gaussian distribution to the histogram
5. Find mean, stdev of fitted Gaussian
6. Compare mean of Gaussian and the mean of the chi square distribution
6. Learn how sample size (N) affects the mean approximation of of the original distribution""")


# Choose the sample size
while True:
    try:
        users_input = input("Type sample size (a single digit):"
                            "\n\tfor example: 50"
                            "\nType here:   ")
        users_input = int(users_input)
        break
    except ValueError:
        print("Please, type a single digit")


# We will estimate mean of this distribution: chi square
# -> set it up

df = 55 # set the shape of the chi square distribution
mean_chi2, var, skew, kurt = chi2.stats(df, moments='mvsk') # get stats for chisquare
stdev_chi2 = (2*mean_chi2)**0.5

# -> get its stats

print(f"""
Statistics for the chi square distribution:
mean of chi2 = {mean_chi2}
variance of chi2 = {var}
stdev of chi2 = {(2 * mean_chi2) ** 0.5}""")

# -> find the range of x-values it takes (used in plotting)
samples = (np.random.chisquare(df, 1000)) # to do it I draw 1000 samples from chi2 distribution
count, bins_chi = np.histogram(samples, bins=50, density=True) # find bins and counts
# -> get y values for the range of x-values
chi2_y = chi2.pdf(bins_chi, df)
chi2_y_max = max(chi2_y)


# Central Limit Theorem
# -> Draw samples from chi square  **iter_n**  times
iter_n = 1000
collected_samples = []
for iteration in range(iter_n):
    smpl_avg = draw_sample_from_chisquare(users_input)
    collected_samples.append(smpl_avg)

# -> fit collected samples with a Gaussian
x, y, mean, sigma, peak = find_stats(collected_samples)
popt, pcov = curve_fit(gaus, x, y, p0=[peak, mean, sigma])
fitted_gaussian_y = gaus(bins_chi, *popt)
gaus_y_max = max(fitted_gaussian_y)


fig = plt.figure()
# plot chi2 distribution and Gaussian
plt.plot(bins_chi, chi2_y/chi2_y_max, linewidth=2, color='r', label='Chi2 distribution')
plt.plot(bins_chi, fitted_gaussian_y/gaus_y_max, color='b', label='Gaussian distribution')

# show mean of both distributions
plt.vlines(x=mean_chi2, ymin=0, ymax=1, linestyles="dashed", label="chi2 mean = " + str(mean_chi2), color="r")
plt.vlines(x=np.round(popt[1],2), ymin=0, ymax=1, linestyles="dashed", label="CLT mean = " + str(np.round(popt[1], 2)), color="b")

plt.legend()
plt.xlabel('bins', fontsize=15)
plt.ylabel('PDF normalized', fontsize=15)
plt.title(f'CLT applied to estimate mean of Chi2 distribution\n{iter_n} samples of size {users_input}', fontsize=15)
plt.grid(b=True, color='grey', alpha=0.3, linestyle='-.', linewidth=2)
plt.rcParams['figure.figsize'] = [8, 8]
plt.show()
