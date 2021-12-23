import enum, random

# An Enum is a typed set of enumerate values. We can use them to make our code more descritpive and readable

class Kid (enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL:
        older_girl +=1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls +=1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl +=1

print("P(both I older):", both_girls / older_girl)
print("P(both I older):", both_girls / either_girl)

# the density function for the uniform distribution is just:

def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x < 1 else 0

# The probability that a random variable following that distribution is between 0.2 and 0.3 is 1/10.
# Python's random.random is a random variable with a uniform density

# We'll often be more interested in the CDF, which gives the probability that a random variable is less than or equal
# to a certain value

def uniform_cdf(x: float) -> float:
    """Returns the prob that a unitform random variable is <=x"""
    if x < 0:   return 0          # Uniform radom is never less than 0
    elif x < 1: return x          # e.g. P(X <= 0.4) = 0.4
    else:       return 1          # uniform random is always less than 1

#-----------------------------------------------------------------------
# Normal Dist
#-----------------------------------------------------------------------

# The normal dist is the classic bell curve shape. It is determined by its mean (mu) and its Standard Deviation (Sigma)
# The mean indicates where the bell is centred and the Std, how wide it is.

import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mu) ** 2 / 2 /sigma ** 2) / (SQRT_TWO_PI * sigma))

import matplotlib.pyplot as plt
xs = [x / 10.0 for x in range (-50,50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs], "-", label="mu=0,sigma=1")
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs], "--", label="mu=0,sigma=2")
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs], ":", label="mu=0,sigma=0.5")
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs], ":", label="mu=-1,sigma=1")
plt.legend()
plt.title("Various Normal PDFs")
plt.show()


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4) # bottom right
plt.title("Various Normal cdfs")
plt.show()


#-----------------------------------------------------------------------
# Central Limit Theorem
#-----------------------------------------------------------------------

# CLT postulates that a random variable defined as the average of a large number of independent and identically
# distributed random variables is itself approximately normally distributed.

# An easy way to isllustrate this is by looking at binomial random variables, which have two parameters n and p.
# A Binomial (n,p) random varaible is simply the sum of n independant random variables, each of which equals 1
# with probability p and 0 with probability 1 - p:

def bernoulli_trial(p: float) -> int:
    """Returns 1 with probability p and 0 with probability 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Returns the sum of n bernoulli(p) trials"""
    return sum(bernoulli_trial(p) for _ in range(n))

from collections import Counter

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from a Binomial(n,p) and plots their histograms"""
    data = [binomial(n, p) for _ in range(num_points)]

# use a bar chart to show the actual binomial samples

    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
        [v /num_points for v in histogram.values()],
        0.8,
        color = "0.75")
    mu = p * n
    sigma = math.sqrt(n * p * (1-p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
          for i in xs]
    plt.plot(xs,ys)
    plt.title("Binomial Dist. vs Normal Approx.")
    plt.show()

binomial_histogram(0.75, 100, 10000)