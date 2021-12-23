#--------------------------------------------------
# Statistical Hyptohesis Testing
#--------------------------------------------------

# In the classical setup, we have a null hyptothesis (h0), that represents some default position,
# and some alternative hypthesis (H1), that we'd like to compare it with. We will use Statistics to decide whether we
# can reject H0 as false or not.

# Imagine we have a coin and we want to test whether it's fair. We'll make the assumption that the coin has some
# probability p of landing heads, and so our null hypothesis is that the coin is fair - that is, p = 0.5
# We will test this against the alt hypothesis, that p =/= 0.5

# Our test will involve flipping a coin some number of times, n, and counting the number of heads, X.
# Each coin flip is a Bernoulli trian, which means  that X is a binomial(n, p) random variable, which we can approximate
# using the normal distribution

from typing import Tuple
import math

def normal_approx_to_binomial(n: int, p: float) ->Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n,p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

# Whenever a random variable follows a normal dist, we can use NORMAL_CDF to figure out the prob that its realized
# value lies within or outside a particular interval:

from scratch.probability import normal_cdf

# The normal cdf _is_ the probability  the variable is below a threshold

normal_prob_below = normal_cdf

# It;s above the threshold if it's not below the threshold

def normal_prob_above(lo:float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo"""
    return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than lo

def normal_prob_between(lo:float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is between lo and hi"""
    return 1 - normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# it;s outside if it;s not betwen

def normal_prob_outside(lo:float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is not between lo and hi"""
    return 1 - normal_prob_between(lo,hi,mu,sigma)

# We can also do the reverse - find either the nontail region or the interval around the mean that acounts for a certain
# level of likelihood. For ecample, if we want to find an interval centred at the mean and containing 60% probability,
# then we find the cutoff where the upper and lower tails each contain 20% of the probability (leaving 60%)

from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    """Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    """Returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
    """Returns the symetric bounds that contain the specified probability"""
    tail_probability = (1 - probability) / 2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

# In particular, let's say that we choose to flip the coin n = 1000 times. If our hypothesis of fairness is true, X
# should be distributed approx. normally with a mean of 500 and standard deviation of 15.8:

mu_0, sigma_0 = normal_approx_to_binomial(1000, 0.5)

# We need to make a decision about significance - how willing we are to make a type 1 error ("False Pos"), in which
# we reject H0 even though it's true. We will choose 5%.

# Consider the test that rejects H0 if X falls outside the bounds given by:

# (469, 531)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# Assuming p really equals 0.5 (i.e., H0 is true), there is just a 5% chance we observe an X that lies outside this
# interval, which is the exact significance we wanted. Said differently, if H0 is true, then, approx 19 times out of 20,
# this test will give the correct result.

# We are also often interested in the power of a test, which is the probability of making a type 2 error ("false neg"),
# in which we fail to reject H0 even though it's false. In order to measure this, we have to specify what exactly H0
# being false means. In particular, let's check what happens if p is really 0.55, so that the coin is slightly biased
# towards heads.

# in that case, we can calculate the power of the test with:

# 95% bounds based on assumtion on p = 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# actual mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approx_to_binomial(1000, 0.55)

# a type 2 error means we fail to reject the null hypothesis, which will happen when X is still in our original interval

type_2_probability = normal_prob_between(lo,hi,mu_1,sigma_1)
power = 1 - type_2_probability      #0.887

# Imagine instead that our null hypothsis was that the coin is not biased toward heads, or that p<= 0.5.
# In that case we want a one-sided test that rejects the null hypothesis when X is much larger than 500 but not when X
# is smaller than 500. So, a 5% significance test involves using normal_prob_below to find the cutoff below which 95%
# of the probability lies:

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 ( < 531, since we need more probability in the upper tail

type_2_probability = normal_prob_between(hi, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.936

# This is a more powerful test, since it no longer rejects H0 when X is below 469 and instead rejects H0 when X is
# between 526 and 531.