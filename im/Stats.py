num_friends = [100,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,
               10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,
               7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,
               4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

from collections import Counter
import matplotlib.pyplot as plt
from typing import List

friend_counts = Counter(num_friends)
xs = range(101)                         # largest value is 100
ys = [friend_counts[x] for x in xs]     # height is just # of friends
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of Friends")
plt.ylabel("# of People")
plt.show()

# Unfortunately, this chart is still too difficult to slip into conversations. So you start generating some statistics.
# Probably the simplist statistic is the number of data points:

num_points = len(num_friends)       # 204

# You're probably also interested in the largest and smallest values
largest_value = max(num_friends)    # 100
smallest_value = min(num_friends)   # 1

# which are just special cases of wanting to know the values in specific positions:

sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]               # 1
second_smallest_value = sorted_values[1]        # 1
second_largest_value = sorted_values[-2]        # 49

# -------------------------------------------------------------------------------------------------------------------
# Central Tendencies
# -------------------------------------------------------------------------------------------------------------------

# Usually, we'll want some notion of where our data is centered. Most commonly, we'll use the mean (or average),
# Which is just the sum of the data divided by its count

def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs)

mean(num_friends)           # 7.3333

# If you have two data points, the mean is simply the point halfway through them. As you add more points, the mean
# Shifts around, but it always depends on the value of every point. For example if you have 10 data points, and
# you increase the value by 1, you increase the mean by 0.1

# We'll also sometimes be interested in the median, which is the middle-most value or the average of the two middle
# values

# For instance, if we have five data points in a sorted vector x, the median is x[5 // 2] or x[2]. If we have 6 data
# points, we want the average of x[2] and x[3]

# Notice that -unlike the mean -the median doesn't fully depend on every value in your data. For example, if you make
# the largest point larger, the middle points remain unchanged. Which means so does the median

# We'll write different functions for the even and odd cases and combine them

# the underscores indicate that these are "private" functions, as they're intended to be called by our median fucntion
# but not by other people using our stats library

def _median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
    """If len(xs) is even, it's the average of the middle two elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2      # e.g. length 4 => hi_midpoint 2
    return (sorted_xs[hi_midpoint -1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
    """Finds the 'middle-most' value of v"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

assert median([1, 10, 2, 9 ,5]) == 5
assert median([1, 9, 2, 10]) == (2+9)/2

# and now we can compute the median number of friends

print(median(num_friends))      # 6

# A generalisation of the median is the Quantile, which represents the value under which a certain percentile of the
# data lies. Median is the 50th %tile.

def quantile(xs: List[float], p: float) -> float:
    """Returns the pth-percentile value in x"""
    p_index = int(p * len(xs))
    return sorted(xs[p_index])

assert quantile(num_friends, 0.10) == 1
assert quantile(num_friends, 0.25) == 3
assert quantile(num_friends, 0.75) == 9
assert quantile(num_friends, 0.90) == 13

# Less commonly, we might want to look at the mode, or most common values:

def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

assert set(mode(num_friends)) == {1, 6}

# But most frequently, we'll be using the mean

# -------------------------------------------------------------------------------------------------------------------
# Dispersion
# -------------------------------------------------------------------------------------------------------------------

# Dispersion is a measure of how spread out the data is. A simple measure of this is the range. The difference between
# the largest and the smallest value

# "range" already means something in Python, so we'll use a different name

def data_range (xs: List[float]) -> float:
    return max(xs)-min(xs)

assert data_range(num_friends) == 99

# A more complex measure of dispersion is the VARIANCE, which is computed as:

from scratch.linear_algebra import sum_of_squares

def de_mean(xs: List[float]) -> List[float]:
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    assert len(xs >= 2, "variance requires atleast two elements")

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

assert 81.54 < variance(num_friends) < 81.55

# Variance units are squared, it can be hard to make sense of this so we often look instead at STANDARD DEVIATION

import math

def standard_deviation(xs: List[float]) -> float:
    """The standard devaiation is the square root of the variance"""
    return math.sqrt(variance(xs))

assert 9.02 < standard_deviation(num_friends) < 9.04

# Both the range and the STD have the same outlier problem that we saw earlier for the mean. using the same example,
# if our friendliest user had instead 200 friends, the standard deviation would be 14.89 - more than 60% higher.

# A more robust alternative computes the difference between the 75th %tile and the 25th %tile

def interquartile_range(xs: List[float]) -> float:
    """Return the difference between 75th and 25th %tile"""
    return quantile(xs,0.75) - quantile(xs, 0.25)

assert interquartile_range(num_friends) == 6

#--------------------------------------------------------------------------------------------------------------------
# Correlation
#--------------------------------------------------------------------------------------------------------------------

# We'll first look at covariance, the paired analogue of variance. Whereas variance measures how a single variable
# Deviates from its mean, covariance measures how two variables vary in tandem from their means.

daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,
                 48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,
                 23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,
                 36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,
                 22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,
                 24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,
                 35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,
                 33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,
                 27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,
                 22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,
                 35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,
                 9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

daily_hours = [dm / 60 for dm in daily_minutes]

from scratch.linear_algebra import dot

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have the same number of elements"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

assert 22.42 < covariance(num_friends, daily_minutes) < 22.43
assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60

# Recall that DOT sums up the products of coreesponding pairs of elements. When corresponding elements of x and y are
# either both above their means or both below their means, a positive number enters the sum. When one is above and the
# other below, a negative number enters the sum. A large POSITIVE covariance means that x tends to be large when
# y is large, and small when y is small. A covariance close to zero indicates no relationship

# Nonetheless, this number can be hard to interpret for a couple of reasons.

# 1: it's units are the product of the inputs' units, which can be hard to make sense of (friend-minutes-per-day)

# 2: If each user had twice as many friends (but the same number of minutes), the covariance would be twice as large
# but in a sense, the variables would be just as interrelated. Said differently, it's hard to say what counts as a
# large covariance.

# for this reason, it;s more common to look at the correlation, which divides out the standard deviations of both
# variables:

def corr(xs: List[float], ys: List[float]) -> float:
    """measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x /stdev_y
    else:
        return 0    # if no variation, correlation is zero

assert 0.24 < corr(num_friends, daily_minutes) < 0.25
assert 0.24 < corr(num_friends, daily_hours) < 0.25

# The correlation is unitless and always lies between -1 and 1. A number like 0.25 represents a relatively weak
# Positive correlation. However, one thing we neglected to do was examine our data.

# The person with 100 friends (who spends 1 min a day on the site) is a huge outlier, and correlation can be very
# sensitive to outliers. What happens if we ignore him?

outlier = num_friends.index(100)    # index of the outlier

num_friends_good = [x
                    for i, x in enumerate(num_friends)
                    if i != outlier]

daily_minutes_good = [x
                      for i, x in enumerate(daily_minutes)
                      if i !=outlier]

daily_hours_good = [dm / 60 for dm in daily_minutes_good]

assert 0.57 < corr(num_friends_good, daily_minutes_good) < 0.58
assert 0.57 < corr(num_friends_good, daily_hours_good) < 0.58

#---------------------------------------------------------------------------------------------------------------------
# Simpson's Paradox
#---------------------------------------------------------------------------------------------------------------------

# One not uncommon surprise when analysing data is Simpson's Paradox, in which correlations can be misleading when
# confounding variables are ignored.

# For example, imagine that you can identify all of your members as either East or West Coast Data Scienctists.
# You decide to examine which coast's data scientists are friendlier. Turns out West Coast have more friends on avg.
# However, when you look at only people with PhDs, the East Coasters have more friends.
# Also, if you look at only people without PhDs, the East Coasters also have more friends on average.

# once you account for the users' degrees, the correlation goes in the opposite direction. Bucketing the data as East/
# West Coast disguises the fact that the East Coast DS skew much more heavily toward PhD types.

