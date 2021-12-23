from matplotlib import pyplot as plt

years = [1950,1960,1970,1980,1990,2000,2010]
gdp = [300.2,543.3,1075.9,2862.4,5989.6,10234.7,14900.3]
#=---------------------------------------------------------------------------------------------------------------

"""Line Charts"""
# Create a line chart, years on the x and gdp on the y
plt.plot(years,gdp,color = "green", marker ="o", linestyle = "solid")

# Add a title
plt.title("Nominal GDP")

# add a label to the y-axis
plt.ylabel("Billions of $")
plt.show()

#----------------------------------------------------------------------------------------------------------------
"""Bar Charts"""

movies = ["Annie Hall","Ben Hur","Casablanca","Gandhi","WSS"]
num_oscars = [5,11,3,8,10]

# Plot bars with left x-coords [0,1,2,3,4], heights [num_oscars]
plt.bar(range(len(movies)), num_oscars)

plt.title("Some Movies")
plt.ylabel("N of Awards")

# label x-axis with movie names at bar centres
plt.xticks(range(len(movies)), movies)

plt.show()

# Bar charts can also be used to visually explore distribution

from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# Buckect Grades by decile, but put 100 in with the 90s

histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],  # Shift bars right by 5
        histogram.values(),                 # give each bar its correct height
        10,                                 # gives each bar width of 10
        edgecolor=(0,0,0))                  # Black edges for each bar

plt.axis([-5,105,0,5])                      # x from -5 to 105
                                            # y from 0 to 5

plt.xticks([10 * i for i in range(11)])         # x labels at 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("N of Students")
plt.title("Dist of Grades")
plt.show()

#------------------------------------------------------------------------------------------------------------------

# Line charts
# As we already saw, we can make line charts using plt.plot. These are a good choice for showing trends.

variance = [1,2,4,8,16,32,64,128,256]
bias_squared = sorted(variance, reverse=True)
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# We can make multiple calls to plt.plot to show multiple series on the same chart

plt.plot(xs, variance,             "g-",    label="Variance")
plt.plot(xs, bias_squared,         "r-.",    label="Bias^2")
plt.plot(xs, total_error,          "b:",    label="Total Error")

# Because we've assigned lables to each series, we can get a legend for free (loc=9 means top centre)

plt.legend(loc=9)
plt.xlabel("Model Complexity")
plt.xticks([])
plt.title("The Bias-Variance Tradeoff")
plt.show()

#--------------------------------------------------------------------------------------------------------------------
# Scatterplots

friends = [ 70,  65,  72,  63,  71,  64,  60,  64,  67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# Label each point
for label, friend_count, minute_count in zip(labels,friends,minutes):
        plt.annotate(label,
                     xy=(friend_count, minute_count),   # put the label with its point
                     xytext=(5,-5),                     # but slighly offest
                     textcoords="offset points")

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of Friends")
plt.ylabel("Daily Minutes Spent on the site")
plt.show()