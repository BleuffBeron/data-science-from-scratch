# Functions in Python
# Functions are a rule for taking zero or more inputs and returning a corresponding output.
# In python, we typically define functions using def

def double(x):
    """
    This is where you put an optional docstring that explains what the
    function does. For example, this function multiplies its input by 2.
    """
    return x * 2
# Python functions are first-class, which means that we can assign them to variables and pass them into functions
# just like any other arguments

def apply_to_one(f):
    """
    Calls the function F with 1 as its argument
    """
    return f(1)
my_double = double                     # Refers to the previously defined function
x = apply_to_one (my_double)           # equals 2

# Its also easy to create short anonymous founctions or lambdas

y = apply_to_one (lambda x: x + 4)

# Function parameters can also be given default arguments
# which only need to be specified when you want a value

def my_print(message = "my default message"):
    print(message)

my_print("hello")                     # prints hello
my_print()                            # prints default msg

def full_name(first = "What's his name", last = "something"):
    return first + " " + last

print(full_name())

# -----------------------------------------------------------------------------------------------------------------
# Strings
# Strings can be delimited by single or double quotation marks

single_quotes = 'Data'
double_quotes = "Data"

# You can create multiline strings using 3 double quotes
multi_liner = """this is the first line.
second
third"""

# Different ways to join strings
first = "aaron"
last = "oconnell"
full_name1 = first + " " + last            # string addition
full_name2 = "{0} {1}".format(first,last)  # string.format

# But f-string is the easiest
full_name3 = f"{first} {last}"

#-------------------------------------------------------------------------------------------------------------------
# Exceptions
# When something goes wrong, Python raises an exceptopm. Unhandled, exceptions will cause your program to crash
# You can handle them using try and except

try:
    print(0/0)
except ZeroDivisionError:
    print("cannot divide by zero, retard")

#--------------------------------------------------------------------------------------------------------------------
# Lists
# Probably the most fundamental data structure in Python - this is simply an ordered collection.

integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list,heterogeneous_list,[]]

list_length = len(integer_list)     # equals 3
list_sum = sum(integer_list)        # equals 6

# You can get or set the nth element of a list with square brackets:

x = [0,1,2,3,4,5,6,7,8,9]

zero = x[0]     # equals 0, lists are 0-indexed
one = x[1]      # equals 1
nine = x[-1]    # equals 9, 'Pythonic' for last element
eight = x[-2]   # equals 8, 'Pythonic' for second to last element
x[0] = -1       # now x set starts with -1 instead of 0

# You can also use square brackets to slice lists. the slice i:j means all elements from i (inclusive) to j
# (not inclusive). If you leave off the start of a slice, you'll slice froim the beginning of the list,
# and if you leave off the end of the slice, you'll slice until the end of the list

first_three = x[:3]
three_to_end = x[3:]
one_to_four = x[1:5]
last_three = x[-3:]
without_first_and_last = x[1:-1]

# You can similarly slice strings and other "sequential" types.
# A slice can take a third argument to indicate its stride, which can be negative

every_third = x[::3]
five_to_three = x[5:2:-1]

# it's easy to concatenate lists together. If you want to modify a list in place, you can use
# extend to add items from another collection.

x = [1,2,3]
x.extend([4,5,6])   # x is now 1,2,3,4,5,6

# if you don't want to modify x, you can use list addition:

x = [1,2,3]
y = x + [4,5,6]

# more frequently, we will append to lists one item at a time

x = [1,2,3]
x.append(0)     # x is no2 1,2,3,0
y = x[-1]       # = 0
z = len(x)      # = 4

# It's often convienent to unpack lists when you know how many elements they contain

x, y = [1, 2]   # now x is 1 and y is 2

#-------------------------------------------------------------------------------------------------------------
# Tuples
# Tuples are like lists. pretty much anything you can do to a list that doesnt involve modifying it,
# you can do to a tuple. You specify a tuple by using parentheses (or nothing) instead of square brackets

my_list = [1,2]
my_tuple = (1,2)
other_tuple = 3,4
my_list[1] = 3      # my list is now 1, 3

# Tuples are a convenient way to return multiple values from functions

def sum_and_product(x, y):
    return (x + y), (x * y)
sp = sum_and_product(2,3)       # SP is (5, 6)
s, p = sum_and_product(5, 10)   # s is 15, p is 50

# Tuples (and lists) can also be used for multiple assignments

x, y = 1, 2     # now x is 1, y is 2
x, y = y, x     # now x is 2, y is 1

#-----------------------------------------------------------------------------------------------------------------
# Dictionaries
# Another fundamental data structure is a dictionary, which associates values with keys and allows you to quickly
# retrieve the value corresponding to a given key:

empty_dict = {}         # Pythonic
empty_dict2 = dict()    # less Pythonic
grades = {"Joel": 80, "Tim": 95}    # dictionary literal

# You can look up the value for a key using square brackets

joels_grade = grades["Joel"]    # = 80

# You can check for the existence of a key using in:

joel_has_grade = "Joel" in grades   # true
kate_has_grade = "Kate" in grades   # False

# This membership check is fast even for large dictionaries

# Dictionaries have a get method that returns a default value (instead of raising an exception) when you look up a key
# that's not in the dictionary

joels_grade = grades.get("Joel", 0)     # = 80
kates_grade = grades.get("Kate", 0)     # = 0
no_ones_grade = grades.get("no one")    # default is none

# you can assign key/value pairs using the same square brackets

grades["Tim"] = 99      # Replaces the old value of 95
grades["Kate"] = 0      # = 0
num_students = len(grades)  # = 3

# you can use dictionaries to represent structured data:

tweet = {
    "User": "Aaron",
    "Text": "Boopity Bop",
    "Retweet_count": 100
}

# Although we'll soon seea better approach
# Besides looking for the specific keys, we can look at all of them

tweet_keys = tweet.keys()   # Iterable for the keys
tweet_values = tweet.values()   # Iterable for the values
tweet_items = tweet.items()   # Iterable for the (key, value) tuples

"user" in tweet_keys        # True, but not Pythonic
"user" in tweet             # Pythonic way of checking for keys
"Aaron" in tweet_values     # True

#---------------------------------------------------------------------------------------------------------------
# Defaultdict
# Imagine we are trying to count the words in a document. An obvious approach would be to create a dictionary in which
# the keys are words and the values are counts. As you check each word, you can increment its count if its already
# in the dictionary and add it to the dictionary if its not:

document = ["data", "data", "science", "from", "from", "from", "scratch"]

word_count = {}
for word in document:
    if word in word_count:
        word_count[word] +=1
    else:
        word_count[word] = 1

# you could also use the "forgiveness is better than permission" approach and just handle the exception from trying
# to look up a missing key

word_count = {}
for word in document:
    try:
        word_count[word] += 1
    except KeyError:
        word_count[word] = 1

# a third approach is to use get, which behaves gracefullt for missing keys:

word_count = {}
for word in document:
    previous_count = word_count.get(word, 0)
    word_count[word] = previous_count + 1

# Theese are a bit unwieldy. Which is why defaultdict is useful. It's like a regular dictionary, except when you try
# look up a key it doesn't contain, it first adds a value for it using a zero-argument function you provided when
# you created it

from collections import defaultdict

word_count = defaultdict(int)       # int() produces 0
for word in document:
    word_count[word] +=1

# They can also be useful with list or dict, or even functions

dd_list = defaultdict(list)     # list() produces and empty list
dd_list[2].append(1)            # now dd_list contains {2: [1]}

dd_dict = defaultdict(dict)     # dict() produces an empty dict
dd_dict["Aaron"]["City"] = "Dublin"     # {"Aaron": {"City": Dublin}}

dd_pair = defaultdict(lambda: [0,0])
dd_pair[2][1]

#---------------------------------------------------------------------------------------------------------------------
# Counters
# A counter turns a sequesnce of values into a defaultdict(int)-likeobject mapping keys to counts

from collections import Counter
c = Counter([0,1,2,0])      # c is basically {0: 2, 1: 1, 2: 1}

# this gives us very simple ways to solve our wordcount probelm:

word_count = Counter(document)

# A counter instance has a most_common method that is frequently useful:

for word, count in word_count.most_common(10):
    print(word, count)

#--------------------------------------------------------------------------------------------------------------------
# Sets
# Another useful data structure is set, which represents a collection of distinct elements.
# You can define a set by listing its elements between curly braces:

primes_below_10 = {2,3,5,7}

# However, that doesn't work for empty sets, as {} already means "empty dict". In that case, you'll need to use set()

s = set()
s.add(1)        # s is now {1}
s.add(2)        # s is now {1, 2}
s.add(2)        # s is still {1, 2}
x = len(s)      # equals 2
y = 2 in s      # equals true
z = 3 in s      # equals false

# We'll use sets for two main reasons. The first is that "in" is a very fast operation on sets.
# if we have a large collection of items that we want to use for a membership test, a set is more appropriate than a
# list

hundreds_of_other_words = []
stopwords_list = ["a", "an", "at"] + hundreds_of_other_words + ["yet", "you"]

"zip" in stopwords_list     # False, but we have to check every element

stopwords_set = set(stopwords_list)
"zip" in stopwords_set      # very fast to check

# The second reason is to find the distict items in a collection:

item_list = [1,2,3,1,2,3]
num_items = len(item_list)              # 6
item_set = set(item_list)               # {1, 2, 3}
num_distinct_items = len(item_set)      # 3
distinct_items_list = list(item_set)    # [1, 2, 3]

# We'll use sets less frequesntly than dictionaries and lists.

#----------------------------------------------------------------------------------------------------------------------
# Control Flow
# As in most programming languages, you can perform an action conditionally using IF:

if 1 > 2:
    message = "if only 1 were greater than two..."
elif 1 > 3:
    message = "elif stands for else if"
else:
    message = "When all else fails, use else"

# You can also write a ternary if-then-else on one line, which we will do occasionally:

parity = "even" if x % 2 == 0 else "odd"

# Python has a while loop:

x = 0
while x < 10:
    print(f"{x} is less than 10")
    x += 1

# Although more often we'll use for and in:
# range(10) is the numbers 0, 1, ... , 9

for x in range(10):
    print(f"{x} is less than 10")

# if you need more complext logic, you can use continue and break:

for x in range(10):
    if x == 3:
        continue    # go immediately to the next iteration
    if x == 5:
        break       # quit the loop entirely
    print(x)
# this prints 0,1,2 and 4
#--------------------------------------------------------------------------------------------------------------------
# Truthiness
# Booleans in Python work as in most other languages, except that they're capatalized:

one_is_less_than_two = 1 < 2        # equals True
true_equals_false = True == False   # equals False

# Python lets you use the value None to indicate a nonexistant value. It is similar to other languages' null

x = None
assert x == None    # This is not the Pythonic way to check for None
assert x is None    # This is the Pythonic way to check for none

"""Python lets you use any value where it expects a Boolean. The following are all "falsy"
False
None
[] (empty list)
{} (empty dict)
""
set()
0
0.0
"""
# Pretty much anything else gets treated as True, This allows you to easily use if statements to test for empty lists,
# empty strings, dictionaries and so on. It also sometimes causes tricky bugs if you're not expecting this behaviour

def some_function_that_returns_a_string():
    return ""
s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = ""

# A shorter (but possibly more confusing) way of doing the same is:

first_char = s and s[0]

# since AND returns its second value when the first is "truthy", and the first value when it's not.
# Similarly, if x is either a number or possibly None:

safe_x = x or 0

# is definitly a number, although:

safe_x = x if x is not None else 0

# is possibly more readable

# Python has an ALL function, which takes an iterable and returns True precisely when every element is truthy, and an
# ANY function, which returns True when atleast one element is true

all([True, 1, {3}])     # True, all are truthy
all([True, 1, {}])      # False, {} is false
any([True, 1, {}])      # True, True is truthy
all([])                 # True,no falsy elements in the list
any([])                 # False, no truthy elements in the list
#-------------------------------------------------------------------------------------------------------------------
# Sorting
# Every Python list has a SORT method that sorts it in place. If you don't want to mess up your list, you can use the
# SORTED function, which returns a new list

x = [4,1,2,3]
y = sorted(x)      # y is now [1,2,3,4]
x.sort()           # x is now [1,2,3,4]

# By default, SORT (and SORTED) sort a list from the smallest to largest based on naively comparing the elements
# If you want the elements sorted from largest to smallest, you can specify a REVERSE = TRUE parameter
# And instead of comparing the elements themselves, you can compare the results of a fucntion that you specify with a
# Key

# Sort the list by absolute value from largest to smallest (absolute = turns - to +
x = sorted([-4, 1, -2, 3], key=abs, reverse=True)

# Sort the words and counts from highest count to lowest
wc = sorted(word_count.items(),
    key=lambda word_and_count: word_and_count[1],
    reverse=True)
#----------------------------------------------------------------------------------------------------------------------
# List Comprehensions
# Frequently, you'll want to transform a list into another list by choosing only certain elements, by transforming
# Elements, or both. The Pythonic way to do this is with LIST COMPREHENSIONS

even_numbers = [x for x in range(5) if x % 2 == 0]     # [0, 2, 4]
squares = [x * x for x in range(5)]                    # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers]           # [0, 4, 16]

# You can similarly turn lists into dictionaries or sets:
square_dict = {x: x * x for x in range(5) if x % 2 == 0}    # {0: 0, 1: 2, 2: 4...4: 16}
square_set = {x * x for x in [1, 2, 3]}                     # {1, 4, 9}

# If you don't need the value from the list, it's common to use an underscore as the variable:
zeros = [0 for _ in even_numbers]                   # Has the same length as even_numbers

# A list comprehension can include multiple FORs:

pairs = [(x, y)
         for x in range(10)
         for y in range(10)]

# and later FORs can use the results of earlier ones:
increasing_pairs = [(x, y)                      # only pairs with x < y
                    for x in range(10)          # range(lo, hi) equals
                    for y in range(x + 1, 10)]  # [lo, lo + 1,..., hi - 1]
print(increasing_pairs)

#---------------------------------------------------------------------------------------------------------------------
# Automated testing and assert
# As data scientists, we'll be writing lots of code. How can we be confident our code is correct?
# One ways is with TYPES (discussed shortly), but another is with AUTOMATED TESTS.
# There are elaborate frameworks for writing and running tests, but in this book we'll just use ASSERT
# This will cause code to raise an ASSERTIONERROR if condition is not truthy:

assert 1+1 == 2

assert 1+1 == 2, "1+1 should equal 2 but it didn't"

# As you can see, you can add an optional message to be printed if the assertation fails.
# It's not particularly interesting to assert that 1+1 = 2. What's more interesting is to assert that functions
# you write are doing what you expect them to:

def smallest_item(xs):
    return min(xs)
assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, -1, 5, 40]) == -1

# Another less common use is to assert things about inputs to functions:

def smallest_item(xs):
    assert xs, "empty list has no smallest item"
    return mins (xs)

#----------------------------------------------------------------------------------------------------------------------
# Object Oriented Programming
# Like many other languages, Python allows you to define CLASSES that encapsulate data and the functions that operate
# on them. We'll use them sometimes to make our code cleaner and simpler.

# Here we'll construct a CLASS representing a couning clicker. It maintains a COUNT, can be CLICKED to increment the
# Count, allows you to READ_COUNT and can be RESET back to zero
# To define a class, you use the CLASS keyword and a PascalCase name:

class CountingClicker:
    """A class can/should have a docstring, just like a function"""
    def __init__(self, count = 0):
        self.count = count

# another such method is __repr__, which produces the string representation of a class instance

    def __repr__(self):
        return f"CountingClicker(count={self.count}"
# finally, we need to implement the public API of our class:

    def click(self, num_times = 1):
        """Click the clicker some number of times."""
        self.count += num_times

    def read(self):
        return self.count

    def reset(self):
        self.count = 0

# Having defined it, let's use ASSERT to write some test cases for our clicker

clicker = CountingClicker()
assert clicker.read() == 0, "Clicker should start with count 0"
clicker.click()
clicker.click()
assert clicker.read() == 2, "After 2 clicks, should count 2"
clicker.reset()
assert clicker.read() == 0, "after reset, should be zero"

# We'll occasionally create SUBCLASSES that INHERIT some of their functionality from a parent class.
# For example, we could create a non-resetable clicker by using CountingClicker as the base class and overriding the
# RESET method to do nothing

class NoResetClicker(CountingClicker):
    # This class has all the same methods as CountingClicker
    # Except that it has a reset method that does nothing
    def reset(self):
        pass
clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read() == 1
clicker2.reset()
assert clicker2.read() == 1, "reset should do nothing"

#----------------------------------------------------------------------------------------------------------------------
# Iterators and Generators
# One nice thing about a list is that you can retrieve specific elements by their indices. But you don't always need to!
# A list of a billion numbers takes up alot of memory. If you only want the elements one at a time, there's no good
# reason to keep them all around. If you only end up needing the first several elements, generating the entire billion
# is wasteful

# Often all we need is to iterate over the collection using For and In. In this case, we can create generators, which
# can be iterated over just like lists but generate their values lazily on deman

# one way to create generators is with fucntion and the Yield operator

def generator_range(n):
    i = 0
    while i < n:
        yield i     # every call to yield produces a value of the generator
        i += 1
# the following loop will consume the YIELDED values one at a time until none are left:#

for i in generator_range(10):
    print(f"i:{i}")

# In fact, the range is itself lazy, so there's no point doing this

# With a generator, you can even create am infinite sequesnce

def natural_numbers():
    """returns 1, 2, 3, ..."""
    n = 1
    while True:
        yield n
        n += 1

# A second way to create generators is by using FOR comprehensions wrapped in parentheses

evens_below_20 = (i for i in generator_range(20) if i % 2 == 0)

# None of these computations *does* anything until we iterate

data = natural_numbers()
evens = (x for x in data if x % 2 == 0)
even_squares = (x ** 2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x % 10 == 6)

# Not infrequently, when we're iterating over a list or generator, we'll want not just the values, but also their
# indices. For this common case, Python provides an ENUMERATE function, which turns values into pairs (INDEX, VALUE);

names = ["Alice", "Bob", "Charlie", "Debbie"]
# not Pythonic
for i in range(len(names)):
     print(f"name {i} is {names[i]}")

print("-" * 5)      # just using this as a print spacer

# Also not Pythonic
i = 0
for name in names:
    print(f"name {i} is {names[i]}")
    i += 1

print("-" * 5)      # just using this as a print spacer

# Pythonic
for i, name in enumerate(names):
    print(f"name {i} is {name}")

#------------------------------------------------------------------------------------------------------------------
# Randomness
# As we learn data science, we will frequesntly need to generate random numbers, which we can do with the RANDOM module

import random
random.seed(10) # this ensures we get the same results every time

four_uniform_randoms = [random.random() for _ in range(4)]

# random.random() produces numbers uniformly between 0 and 1. We'll us this most often

# The random module actually produces pseudorandom (that is, deterministic) numbers based on an internal state that you
# can set with random.seed if you want to get reproducable results
# We'll sometimes use random.randrange which takes either one or two arguments and returns an element chosen randomly
# from the corresponding range:

random.randrange(10)        # Choose randomly from range(10) = [0, 1,..., 9]
random.randrange(3, 6)      # ""       ""      ""    "" (3, 5)

# There are a few more methods that we;ll sometimes find convenient. For example, Random.shuffle randomly reorders
# the elements of the list
up_to_ten = [1,2,3,4,5,6,7,8,9,10]
random.shuffle(up_to_ten)
print(up_to_ten)

# If you need to randomly pick one element from a list, you can use RANDOM.CHOICE:

my_best_friend = random.choice(["Alice", "Bob", "Charlie"])

# And if you need to randomly choose a sample of elements without replacement, you can use RANDOM.SAMPLE:

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)

# or with replacement:

four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)

#-------------------------------------------------------------------------------------------------------------------
# Regular Expressions
# Provides a way of searching text. They are incredibly useful, but also fairly complicated, so much so that there are
# entire books written about them. We will get into their details the few times we encounter them:

import re

re_examples = [
    not re.match("a", "cat"),
    re.search("a", "cat"),
    not re.search("c", "dog"),
    3 == len(re.split("[ab]", "carbs")),
    "R-D-" == re.sub("[0-9]", "-", "R2D2")
]
#---------------------------------------------------------------------------------------------------------------------
# Zip and Argument Unpacking
# Often we will need to ZIP two or more iterables together. The ZIP function transforms multiple iterables into a single
# iterable of tuples of corresponding fucntion:

list1 = ["a", "b", "c"]
list2 = [1,2,3]

# zip is lazy, so you have to do something like the following

[pair for pair in zip(list1, list2)]        # is [(a, 1), (b, 2), (c, 3)]

# you can also "unzip" a list using a strange trick

pairs = [("a", 1), ("b", 2), ("c", 3)]
letters, numbers = zip(*pairs)

# The * performs argument unpacking, which uses the elemts of pairs as individual arguments to ZIP. It ends up the same
# as if you'd called:

letters, numbers = zip(("a", 1), ("b", 2), ("c", 3))

# You can use argument unpacking with any function

def add(a, b): return a+b

add(1, 2)
try:
    add([1,2])
except TypeError:
    print("add expects two inputs")
add(*[1,2])

#---------------------------------------------------------------------------------------------------------------------
# args and kwargs
# Let's say we want to create a higher-order function that takes as input some function F and returns a new function
# that for any input returns twice the value of F:

def doubler(f):
    # Here we define a new function that keeps a reference to f
    def g(x):
        return 2 * f(x)

    # And return that new function.
    return g

def f1(x):
    return x + 1

g = doubler(f1)
assert g(3) == 8,  "(3 + 1) * 2 should equal 8"
assert g(-1) == 0, "(-1 + 1) * 2 should equal 0"

def f2(x, y):
    return x + y

g = doubler(f2)
try:
    g(1, 2)
except TypeError:
    print("as defined, g only takes one argument")

def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)

magic(1, 2, key="word", key2="word2")

# prints
#  unnamed args: (1, 2)
#  keyword args: {'key': 'word', 'key2': 'word2'}

def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = {"z": 3}
assert other_way_magic(*x_y_list, **z_dict) == 6, "1 + 2 + 3 should be 6"

def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        """whatever arguments g is supplied, pass them through to f"""
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
assert g(1, 2) == 6, "doubler should work now"