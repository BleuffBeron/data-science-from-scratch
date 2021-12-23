from typing import List

Vector = List[float]

height_weight_age = [70,  # inches,
                     170, # pounds,
                     40 ] # years

grades = [95,   # exam1
          80,   # exam2
          75,   # exam3
          62 ]  # exam4

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def subtract(v: Vector, w: Vector) -> Vector:
    """subtracts corresponding elements"""
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5,7,9], [4,5,6]) == [1,2,3]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check the list is not empty
    assert vectors, "No vectors provided!"

    # Check the vectors are the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

# We'll also need to be able to multiply a vector by a scalar, which we do simply by multiplyinh each element of the
# vector by that number

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1,2,3]) == [2,4,6]

# This allows us to compute the componentwise means of a list of (same -sized) vectors:

def vector_mean (vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1,2], [3,4], [5,6]]) == [3,4]

# A less obvious tool is the DOT PRODUCT. The dot product of two vectors is the sum of their componentwise product

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "Vectors must be the same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1,2,3], [4,5,6]) == 32  # 1*4 + 2*5 + 3*6

# If W has magnitude 1, the dot product measures how far the vector V extends in the W direction.
# For example, if w = [1, 0], then dot [v, w] is just the first component of v.
# Another way of saying this is that it's the length of the vector you'd get if you projected v onto w

# Using this, it's easy to compute a vector's sum of squares:



def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v,v)

assert sum_of_squares([1,2,3]) == 14    # 1*1 + 2*2 + 3*3

# Which we can use to compute its MAGNITIUDE (or length):

import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude or length of v"""
    return math.sqrt(sum_of_squares(v))         # math.sqrt is square root fucntion

assert magnitude([3,4]) == 5

# We now have all the pieces we need to compute the distance between two vectors.

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v,w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))

# This is possibly clearer if we write it as (the equivalent):

def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))

#---------------------------------------------------------------------------------------------------------------------
# Matrices
#----------------------------------------------------------------------------------------------------------------------

# A matrix is a two-dimensional collection of numbers. We will represent matrices as lists of lists, with each inner
# list having the same size and representing a ROW of the matrix. If A is a matrix, then A[i][j] is the element in the
# iTH row and the jTH column. Per mathemathical convention, we will frequently use capital letters to represent
# matrices. For example:

# Another type alias

Matrix = List[List[float]]

A = [[1,2,3],   # A has 2 rows and 3 columns
     [4,5,6]]

B = [[1,2],     # B has 3 rows and 2 columns
     [3,4],
     [5,6]]

# Given this list-of-lists representation, the matrix A has len(A) rows and len(A[0]) column, which we consider its
# SHAPE

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0    # number of elements in first row
    return num_rows, num_cols

assert shape([[1,2,3], [4,5,6]]) == (2,3)   # 2 rows, 3 columns

# If a matrix has n rows and k columns, we will refer to it as an n x k matrix. We can think of each row of an n x k
# matrix as a vector of length k and each column as a vector of length n:

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a vector)"""
    return A[i]     # A[i] is already the ith row

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a vector)"""
    return [A_i[j]          # j-th element of row A_i
            for A_i in A]   # for each row in A_i

# We'll also want to be able to create a matrix given its shape and a fucntion for generalizing its elements.
# We can do this using a nested list comprehension

from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int,int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i, j) -th entry is entry_fn(i, j)
    """
    return [[entry_fn(i,j)                  # given i, create a list
             for j in range(num_cols)]      # [entry_fn(i, 0), ...]
            for i in range(num_rows)]       # create one list for each i

# Given this function, you could make a 5x5 identity matrix (with 1s on the diagonal and 0s elsewhere, like so:

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

print(identity_matrix(5))

# Matricies will be important for several reasons. First, we can use a matrix to represent a dataset consisting of
# multiple vectors, simply by considering each vector as a row of the matrix. For example, if you had the
# heights, weights and ages of 1,000 people, you could put them in a 1,000 x 3 matrix

data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19],
        # ....
       ]
# Second, as we'll see later, we can use an n x k matrix to represent a linear function that maps k-dimentional
# vectors to n-dimentional vectors, Several of our techniques and concepts will involve such functions.

# Third, matrices can be used to represent binary relationships. In chapter1, we represented the edgest of a network
# as a colection of pairs (i, j). An alternative would be to create a matrix A such that A[i][j] is 1 if nodes i and j
# are connected and 0 otherwise

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#            also represented as:
#            user 0  1  2  3  4  5  6  7  8  9
#
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9


