import numpy as np
import pandas as pd
import math

"""
计算弗雷德距离，输入数据为两个一维数组，shape=[N,]
"""

def calculate_euclid(point_a, point_b):
    """
    Args:
        point_a: a data point of curve_a
        point_b: a data point of curve_b
    Return:
        The Euclid distance between point_a and point_b
    """
    return math.sqrt((point_a - point_b)**2)

def calculate_frechet_distance(dp,i,j ,curve_a, curve_b):
    """
    Args:
        dp: The distance matrix
        i: The index of curve_a
        j: The index of curve_b
        curve_a: The data sequence of curve_a
        curve_b: The data sequence of curve_b
    Return:
        The frechet distance between curve_a[i] and curve_b[j]
    """
    if dp[i][j] > -1:
        return dp[i][j]
    elif i == 0 and j ==0:
        dp[i][j] = calculate_euclid(curve_a[0], curve_b[0])
    elif i > 0 and j == 0:
        dp[i][j] = max(calculate_frechet_distance(dp, i-1, 0, curve_a, curve_b), calculate_euclid(curve_a[i], curve_b[0]))
    elif i == 0 and j > 0:
        dp[i][j] = max(calculate_frechet_distance(dp, 0, j-1, curve_a,curve_b), calculate_euclid(curve_a[0],curve_b[j]))
    elif i > 0 and j > 0:
        dp[i][j] = max(min(calculate_frechet_distance(dp, i-1, j, curve_a, curve_b), calculate_frechet_distance(dp, i-1, j-1, curve_a, curve_b), calculate_frechet_distance(dp, i, j-1, curve_a, curve_b)), calculate_euclid(curve_a[i], curve_b[j]))
    else:
        dp[i][j] = float("inf")
    return dp[i][j]

def get_similarity(curve_a,curve_b):
    dp = [[-1 for _ in range(len(curve_b))] for _ in range(len(curve_a))]
    similarity =  calculate_frechet_distance(dp,len(curve_a)-1,len(curve_b)-1, curve_a, curve_b)
    return max(np.array(dp).reshape(-1,1))[0]

if __name__=='__main__':
    get_similarity(a,b) # a.shape = (N,1) b.shape=(N,1)
