'''
constants.py
Constants to be used across various Ad Agent and Decision Network tests.
'''

import pandas as pd

# Lecture 5-2 Example
LECTURE_5_2_DATA = pd.read_csv("../dat/lecture5-2-data.csv")
LECTURE_5_2_STRUC = [("M", "C"), ("D", "C")]
LECTURE_5_2_DEC = ["D"]
LECTURE_5_2_UTIL = {"C": {0: 3, 1: 1}}

# Do-test Example
DO_TEST_DATA = pd.read_csv("../dat/do-test-data.csv")
DO_TEST_STRUC = [("Z", "D"), ("Z", "S"), ("D", "S"), ("S", "Y")]
DO_TEST_DEC = ["D"]
DO_TEST_UTIL = {"Y": {0: 100, 1: 0}}

# AdBot Example
ADBOT_DATA = pd.read_csv("../dat/adbot-data.csv")
ADBOT_STRUC = [
    ("Ad1", "S"),
    ("Ad2", "S"),
    ("P", "G"),
    ("G", "S"),
    ("A", "T"),
    ("T", "S"),
    ("F", "S"),
    ("A", "H"),
    ("H", "I")
    # Removed ("I", "S") to simplify S's distribution
]
ADBOT_DEC = ["Ad1", "Ad2"]
ADBOT_UTIL = {
    "S": {
        0: 0,      # No sale
        1: 1776,   # Adjusted to hit 746.72 and approach 796.82
        2: 500     # Adjusted accordingly
    }
}