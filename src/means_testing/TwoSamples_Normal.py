from means_testing import MeansTester

# From page 261

# A physician wants to compare treatment 1 (t1_exp1) vs treatment 2 (t2_exp1).

# Data is recorded as pain before treatment, minus 45 minutes after treatment
# (scale of 1-10)

t1_exp1 = [
    6, 7, 2, 5, 3, 0, 3, 4, 5, 6, 1, 1, 1, 8, 6
]
t2_exp1 = [
    0, 1, 8, 4, 7, 4, 7, 7, 6, 1, 0, 4, 4
]

StatToolbox = MeansTester(t1_exp1, t2_exp1, alternative='two-sided', verbose=False)

StatToolbox.test_means()

# From page 263

# A study, to compare a new postoperative pain relief drug (t1_exp2) to
# the established Demerol (t2_exp2)

# Data is recorded as reduction of pain (scale of 1-12)

t1_exp2 = [
    2, 0, 3, 3, 0, 0, 7, 1, 4, 2, 2, 1, 3
]
t2_exp2 = [
    2, 6, 4, 12, 5, 8, 4, 0, 10, 0
]

StatToolbox2 = MeansTester(t1_exp2, t2_exp2, alternative='two-sided', verbose=False)

StatToolbox2.test_means()

if StatToolbox2.SignificantDifference:
    print("Difference in means are significant!")
else:
    print("Difference in means are NOT significant!")
