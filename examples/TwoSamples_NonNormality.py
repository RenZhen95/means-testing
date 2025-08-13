import numpy as np
from means_testing import MeansTester

# Data from book (p234)
# - Percent functionality of the joint

# Question: Orthopedist removes hardware from 9 patients but leaves it in for the other patients.
# Is there a functionality difference between the two?

# Group that did not remove hardware
Group0 = [45, 72, 85, 90, 93, 95, 95, 98, 99, 100]
Group0Median = np.median(Group0)

# Group that removed the hardware
Group1 = [35, 63, 65, 70, 75, 78, 80, 90, 100]
Group1Median = np.median(Group1)

StatToolbox = MeansTester(Group0, Group1, alternative="less")

StatToolbox.test_means()

print(f"Group0 Median: {Group0Median}")
print(f"Group1 Median: {Group1Median}")
