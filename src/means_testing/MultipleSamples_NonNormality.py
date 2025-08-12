from means_testing import MeansTester

# Instrument 1
data1 = [2.00, 2.00, 2.01, 2.03, 2.03, 2.04, 2.07, 2.07, 2.09, 2.12]

# Instrument 2
data2 = [1.90, 1.95, 1.96, 1.97, 2.01, 2.03, 2.04, 2.06, 2.07, 2.31]

# Instrument 3
data3 = [1.95, 1.96, 1.96, 1.98, 2.00, 2.01, 2.04, 2.06, 2.07, 2.76]

# Instrument 4
data4 = [1.96, 1.98, 1.98, 2.01, 2.04, 2.06, 2.08, 2.10, 2.11, 3.02]

# Instrument 5
data5 = [1.96, 1.98, 1.98, 2.00, 2.03, 2.03, 2.06, 2.07, 2.11, 3.03]

StatToolbox = MeansTester(
    data1, data2, data3, data4, data5
)

StatToolbox.test_means()
