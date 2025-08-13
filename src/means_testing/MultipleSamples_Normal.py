from means_testing import MeansTester

# Exercise 12.7 on page 272

# Are experimental animals equally resistant to parasites?

# Rats are injected with 500 larvae each of a parasitic worm. 10 days later,
# rats were sacrificed and number of adult worms counted.

# Is there a batch-to-batch difference in resistance to parasite infestation
# by groups of rats received from the supplier?


group1 = [279, 338, 334, 198, 303]
group2 = [378, 275, 412, 265, 286]
group3 = [172, 335, 335, 282, 250]
group4 = [381, 346, 340, 471, 318]

StatToolbox = MeansTester(group1, group2, group3, group4, two_tailed=True)

StatToolbox.test_means()
