import sys
import time
import threading
import numpy as np
import pandas as pd

from scipy.stats import norm, chi2
from scipy.stats import shapiro, f, levene
from scipy.stats import ttest_ind, f_oneway

MapFalseTrue_toNoYes = lambda x: 'No' if not x else 'Yes'

def animate_loading(stop_event, msg):
    loading_chars = [
        f'{msg}    ',
        f'{msg} .  ',
        f'{msg} .. ',
        f'{msg} ...'
    ]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write('\r' + loading_chars[i % 4])
        sys.stdout.flush()
        time.sleep(0.5)
        i += 1

    # After the loop, clear the line and print "Done!"
    sys.stdout.write(f'\r{msg} DONE!\n')

class MeansTester:
    '''
    Test if means/medians of given samples are significantly different via
    various statistical parametric tests.

    Steps
    -----
    1. Samples are first checked for equal variances
     - F-Test for 2 samples
     - Levene's Test for 3 samples or more
    2. Samples are then checked for normality
     - Shapiro-Wilk Test
    3. Samples' centers/means are then checked for statistical difference

    If at least one sample is NOT normal (distribution-wise), a
     - Mann-Whitney test for two samples
     - Kruskal-Wallis for three or more samples
    is carried out.

    Else, a
     - t-test for two samples
     - ANOVA for three or more samples
    is carried out, while accounting for unequal variances if variances
    between samples are unequal.

    Note
    ----
    1. In carrying out the F-Test, the F-statistic is defined here, always,
       as the bigger variance over the smaller one. Hence, a one-tailed test
       is always assumed.

    Parameters
    ----------
    *_samples : np.array
        Samples to investigate

    sig_level_variance : float (default = 0.05)
        Significance level for testing for unequal variances between samples.

    sig_level_normality : float (default = 0.05)
        Significance level for testing if samples are drawn from a normal
        distribution.

    sig_level_means : float (default = 0.05)
        Significance level for testing if means/medians of the samples
        are statistically different.

    alternative : {'two-sided', 'less', 'greater'}
        Defines the alternative hypothesis. Default is two-sided.
    
        * 'two-sided' : Samples have unequal means
        * 'less' : Mean/Median of first sample is less than mean of second
          sample
        * 'greater' : Mean of first sample is greater than mean of second
          sample

    verbose : bool (default = False)
    '''
    def __init__(
            self, *_samples,
            sig_level_variance=0.05,
            sig_level_normality=0.05,
            sig_level_means=0.05,
            alternative='two-sided',
            verbose=False
    ):
        # Parameters
        self.sig_level_variance = sig_level_variance
        self.sig_level_normality = sig_level_normality
        self.sig_level_means = sig_level_means
        self.alternative = alternative
        self.verbose = verbose

        # Attributes
        self.Samples = list(_samples)

        # Convert to numpy arrays if samples are passed in list
        for i in range(len(self.Samples)):
            if isinstance(self.Samples[i], list):
                self.Samples[i] = np.array(self.Samples[i])

        # - Variance Test
        self.VarianceTest_H1_Stats = None

        # - Normality Test
        self.NormalityTest_H1_Stats = [None for _ in range(len(self.Samples))]

        # - Means Test
        self.MeansTest_H1_Stats = None
        self.SignificantDifference = None

    def test_equalvariances(self):
        '''
        Test if the samples passed have equal variances. Depending on how many
        samples are passed, the following tests will be conducted:
         - F-Test for two samples
         - Levene-Test for three samples or more

        Returns
        -------
        rejectH0 : bool
         - If True, samples have statistically different variances
        '''
        if len(self.Samples) == 1:
            raise ValueError("At least two sample should be passed!")

        if len(self.Samples) == 2:
            stat, p, rejectH0= self.f_test()
            res = {'test_type': 'F-Test', 'stat': stat, 'p': p}

        elif len(self.Samples) >= 3:
            stat, p = levene(*self.Samples)
            if p <= self.sig_level_variance:
                rejectH0 = True
            else:
                rejectH0 = False
            res = {'test_type': 'Levene', 'stat': stat, 'p': p}

        return rejectH0, res

    def f_test(self):
        '''
        Test if variances from two samples are equal.

        Returns
        -------
        F : float
         - The computed F-statistic

        rejectH0 : bool
         - If True, samples have different variances
        '''
        # 1. Calculate variances
        sd1 = self.Samples[0].std(ddof=1) # should be the bigger one
        sd2 = self.Samples[1].std(ddof=1)

        if sd1 < sd2:
            sd1, sd2 = sd2, sd1
            n1 = len(self.Samples[1])
            n2 = len(self.Samples[0])
        else:
            n1 = len(self.Samples[0])
            n2 = len(self.Samples[1])

        # 2. Calculate F
        F = (sd1**2) / (sd2**2)

        # 3. Get critical F
        dfn = n1 - 1
        dfd = n2 - 1

        Fcrit = f.isf(self.sig_level_variance, dfn, dfd, loc=0, scale=1)

        # 4. Reject null hypothesis if F > Fcrit
        if F >= Fcrit:
            rejectH0 = True
        else:
            rejectH0 = False

        # 5. Also get the probability (area under the curve)
        p = f.sf(F, dfn, dfd, loc=0, scale=1)

        return F, p, rejectH0

    def rank_order_center_test(self):
        '''
        Rank-order test if samples have same centers.

        Returns
        -------
        rejectH0 : bool
         - If true, samples have statistically different medians (centers)
        '''
        self.create_rank_table()

        if len(self.Samples) == 2:
            stat, p, rejectH0 = self.mann_whitney()
            res = {'test_type': 'Mann-Whitney', 'stat': stat, 'p': p}

        elif len(self.Samples) >= 3:
            stat, p, rejectH0 = self.kruskal_wallis()
            res = {'test_type': 'Kruskal-Wallis', 'stat': stat, 'p': p}

        return rejectH0, res

    def mann_whitney(self):
        '''
        Test the null hypothesis that the centers (median) of two samples are
        statistically different. The computed Mann-Whitney U statistic follows
        a normal distribution.

        The Mann-Whitney test only applies to two samples.

        Returns
        -------
        rejectH0 : bool
         - If True, the two samples have statistically different centers
           (medians)
        '''
        n1Group = self.df_wRanks[self.df_wRanks["Group"]==0]
        n2Group = self.df_wRanks[self.df_wRanks["Group"]==1]

        # 1. Get n1 and n2
        n1 = n1Group.shape[0]
        n2 = n2Group.shape[0]

        # 2. Get T
        T = n1Group["Rank"].sum()

        # 3. Get mu
        mu = n1*(n1 + n2 + 1)/2

        # 4. Get standard deviation
        sd = (n1*n2*(n1 + n2 + 1)/12)**0.5

        # 5. Get z
        z = (T-mu)/sd

        # 6. Get critical z and computed p value
        # - Two-tailed test
        if self.alternative == 'two-sided':
            # The area (significant level) must be divided by half and split
            # on both ends. The normal distribution is symmetrical
            zcrit = norm.isf(self.sig_level_means/2, loc=0, scale=1)

            if abs(z) >= zcrit:
                rejectH0 = True
            else:
                rejectH0 = False

            p = norm.sf(abs(z), loc=0, scale=1)*2

        # - One-tailed test, test first sample against second sample
        else:
            zcrit = norm.isf(self.sig_level_means, loc=0, scale=1)

            # Note: The direction does not affect the computed z-value
            # Positive end
            if self.alternative == 'greater':
                if z >= zcrit:
                    rejectH0 = True
                else:
                    rejectH0 = False

                p = norm.sf(z, loc=0, scale=1)

            # Negative end
            # - The normal distribution is symmetrical
            elif self.alternative == 'less':
                if z <= -zcrit:
                    rejectH0 = True
                else:
                    rejectH0 = False

                p = 1 - norm.sf(z, loc=0, scale=1)

            else:
                raise ValueError(
                    "Possible options for parameter alternative must either " +
                    "be {'two-sided', 'greater', 'less'}."
                )

        return z, p, rejectH0

    def kruskal_wallis(self):
        '''
        Test the null hypothesis that the centers (median) of three or more
        samples are statistically different. The computed Kruskal-Wallis H
        statistic follows the chi-square distribution

        Returns
        -------
        rejectH0 : bool
         - If True, the samples have statistically different centers (medians)
        '''
        # 1. Get number of groups
        groups = list(set(list(self.df_wRanks["Group"])))

        # 2. Compute H-statistic
        SumofRanks = 0
        for g in groups:
            SumofRanks += (
                (self.df_wRanks[self.df_wRanks["Group"]==g]["Rank"].sum())**2
            ) / self.df_wRanks[self.df_wRanks["Group"]==g].shape[0]

        n = self.df_wRanks.shape[0]

        H = 12 / (n*(n+1)) * SumofRanks - 3*(n+1)

        # Calculate Hcrit
        Hcrit = chi2.isf(self.sig_level_means, len(groups)-1, loc=0, scale=1)

        if H >= Hcrit:
            rejectH0 = True
        else:
            rejectH0 = False

        # 3. Also get the probability (area under the curve)
        p = chi2.sf(H, len(groups)-1, loc=0, scale=1)

        return H, p, rejectH0

    def create_rank_table(self):
        '''
        Create a table comprised of the samples all joined together and
        ranked. Necessary for rank-order methods.
        '''
        # Initialize table
        data_ = {"Value": [], "Group": []}
        for i, sample in enumerate(self.Samples):
            for v in sample:
                data_["Value"].append(v)
                data_["Group"].append(i)

        self.df_wRanks = pd.DataFrame(data=data_)

        # Sort values in ascending order
        self.df_wRanks = self.df_wRanks.sort_values("Value")

        # Assign rank
        self.df_wRanks["Rank"] = np.arange(1, self.df_wRanks.shape[0]+1, 1, dtype=float)

        # Handle ties
        row_wties = self.df_wRanks.duplicated(subset=["Value"])
        idx_wties = row_wties.loc[row_wties == True].index.to_list()
        values_vties = self.df_wRanks.loc[idx_wties, "Value"].to_list()
        for v in values_vties:
            self.df_wRanks.loc[self.df_wRanks["Value"] == v, "Rank"] = np.mean(
                self.df_wRanks[self.df_wRanks["Value"] == v]["Rank"].to_list()
            )

    def continuous_mean_test(self):
        '''
        Means test for continuous data. Assumes samples are drawn from
        normal distribution.

        Returns
        -------
        rejectH0 : bool
         - If true, samples have statistically different means
        '''
        is_unequal_variance = self.VarianceTest_H1_Stats[0]
        if len(self.Samples) == 2:
            res_ttest = ttest_ind(
                *self.Samples, equal_var=(not is_unequal_variance),
                alternative=self.alternative
            )
            res = {
                'test_type': 't-test',
                'stat': res_ttest.statistic,
                'p': res_ttest.pvalue
            }
        elif len(self.Samples) >= 3:
            res_onewayANOVA = f_oneway(
                *self.Samples, equal_var=(not is_unequal_variance),
            )
            res = {
                'test_type': 'one-way ANOVA',
                'stat': res_onewayANOVA.statistic,
                'p': res_onewayANOVA.pvalue
            }

        if res['p'] <= self.sig_level_means:
            rejectH0 = True
        else:
            rejectH0 = False

        return rejectH0, res

    def test_means(self):
        '''
        Test for statistical difference of means/centers.
        '''
        # === === === === === === === === === === === ===
        # 1. Check if samples have unequal variances
        stop_event = threading.Event()
        animation_thread = threading.Thread(
            target=animate_loading,
            args=(stop_event, "Checking if samples have equal variances")
        )
        animation_thread.start()
        try:
            is_unequal_variance, res_vartest = self.test_equalvariances()
        finally:
            stop_event.set()
            animation_thread.join()

        self.VarianceTest_H1_Stats = (is_unequal_variance, res_vartest)
        if self.verbose:
            for k, v in res_vartest.items():
                print(f" - {k:<10}: {v}")
        print(
            "Samples have unequal variances: " +
            f"{MapFalseTrue_toNoYes(is_unequal_variance)}\n"
        )

        # === === === === === === === === === === === ===
        # 2. Check if samples have normal distribution
        #  - If at least one sample does not have a normal distribution,
        #    assign True
        stop_event = threading.Event()
        animation_thread = threading.Thread(
            target=animate_loading,
            args=(stop_event, "Checking if samples are drawn from normal distributions")
        )
        animation_thread.start()
        atleast_one_non_normal = False
        try:
            for i, sample in enumerate(self.Samples):
                normality_test = shapiro(sample)
                res_normtest = {
                    'test_type': 'Shapiro-Wilk',
                    'stat': normality_test.statistic, 'p': normality_test.pvalue
                }
                # Reject null hypothesis, i.e. samples are sampled from a normal
                # distribution
                if normality_test.pvalue <= self.sig_level_normality:
                    self.NormalityTest_H1_Stats[i] = (True, res_normtest) # reject H0
                    atleast_one_non_normal = True
                else:
                    self.NormalityTest_H1_Stats[i] = (False, res_normtest)
        finally:
            stop_event.set()
            animation_thread.join()

        if self.verbose:
            for i, sample in enumerate(self.NormalityTest_H1_Stats):
                print(
                    f"Samples #{i+1} are NOT drawn from a normal distribution: " +
                    f"{sample[0]}\n - {sample[1]}"
                )
        print(
            "At least one sample not drawn from a normal distribution: " +
            f"{MapFalseTrue_toNoYes(atleast_one_non_normal)}\n"
        )

        # === === === === === === === === === === === ===
        # 3. Test centers
        stop_event = threading.Event()
        animation_thread = threading.Thread(
            target=animate_loading,
            args=(stop_event, "Comparing means of samples")
        )
        animation_thread.start()
        try:
            if atleast_one_non_normal:
                is_sig_different, res_meanstest = self.rank_order_center_test()
            else:
                is_sig_different, res_meanstest = self.continuous_mean_test()
        finally:
            stop_event.set()
            animation_thread.join()

        self.MeansTest_H1_Stats = (is_sig_different, res_meanstest)
        if self.verbose:
            for k, v in res_meanstest.items():
                print(f" - {k:<10}: {v}")
        print(
            "Samples are significantly different: " +
            f"{MapFalseTrue_toNoYes(is_sig_different)}\n"
        )

        self.SignificantDifference = is_sig_different
