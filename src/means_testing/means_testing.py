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
    In carrying out the F-Test, the F-statistic is defined here, always,
    as the bigger variance over the smaller one. Hence, a one-sided test
    is always assumed.    

    Parameters
    ----------
    *_samples : np.array
     - Samples to investigate

    sig_level_variance : float (default = 0.05)
     - Significance level for testing for unequal variances between samples

    sig_level_normality : float (default = 0.05)
     - Significance level for testing if samples are drawn from a normal
       distribution

    sig_level_means : float (default = 0.05)
     - Significance level for testing if means/medians of the samples are
       statistically different

    two_sided : bool
     - Carry out two-sided test if True
    '''
    def __init__(
            self, *_samples,
            sig_level_variance=0.05,
            sig_level_normality=0.05,
            sig_level_means=0.05,
            two_sided=True
    ):
        self.samples = _samples
        self.sig_level_variance = sig_level_variance
        self.sig_level_normality = sig_level_normality
        self.sig_level_means = sig_level_means
        self.two_sided = two_sided

    def test_equalvariances(self):
        '''
        Test if the samples passed have equal variances. Depending on how many
        samples are passed, the following tests will be conducted:
         - F-Test for two samples
         - Levene-Test for three samples or more
        '''
        if len(self.samples) == 1:
            raise ValueError("At least two sample should be passed!")

        if len(self.samples) == 2:
            stat, reject_H0 = self.f_test()

        elif len(self.samples) >= 3:
            stat, p = levene(self.samples)
            if p <= self.sig_level_variance:
                reject_H0 = True
            else:
                reject_H0 = False
    
        return reject_H0
    
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
        sd1 = self.samples[0].std(ddof=1) # should be the bigger one
        sd2 = self.samples[1].std(ddof=1)
    
        if sd1 < sd2:
            sd1, sd2 = sd2, sd1
            n1 = len(self.samples[1])
            n2 = len(self.samples[0])
        else:
            n1 = len(self.samples[0])
            n2 = len(self.samples[1])
    
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
    
        return F, rejectH0

    def create_rank_table(self):
        '''
        Create a table comprised of the samples all joined together and 
        ranked. Necessary for rank-order methods.
        '''
        # Initialize table
        data_ = {"Value": [], "Group": []}
        for i, sample in enumerate(self.samples):
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
    
    def mann_whitney(self):
        '''
        Test the null hypothesis that the centers (median) of two samples are
        statistically different. The computed Mann-Whitney U statistic follows
        a normal distribution.
    
        The Mann-Whitney test only applies to two samples.

        Returns
        -------
        rejectH0 : bool
         - If True, the two samples do not have similar centers (medians)
        '''
        if (
                self.df_wRanks[self.df_wRanks["Group"]==0].shape[0] <
                self.df_wRanks[self.df_wRanks["Group"]==1].shape[0]
        ):
            n1Group = self.df_wRanks[self.df_wRanks["Group"]==0]
            n2Group = self.df_wRanks[self.df_wRanks["Group"]==1]
        else:
            n1Group = self.df_wRanks[self.df_wRanks["Group"]==1]
            n2Group = self.df_wRanks[self.df_wRanks["Group"]==0]
    
        # Get n1 and n2
        n1 = n1Group.shape[0]
        n2 = n2Group.shape[0]
    
        # Get T
        T = n1Group["Rank"].sum()
    
        # Get mu
        mu = n1*(n1 + n2 + 1)/2
        
        # Get s.d
        sd = (n1*n2*(n1 + n2 + 1)/12)**0.5
        
        # Get z
        z = (T-mu)/sd
    
        # Get critical z (two-sided test)
        if self.two_sided:
            zcrit = norm.isf(self.sig_level_means/2, loc=0, scale=1)
    
            if abs(z) >= zcrit:
                rejectH0 = True
            else:
                rejectH0 = False

        else:
            zcrit = norm.isf(self.sig_level_means, loc=0, scale=1)
    
        return rejectH0
    
    def kruskal_wallis(df, _sig_level=0.05):
        '''
        Test the null hypothesis that the centers (median) of three or more 
        samples are statistically different. The computed Kruskal-Wallis H
        statistic follows a normal distribution.
    
        This function ONLY assumes a two-sided test!
    
        Parameters
        ----------
        df : pd.DataFrame
         - Table of sample values ordered in ascending order, together with
           corresponding ranks
    
        _sig_level : float
    
        Returns
        -------
        rejectH0 : bool
         - If True, the two samples do not have similar centers (medians)
        '''
        # Get number of groups
        groups = list(set(list(df["Group"])))
    
        # Compute H-statistic    
        SumofRanks = 0
        for g in groups:
            SumofRanks += (
                (df[df["Group"]==g]["Rank"].sum())**2
            ) / df[df["Group"]==g].shape[0]
    
        n = df.shape[0]
        
        H = 12 / (n*(n+1)) * SumofRanks - 3*(n+1)
    
        # Calculate Hcrit
        Hcrit = chi2.isf(self.sig_level_means, len(groups)-1, loc=0, scale=1)
    
        if H >= Hcrit:
            rejectH0 = True
        else:
            rejectH0 = False
    
        return rejectH0
    
    def rank_order_center_test(self):
        '''
        Rank-order test if samples have same centers.

        Returns
        -------
        means_are_different : bool
        '''
        table_wRanks = create_rank_table(_samples)
    
        if len(_samples) == 2:
            rejectH0 = mann_whitney(table_wRanks, _sig_level=_sig_level)
        elif len(_samples) >= 3:
            rejectH0 = kruskal_wallis(table_wRanks, _sig_level=_sig_level)
    
        return rejectH0
    
    def test_means(self):
        '''
        Test for statistical difference of means/centers.
        '''
        # 1. Check if samples have unequal variances
        is_unequal_variance = test_equalvariances(*_samples, _sig_level=0.05)
    
        # 2. Check if samples have normal distribution
        #  - If at least one sample does not have a normal distribution,
        #    assign True
        are_non_normal = [False for _ in range(len(_samples))]
        atleast_one_non_normal = False
        for i, sample in enumerate(_samples):
            normality_test = shapiro(sample)
    
            # Reject null hypothesis, i.e. samples are sampled from a normal
            # distribution
            if normality_test.pvalue <= _sig_level:
                are_non_normal[i] = True
                atleast_one_non_normal = True
    
        # 3. Test centers
        if atleast_one_non_normal:
            is_significant = rank_order_center_test(*_samples)
        else:
            is_significant = means_test(*_samples)
    
        # print(is_unequal_variance)
        # print(is_non_normal)
        return is_significant
