import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=self.verbose).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # to calculate: BIC = -2 * logL + p * logN
        # where L is the likelihood of the fitted model, p is the number of parameters,
        # and N is the number of data points. The term −2 log L decreases with
        # increasing model complexity (more parameters), whereas the penalties
        # p log N increase with increasing complexity. The BIC applies a larger penalty
        # when N > e^2 = 7.4.
        min_bic = float("+inf")
        best_model = None

        N = len(self.X)
        num_features = len(self.X[0])

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_components)

                # score computes the log probability under the model (http://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.base._BaseHMM.score).
                logL = model.score(self.X, self.lengths)

                # number of free states (https://discussions.udacity.com/t/verifing-bic-calculation/246165/2)
                # According to the formula, p(number of free parameters) is sum of these 4 terms:
                # - Transition probs are the transmat array: num_components * num_components
                # Since we know they add to 1.0, the last row can be calculated from the others,
                # so the finally for learned parameters it is `num_components * (num_components - 1)`
                # - Starting probabilities are the startprob array and are learned and are size
                # num_components, but since they add up to 1.0, so it will be `num_components - 1`
                # - Number of means= `num_components * num_features`
                # - Variances are the size of the covars array, Since we are using
                # "diag" it will be `num_components * num_features`
                transition_probs = num_components * (num_components - 1)
                starting_probs = num_components - 1
                n_means = num_components * num_features
                n_variances = num_components * num_features

                p = transition_probs + starting_probs + n_means + n_variances

                bic = -2 * logL + p * math.log(N)

                if  bic < min_bic:
                    min_bic = bic
                    best_model = model
            except:
                continue
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        ModelSelector.__init__(self, all_word_sequences, all_word_Xlengths, this_word,
                 n_constant, min_n_components, max_n_components,random_state, verbose)

        self.other_words = [word for word in self.hwords if word != self.this_word]

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_dic = float("-inf")
        best_model = None
        M = len(self.hwords)

        for num_components in range(self.min_n_components,self.max_n_components + 1):
            try:
                model = self.base_model(num_components)
                logL = model.score(self.X, self.lengths)

                logL_other_words = [
                        model.score(X, lengths)
                        for (X, lengths)
                        in [self.hwords[word] for word in self.other_words]
                    ]

                dic = logL - np.average(logL_other_words)

                if dic > max_dic:
                    max_dic = dic
                    best_model = model
            except Exception as e:
                if self.verbose:
                    print("Error calculating for {} components: {}".format(num_components, str(e)))

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        raise NotImplementedError
