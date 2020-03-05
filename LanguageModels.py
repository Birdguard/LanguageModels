
import os
import pandas as pd
import numpy as np
import requests
import time
import re
import string

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    time.sleep(5)
    request = requests.get(url)
    request.encoding = 'utf-8'
    text = request.text
    pattern = "\START OF THIS PROJECT GUTENBERG[^*]+\*{3}|END OF THIS PROJECT GUTENBERG[^*]+\*{3}"
    text = re.split(pattern, text)
    content = text[len(text)//2]
    pattern = "\r\n"
    content = re.sub(pattern, "\n", content)
    return content
    

def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    book_string = book_string.strip()
    pattern = "\s*\n{2,}\s*"
    # pattern = "\n{2,}"
    paragraphs = re.split(pattern, book_string)
    # pattern = "\\b[a-zA-Z0-9]+\\b|[^\s]"
    pattern = "\\b\w+\\b|[^\s]"
    # pattern = "\\b\w+\\b|[,.;']"

    # pattern = "[\w]+[^\s]+"
    tokens = []
    # return paragraphs
    for paragraph in paragraphs:
        tokens.append('\x02')
        tokens += re.findall(pattern, paragraph)
        tokens.append('\x03')

    return tokens


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        tokens = pd.Series(list(tokens))
        unique = tokens.unique()
        uniform_probability = 1/len(unique)
        token_probability = pd.Series([uniform_probability]*len(unique))
        token_probability.index = unique
        return token_probability
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        prob_words = 1
        for word in words:
            if (word not in self.mdl.index):
                return 0
            else:
                prob_words *= self.mdl[word]
        return prob_words
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        tokens = pd.Series(self.mdl.index).sample(n=M, replace=True, weights=self.mdl.values).tolist()
        return " ".join(tokens)




class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        token_counts = pd.Series(list(tokens)).value_counts()
        token_probability = token_counts/token_counts.sum()
        return token_probability
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        prob_words = 1
        for word in words:
            if (word not in self.mdl.index):
                return 0
            else:
                prob_words *= self.mdl[word]
        return prob_words
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        tokens = pd.Series(self.mdl.index).sample(n=M, replace=True, weights=self.mdl.values).tolist()
        return " ".join(tokens)
        

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        n_grams = []
        for token_idx in range(0, len(tokens)+1-self.N):
            n_gram = [tokens[token_idx+i] for i in range(self.N)]
            n_grams.append(tuple(n_gram))
        return n_grams
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe indexed on distinct tokens, with three
        columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """

        # ngram counts C(w_1, ..., w_n)
        ...
        # n-1 gram counts C(w_1, ..., w_(n-1))
        ...

        # Create the conditional probabilities
        ...
        
        # Put it all together

        ...
        df_dict = {'ngram':[], "n1gram":[], "prob":[]}
        if (len(ngrams) == 0):
        	return pd.DataFrame.from_dict(df_dict)
        ngram_counts = pd.Series(list(ngrams)).value_counts()
        unique_ngrams = ngram_counts.index
        num_ngrams = ngram_counts.sum()
        n1grams = []
        usedLast = False
        for i in range(len(ngrams) + 1):
        	n1gram = None
        	if (i + self.N - 1 < len(ngrams)):
        		n1gram = [ngrams[i][0] for i in range(i, i + self.N-1)]
        	elif (i + self.N - 1 == len(ngrams)):
        		n1gram = [ngrams[i][0] for i in range(i, i + self.N-1)]
        		usedLast = True
        	else:
        		if (usedLast == False):
        			n1gram = [ngrams[i-1][j] for j in range(0, self.N-1)]
        			usedLast = True
        			# return n1gram
        		else:
        			n1gram = ngrams[i-1][1:]

        	n1grams.append(tuple(n1gram))

        # if (self.N == 2):
        # 	print('what')
        # 	print(pd.Series(n1grams)[0])
        # 	n1grams = pd.Series(n1grams).apply(lambda x: x[0])
        n1gram_counts = pd.Series(n1grams).value_counts()
        # print(n1gram_counts)
        # return
        for i in range(len(unique_ngrams)):
        	# print(ngram_count)
        	df_dict['ngram'].append(unique_ngrams[i])
        	n1gram_associated = unique_ngrams[i][:-1]
        	df_dict['n1gram'].append(n1gram_associated)
        	df_dict['prob'].append(ngram_counts[i]/n1gram_counts[n1gram_associated])
        df = pd.DataFrame.from_dict(df_dict)
        if (self.N == 2):
        	df['n1gram'] = df['n1gram'].apply(lambda x: x[0])
        return df
    

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        total_words = len(words)
        num_words = self.N
        prob = 1
        curr_mdl = self
  
        sets_of_words = [words[i:i+num_words] for i in range(total_words-num_words + 1)]
        
        while(num_words != 0):
                	for set_of_words in sets_of_words:
        		prob_df = curr_mdl.mdl
        		if (num_words != 1):
        			ngram_prob = prob_df['ngram'] == tuple(set_of_words)
        			if (ngram_prob.sum() != 0):
        				prob *= prob_df.loc[prob_df['ngram'] == tuple(set_of_words)]['prob'].values
        			else:
        				return 0
        		else:
        			prob *= prob_df.loc[set_of_words]


        	sets_of_words = [sets_of_words[0][:-1]]


        	num_words -= 1
        	if (num_words != 0):
        		curr_mdl = curr_mdl.prev_mdl
        return prob[0]


    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        
        # Transform the tokens to strings
        ...
        tokens = ""
        curr_mdl = self
        num_tokens = self.N - 1
        if (M < self.N):
        	for i in range(self.N - 1 - M):
        		curr_mdl = curr_mdl.prev_mdl
        		num_tokens -= 1
        ngram_df = curr_mdl.mdl
        if (M == 1):
        	tokens += '\x02'
        	ngram_df = ngram_df.loc[ngram_df['n1gram'] == '\x02']
        	ngram = pd.Series(ngram_df['ngram']).sample(n=1, replace=True, weights=ngram_df['prob']).values[0][-1]
        	tokens += ' ' + ngram
        else:
        	tokens += self.sample(M-1)
        	tokens_list = tokens.split()
        	prior_tokens = tuple(tokens_list[len(tokens_list) - num_tokens:])
        	prior_tokens_availability = ngram_df['n1gram'] == tuple(prior_tokens)
        	tokens += ' '
        	if (prior_tokens_availability.sum() == 0):
        		tokens += '\x03'
        	else:
        		ngram_df = ngram_df.loc[prior_tokens_availability]

        		ngram = pd.Series(ngram_df['ngram']).sample(n=1, replace=True, weights=ngram_df['prob']).tolist()[0][-1]
        		tokens += ngram
 
        return tokens


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
