# nlp-feature-extractor
Feature extractors for NLP<br>
WARNING: Pre-alpha status!<br><br>
The extractors assume a very specific dataframe format consisting of the following columns:
- sentence (string)
- word (string)
- index (int)
- label (int, either 0 or 1)

You should also run `pip install -r requirements.txt` in order to get all packages needed **and** in run `nltk.download('all')` 
inside of Python to make sure, that nltk has all corpora needed.

# Content
The following extractors are currently implemented
0. Label (to get the label)
1. Word (optional w/ offset)
2. WordLength
3. ContainsChar
4. MatchesRegex
5. POS Tag (optional w/ offset)
6. Stem
7. SentenceLength
8. DistinctWordsInSentence (number of distinct words)
9. EditDistance (to a reference word)
10. NumberSynonyms
11. NumberHyponyms
12. NumberHypernyms
13. NumberPronounciations
14. NumberVowels
15. NumberConsonants

# Sample Usage
```python
from extractors import Word, MatchesRegex, fu_to_df

df_data = pd.read_csv(...) # see above for dataframe requirements

fu = FeatureUnion([
  ("word", Word()),
  ("contains_ii", MatchesRegex(r".*ii.*"),
], n_jobs=-1)

df_transformed = fu_to_df(fu, df_data)
```
