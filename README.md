adsstat
=======
**Preliminary documentation**

Query ADS about one author publications and provide publication rate, authorship, citation statistics and even wordcloud


This package offers simple proxy to query ADS about a given author. Each query collect the list of publication (with detailed description)
associated to one author into a Query object. Each paper can then be accessed individually, and author statistics can be
made, including plots and now wordcloud as well!

Version history
---------------
    v1.0: initial version
    v2.0: added word cloud figure (requires running make, auto-fallback)


Example usage
-------------

```python
    >>> from papers import Query
    # pylab to make multiple figures for this example
    >>> import pylab as plt
    # run a query
    >>> r = Query('fouesneau')
    # plot the number of publications per year
    >>> r.plot_year()
    # plot the distribution of number of citations per year
    >>> plt.figure()
    >>> r.hist2d('year', 'ncited')
    #make wordcloud
    >>> r.show_word_cloud()  # take a while depending on the abstract data
```
