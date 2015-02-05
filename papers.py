"""
This package offers simple proxy to query ADS about a given author.
Each query collect the list of publication (with detailed description)
associated to one author into a Query object.
Each paper can then be accessed individually, and author statistics can be
made, including plots and now wordcloud as well!

Version history:
    v1.0: initial version
    v2.0: added word cloud figure (requires running make, auto-fallback)

Example usage:
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
"""
from __future__ import print_function

__author__ = 'M. Fouesneau'
__version__ = '2.0'

import xml.etree.cElementTree as ET
import numpy as np
import pylab as plt

import sys
import operator

if sys.version_info[0] > 2:
    py3k = True
    from urllib.request import urlopen
    iteritems = operator.methodcaller('items')
    iterkeys = operator.methodcaller('keys')
    itervalues = operator.methodcaller('values')
else:
    py3k = False
    from urllib2 import urlopen
    iteritems = operator.methodcaller('iteritems')
    iterkeys = operator.methodcaller('iterkeys')
    itervalues = operator.methodcaller('itervalues')

try:
    import wordcloud
except ImportError:
    wordcloud = None

__all__ = ['ezrc', 'Paper', 'Query', 'WordCloud']


def ezrc(fontSize=16., lineWidth=2., labelSize=None, tickmajorsize=10, tickminorsize=5):
    """
    Define pylab rcParams to make pretty figures
    This function is not called by default.
    """
    from pylab import rc, rcParams
    if labelSize is None:
        labelSize = fontSize + 5
    rc('figure', figsize=(8, 6))
    rc('lines', linewidth=lineWidth)
    rc('font', size=fontSize, family='serif', weight='small')
    rc('axes', linewidth=lineWidth, labelsize=labelSize)
    rc('legend', borderpad=0.1, markerscale=1., fancybox=False)
    rc('text', usetex=True)
    rc('image', aspect='auto')
    rc('ps', useafm=True, fonttype=3)
    rcParams['xtick.major.size'] = tickmajorsize
    rcParams['xtick.minor.size'] = tickminorsize
    rcParams['ytick.major.size'] = tickmajorsize
    rcParams['ytick.minor.size'] = tickminorsize
    rcParams['font.sans-serif'] = 'Helvetica'
    rcParams['font.serif'] = 'Helvetica'


class Paper(object):
    """ A publication paper class that stores all available information from
    the xml record definition.
    Publication properties can be accessed through get_xxx commands.
    An author position can be found using find_author_pos
    """

    def __init__(self, rec):
        """ Create a paper instance from an xml record
        INPUTS
        ------
        rec: xml node
            paper node in the xml tree
        """
        self.__dict__.update(self.parseRecord(rec))
        self._xml = rec

    def get(self, k, default=None):
        """ Get an attribute or a default value """
        return self[k] or default

    def parseRecord(self, record):
        """ from xml node to __dict__
        INPUTS
        ------
        rec: xml node
            paper node in the xml tree
        """
        d = {}
        for c in record.getchildren():
            tag = c.tag.split('}')[-1]
            if tag in d:
                if type(d[tag]) == list:
                    d[tag].append(c.text)
                else:
                    d[tag] = [d[tag], c.text]
            else:
                d[tag] = c.text
        return d

    def pprint(self, short_authors=True):
        """ generate a pretty ascii representation
            author list, year, title, journal volume, page [citation count]

        KEYWORDS
        --------
        short_authors: bool
            if set, any author list longer than 3 will be shorten to "first author, et al."

        OUTPUTS
        -------
        txt: string
            e.g.: author list, year, title, journal volume, page [citation count]
        """
        au = self.get_author(short=short_authors)
        yr = self.get_year()
        ti = self.get_title()
        jo = self.get_abbrJournal()
        vo = self.get_volume()
        pp = self.get_page()
        nc = self.get_ncitations()
        return '{}, {}, "{}", {}, {}, {} [{} citations]'.format(au.encode('utf8', 'ignore'), yr, ti.encode('utf8', 'ignore'), jo, vo, pp, nc)

    def get_year(self):
        """ return the year of publication """
        return self['bibcode'][:4]

    def get_ncitations(self):
        """ return the nuber of citations """
        return self.get('citations', 0)

    def get_abbrJournal(self):
        """ return the abbrevated journal name """
        return self['bibcode'][4:9].replace('.', '')

    def get_volume(self):
        """ Return the volume of publication """
        return self['bibcode'][9:13].replace('.', '')

    def get_page(self):
        """ return the papge of publication """
        return self.get('page', self['bibcode'][13:-1].replace('.', '') )

    def get_author(self, short=True):
        """ return the author list

        KEYWORDS
        --------
        short_authors: bool
            if set, any author list longer than 3 will be shorten to "first author, et al."
        """
        au = self['author']
        if (not short) | (len(au) <= 3):
            au = ', '.join(au)
        else:
            au = au[0].split(',')[0] + ' et al.'
        return au

    def get_title(self):
        """ return the title of the publication """
        return self.get('title', 'NA')

    def find_author_pos(self, author):
        """ find the position of an author in the author list
        INPUTS
        ------
        author: string
            author to look for

        OUTPUTS
        -------
        pos: int
            the position in the author list
        """
        au = self['author']
        _author = author.split(',')[0]
        if type(au) == str:
            au = [au]
        try:
            return [e + 1 for (e, k) in enumerate(list(au)) if _author.lower() in k.lower()][0]
        except IndexError:
            return -1

    def __getitem__(self, k):
        return self.__dict__.get(k)

    def __repr__(self):
        return self.pprint(short_authors=True)

    def __str__(self):
        return self.pprint(short_authors=True)


class Query(object):
    """ Query objects collect all publications of a given author from NASA ADS abtract query
    Each publication is stored as a Paper instance and statistics are made
    available as well as functions to produce common figures such as the
    repartition of citations or the number of publications per year.
    """
    def __init__(self, author):
        """ Constructor
        INPUTS
        ------
        author: string
            author to query ADS for
        """
        self.author = author
        self._xml = self._query(self._build_query(author))
        tmp = self._xml.getroot().getchildren()
        self.papers = [Paper(k) for k in tmp]

        self.labels = { 'year':  'publications per year',
                        'ncited': 'citations per paper',
                        'nauthors': 'number of authors per paper',
                        'pos': 'author position per paper'}

        dtype = np.dtype(
            [ ('author', 'S25'), ('year', 'i4'), ('nauthors', 'i8'), ('ncited', 'i8'),
              ('journal', '<S25'), ('bibcode', '<S20'), ('pos', 'i8') ] )

        stats = [ (k.get_author(short=True).encode('utf8', 'ignore'), k.get_year(),
                   len(k.author), k.get_ncitations(), k.get_abbrJournal(),
                   k.bibcode, k.find_author_pos(self.author)) for k in self.papers
                  ]

        self.stats = np.asarray(stats, dtype=dtype)
        self.wordcloud = None
        self.wordcloud_image = None

    def make_word_cloud(self, filter_common=True, group_similar=True, **kwargs):
        print('creating the cloud... This may take a while.')
        a = ' '.join([k.abstract for k in self.papers])
        self.wordcloud = WordCloud(a, filter_common, group_similar)
        print('Cloud created. Making image...')
        self.wordcloud.make_image(**kwargs)
        print('Done.')

    def show_word_cloud(self, ax=None, **kwargs):
        if self.wordcloud is None:
            self.make_word_cloud(**kwargs)

        if wordcloud is not None:
            if ax is None:
                ax = plt.gca()
            ax.imshow(np.asarray(self.wordcloud.image), aspect='equal')
            ax.set_axis_off()
            plt.draw_if_interactive()
        else:
            print(self.worldcloud.words.most_common(20))

    def _build_query(self, author, nmax=None, yrmin=None, yrmax=None):
        """ generate the query with all pieces of information """
        _author = author.replace(', ', '%2C+')
        url = 'http://adsabs.harvard.edu/cgi-bin/nph-abs_connect?'
        url += 'db_key=AST&db_key=PHY&db_key=PRE&qform=AST'
        url += '&arxiv_sel=astro-ph&arxiv_sel=cond-mat&arxiv_sel=cs&arxiv_sel=gr-qc'
        url += '&arxiv_sel=hep-ex&arxiv_sel=hep-lat&arxiv_sel=hep-ph&arxiv_sel=hep-th'
        url += '&arxiv_sel=math&arxiv_sel=math-ph&arxiv_sel=nlin&arxiv_sel=nucl-ex'
        url += '&arxiv_sel=nucl-th&arxiv_sel=physics&arxiv_sel=quant-ph&arxiv_sel=q-bio'
        url += '&sim_query=YES&ned_query=YES&adsobj_query=YES&aut_logic=OR&obj_logic=OR'
        url += '&author={}'.format(_author)
        url += '&object='
        url += '&start_mon=&start_year={}'.format(yrmin or '')
        url += '&end_mon=&end_year={}'.format(yrmax or '')
        url += '&ttl_logic=OR&title=&txt_logic=OR&text='
        url += '&nr_to_return={}&start_nr=1'.format(nmax or '')
        url += '&jou_pick=ALL&ref_stems=&data_and=ALL&group_and=ALL&start_entry_day=&start_entry_mon=&start_entry_year='
        url += '&end_entry_day=&end_entry_mon=&end_entry_year=&min_score=&sort=SCORE&data_type=SHORT&aut_syn=YES'
        url += '&ttl_syn=YES&txt_syn=YES&aut_wt=1.0&obj_wt=1.0&ttl_wt=0.3&txt_wt=3.0'
        url += '&aut_wgt=YES&obj_wgt=YES&ttl_wgt=YES&txt_wgt=YES&ttl_sco=YES&txt_sco=YES&version=1'
        url += '&data_type=XML'
        return url

    def _query(self, url):
        """ make the query """
        f = urlopen(url)
        data = ET.parse(f)
        f.close()
        return data

    def argsort(self, keys):
        """ sort papers by given list of keys, numbers can be preceeded by a
        minus sign to make a decreasing the order """
        npapers = len(self.papers)

        def keyfn(x):
            if hasattr(keys, '__iter__'):
                r = []
                for rk in keys:
                    if rk[0] == '-':
                        r.append(-self.stats[x][rk[1:]])
                    else:
                        r.append(self.stats[x][rk])
            else:
                if keys[0] == '-':
                    r = -self.stats[x][keys[1:]]
                else:
                    r = self.stats[x][keys]
            return r

        return sorted(range(npapers), key=keyfn)

    def pprint_list(self, short_authors=False, delim='\n', latex_escapes=False):
        """ Return a pretty representation of the full list of papers
        KEYWORDS
        --------
        short_authors: bool (default False)
            if set, author lists with more than 3 authors are shorten to "first author", et al.

        delim: string (default '\n')
            useful flexibility for producing latex lists ('\n \\item')

        latex_escapes: bool
            if set, add escape character in front of '&'

        OUTPUTS
        -------
        txt: string
            list of publications
        """
        txt = delim.join(k.pprint(short_authors=short_authors) for k in self.papers)
        if latex_escapes:
            txt = txt.replace('&', '\&')
        return txt

    def hist(self, key, ax=None, alpha=0.5, histtype='stepfilled', bins=None, align='mid', **kwargs):
        """ Generate the histogram of a quantity key
        INPUTS
        ------
        key: string
            key present in self.stats ('author', 'year', 'nauthors', 'ncited', 'journal', 'bibcode', 'pos')

        KEYWORDS
        --------
        ax: matplotlib.Axes instance (default plt.gca())
            axis to use if provided.

        alpha: 0<= float <= 1 (default 0.5)
            alpha value for the histogram

        histtype: string
            type of histogram (see plt.hist)

        bins: int, iterable
            bins to use in histogram (see plt.hist)

        align: string in (left, mid, right)
            Controls how the histogram is plotted. (see plt.hist)

        extra keywords are forwarded to plt.hist call
        """
        if ax is None:
            ax = plt.gca()
        data = self.stats[key]
        if bins is None:
            bins = np.arange(min(data) - 0.5, max(data) + 1.5)
        n, _, _ = ax.hist(data, histtype=histtype, alpha=alpha, bins=bins, align=align, **kwargs)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], max(n) + 0.1)
        ax.set_xlabel(self.labels[key])
        ax.set_ylabel('counts')
        plt.draw_if_interactive()

    def hist2d(self, key1, key2, ax=None, bins=None, cmap=plt.cm.bone_r, **kwargs):
        """
        Generate the 2d distribution of a pair of keys from stats

        INPUTS
        ------
        key1: string
            key to use on the xaxis

        key2: string
            key to use on the yaxis

        keys must be present in self.stats:
            ('author', 'year', 'nauthors', 'ncited', 'journal', 'bibcode', 'pos')

        KEYWORDS
        --------
        ax: matplotlib.Axes instance (default plt.gca())
            axis to use if provided.

        bins: int, iterable
            bins to use in histogram (see plt.hist)

        cmap: colormap instance (default plt.cm.bone_r)
            colormap to use for the plot

        extra keywords are forwarded to plt.pcolor call
        """
        if ax is None:
            ax = plt.gca()
        d1 = self.stats[key1]
        d2 = self.stats[key2]
        if bins is None:
            b1 = np.arange(min(d1) - 0.5, max(d1) + 1.5)
            b2 = np.arange(min(d2) - 0.5, max(d2) + 1.5)
            bins = [b1, b2]
        im, ex, ey = np.histogram2d( d1, d2, bins=bins )
        ax.pcolor(ex, ey, im.T, cmap=cmap, **kwargs)
        if key1 == 'year':
            self._pprint_date_axis(ax, 'x')
        if key2 == 'year':
            self._pprint_date_axis(ax, 'y')
        ax.set_xlim(min(ex), max(ex))
        ax.set_ylim(min(ey), max(ey))
        ax.set_xlabel(self.labels[key1])
        ax.set_ylabel(self.labels[key2])
        plt.draw_if_interactive()

    def plot_ncited(self, ax=None, **kwargs):
        """
        Quick access to the plot of the distribution of number of citations per
        paper
        Equivalent to: self.hist('ncited',  ax=ax, **kwargs)
        """
        if ax is None:
            ax = plt.gca()
        self.hist('ncited',  ax=ax, **kwargs)

    def plot_nauthors(self, ax=None, **kwargs):
        """
        Quick access to the plot of the distribution of number of authors per
        paper
        Equivalent to: self.hist('nauthors',  ax=ax, **kwargs)
        """
        if ax is None:
            ax = plt.gca()
        self.hist('nauthors',  ax=ax, **kwargs)

    def plot_author_pos(self, ax=None, **kwargs):
        """
        Quick access to the plot of the distribution of author's positions per
        paper
        Equivalent to: self.hist('pos',  ax=ax, **kwargs)
        """
        if ax is None:
            ax = plt.gca()
        self.hist('pos',  ax=ax, **kwargs)

    def plot_year(self, ax=None, alpha=0.5, histtype='stepfilled', bins=None, **kwargs):
        """
        Quick access to the plot of the distribution of number of publications per year
        Axis format is updated to make nice year visualization
        Equivalent to: self.hist('year',  ax=ax, **kwargs)
        """
        if ax is None:
            ax = plt.gca()
        if bins is None:
            bins = np.arange(np.ptp(self.stats['year']) + 2) + self.stats['year'].min() - 0.5
        self.hist('year',  ax=ax, bins=bins, alpha=alpha, histtype=histtype, **kwargs)
        self._pprint_date_axis(ax, 'x')

    def _pprint_date_axis(self, ax, which='x', rotation=30, ha='right', margin=0.14, *args, **kwargs):
        """ Update one (or both) axis format when a date is represented
        INPUTS
        ------

        ax: matplotlib.Axes instance
            axis to update

        KEYWORDS
        --------

        which: string in ['x', 'y', 'both'] (default 'x')
            set to which axis to update

        rotation: float (default 30)
            angle of rotation to apply to the ticklabels

        ha: string in ['left', 'center', 'right'] (default 'right')
            horizontal alignement of the labels

        margin: float (default 0.14)
            update the margin of the plot to account for this update
        """
        if (which == 'x') | (which == 'both'):
            ax.xaxis.get_major_formatter().set_useOffset(False)
            for label in ax.get_xticklabels():
                    label.set_ha(ha)
                    label.set_rotation(rotation)
            plt.subplots_adjust(bottom=margin)
        elif (which == 'y') | (which == 'both'):
            ax.yaxis.get_major_formatter().set_useOffset(False)
            for label in ax.get_yticklabels():
                    label.set_ha(ha)
                    label.set_rotation(rotation)
            plt.subplots_adjust(left=margin)


import re
from collections import Counter
import difflib


class WordCloud(object):
    def __init__(self, txt, filter_common=True, group_similar=True):
        self.txt = txt.encode('ascii', 'ignore')
        self.words = None
        self.filter_common = filter_common
        self.group_similar = group_similar
        self.image = None

        self.common_words = """
            able about above according accordingly across actually after afterwards
            again against ain't all allow allows almost alone along already also
            although always am among amongst an and another any anybody anyhow
            anyone anything anyway anyways anywhere apart appear appreciate
            appropriate are aren't around as aside ask asking associated at
            available away awfully be became because become becomes becoming been
            before beforehand behind being believe below beside besides best better
            between beyond both brief but by c'mon c's came can can't cannot cant
            cause causes certain certainly changes clearly co com come comes
            concerning consequently consider considering contain containing
            contains corresponding could couldn't course currently definitely
            described despite did didn't different do does doesn't doing don't done
            down downwards during each edu eg eight either else elsewhere enough
            entirely especially et etc even ever every everybody everyone
            everything everywhere ex exactly example except far few fifth first
            five followed following follows for former formerly forth four from
            further furthermore get gets getting given gives go goes going gone got
            gotten greetings had hadn't happens hardly has hasn't have haven't
            having he he's hello help hence her here here's hereafter hereby herein
            hereupon hers herself hi him himself his hither hopefully how howbeit
            however i'd i'll i'm i've ie if ignored immediate in inasmuch inc
            indeed indicate indicated indicates inner insofar instead into inward
            is isn't it it'd it'll it's its itself just keep keeps kept know knows
            known last lately later latter latterly least less lest let let's like
            liked likely little look looking looks ltd mainly many may maybe me
            mean meanwhile merely might more moreover most mostly much must my
            myself name namely nd near nearly necessary need needs neither never
            nevertheless new next nine no nobody non none noone nor normally not
            nothing novel now nowhere obviously of off often oh ok okay old on once
            one ones only onto or other others otherwise ought our ours ourselves
            out outside over overall own particular particularly per perhaps placed
            please plus possible presumably probably provides que quite qv rather
            rd re really reasonably regarding regardless regards relatively
            respectively right said same saw say saying says second secondly see
            seeing seem seemed seeming seems seen self selves sensible sent serious
            seriously seven several shall she should shouldn't since six so some
            somebody somehow someone something sometime sometimes somewhat
            somewhere soon sorry specified specify specifying still sub such sup
            sure t's take taken tell tends th than thank thanks thanx that that's
            thats the their theirs them themselves then thence there there's
            thereafter thereby therefore therein theres thereupon these they they'd
            they'll they're they've think third this thorough thoroughly those
            though three through throughout thru thus to together too took toward
            towards tried tries truly try trying twice two un under unfortunately
            unless unlikely until unto up upon us use used useful uses using
            usually value various very via viz vs want wants was wasn't way we we'd
            we'll we're we've welcome well went were weren't what what's whatever
            when whence whenever where where's whereafter whereas whereby wherein
            whereupon wherever whether which while whither who who's whoever whole
            whom whose why will willing wish with within without won't wonder would
            would've wouldn't yes yet you you'd you'll you're you've your yours
            yourself yourselves zero a b c d e f g h i j k l m n o p q r s t u v w
            x y z present paper
            """.split()

        self.find_tags()

    def find_tags(self):

        # clean symbols and numbers
        if py3k:
            self.words = re.sub('[_:;,?.!()"~+-/*<>^1234567890]', ' ',
                                self.txt.decode('utf8').replace('\n', ' ')).lower()
        else:
            self.words = re.sub('[_:;,?.!()"~+-/*<>^1234567890]', ' ',
                                self.txt.replace('\n', ' ')).lower()

        # get rid of the common words in english
        if self.filter_common is True:
            w = self.words
            for wk in self.common_words:
                w = re.sub(' ' + wk + ' ', ' ', w)
            self.words = w.split()
        else:
            self.words = self.words.split()

        # find similar words
        if self.group_similar:
            sc = set(self.words)
            d = {}
            for k in sc:
                if not (k in self.common_words):
                    t = difflib.get_close_matches(k, sc, len(sc), 0.8)
                    bt = sum( tk in d.keys() for tk in t )
                    if bt == 0:
                        tt = t[ np.argmin( [len(tk) for tk in t] ) ]
                        d[tt] = t
            self.similar_words = d

            cc = {}
            for wk, lst in iteritems(d):
                cc[wk] = sum(self.words.count(lstk) for lstk in lst)
            self.words = Counter(cc)
        else:
            self.words = Counter(self.words)

    def make_image(self, **kwargs):
        if wordcloud is not None:
            k = np.asarray(list(self.words.keys()))
            v = np.asarray(list(self.words.values()))
            self.image = wordcloud.make_wordcloud(k, v, **kwargs)
