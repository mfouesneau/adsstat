word cloud
==========

A little word cloud generator in Python.

Needs PIL, numpy and Cython (``>= 0.16``).

The example uses scikit-learn for extracting word counts from a text.
For scikit-learn ``<= 0.11``, you have to remove the ``min_df`` keyword.

Build using ``make`` or ``python setup.py build_ext -i``.

If you are not on Ubuntu, you need to adjust ``FONT_PATH`` to point to some
existing font.

See my blog for some details:
http://peekaboo-vision.blogspot.de/2012/11/a-wordcloud-in-python.html

For the blog post I removed the word ``shall`` from the constitution word count
which is not in the scikit-learn stopword list.


**Note**:
This version is slighlty different. `make_world_cloud` now returns the image
instead of just saving it and some options were added.
The default font file is provided with the package so that it is fully
standalone.
