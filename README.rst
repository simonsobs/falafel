=======
falafel
=======


.. image:: https://img.shields.io/pypi/v/falafel.svg
        :target: https://pypi.python.org/pypi/falafel

.. image:: https://img.shields.io/travis/simonsobs/falafel.svg
        :target: https://travis-ci.org/simonsobs/falafel

.. image:: https://readthedocs.org/projects/falafel/badge/?version=latest
        :target: https://falafel.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




This library allows you to reconstruct CMB lensing maps on cut skies. It is SHT-based and uses a cylindrical projection in intermediate steps
which allows it to be fast and memory efficient. The implementation of the lensing quadratic estimators closely
follows that in the Planck 2018 lensing analysis by Julien Carron and others. SHT-based estimators for shear, point source and mask hardening have also been implemented by Frank Qu.

Only the unnormalized quadratic estimate is provided. For the full-sky normalization for various estimators, we recommend tempura (https://github.com/simonsobs/tempura).


* Free software: BSD license
* Documentation: https://falafel.readthedocs.io. (in progress)



