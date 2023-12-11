=======
falafel
=======


.. image:: https://img.shields.io/pypi/v/falafel.svg
        :target: https://pypi.python.org/pypi/falafel

.. image:: https://img.shields.io/travis/simonsobs/falafel.svg
        :target: https://travis-ci.org/simonsobs/falafel


This library allows you to reconstruct mode-couplings (e.g. CMB lensing, kSZ) on cut skies. It is SHT-based and uses a cylindrical projection in intermediate steps which allows it to be fast and memory efficient. The implementation of the lensing quadratic estimators closely follows that in the Planck 2018 lensing analysis by Julien Carron and others. SHT-based estimators for shear, point source and mask hardening have also been implemented by Frank Qu.

Only the unnormalized quadratic estimate is provided. For the full-sky normalization for various estimators, we recommend tempura (https://github.com/simonsobs/tempura).

Installation
------------

Use

.. code-block:: console

    python setup.py develop

(for in-place development)

or 

.. code-block:: console

    python setup.py install


(Note that ``pip install`` may not work)

A good place to start with the code is `tests/simple.py`.



