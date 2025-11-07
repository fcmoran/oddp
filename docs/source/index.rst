Oddp documentation
==================

In [1] three chain maps between four resolutions of the cyclic group were
explicitly described. The composition of these three maps gives explicit
chain formulas to compute higher cup products for odd primes. This package
implements these three maps, their duals and two algorithms to compute odd power operations
on simplicial complexes. The definition of one of the maps requires defining
a product in the negative part of the Tate periodic resolution of the cyclic
group. This product is implemented too.

References
----------
[1] Cantero-Morán and Medina-Mardones, Steenrod operations and Tate resolutions

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   oddp.objects
   oddp.tate_product
   oddp.chain_maps
   oddp.powers