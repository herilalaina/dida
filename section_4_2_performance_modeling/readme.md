### Section 4.2

* Learning dataset cluster (w.r.t top-k distance) in ``main_cluster_prediction.py``

* Generate metafeatures from learned model ``calculate_metafeatures.py``

* Learned new representation of the latter metafeatures with ranking loss (use [allRank library](https://github.com/allegro/allRank) with ``local_config.json`` as configuration file)

* Handcrafted and landmark metafeatures are generated in ``calculate_hc_landmark_metafeatures.py``
