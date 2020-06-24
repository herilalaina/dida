import glob
import numpy as np
import pandas as pd

from metalearn import Metafeatures


def get_list_metafeatures(list_X, list_y, type_metafeatures):
    metafeatures = Metafeatures()
    list_dataset_metafeatures = []

    for X, y in tqdm(zip(list_X, list_Y), total=7084):
        mfs = metafeatures.compute(
                        pd.DataFrame(X),
                        Y=pd.Series(y, dtype="category"),
                        metafeature_ids=metafeatures.list_metafeatures(group=type_metafeatures),
                        exclude=None,
                        seed=0,
                        #verbose=True,
                        timeout=60,
                        # return_times=True,
                    )
        list_dataset_metafeatures.append(pd.DataFrame(mfs).reset_index(drop=True))

    df_metafeatures = pd.concat(list_dataset_metafeatures).fillna(0)
    df_metafeatures["index"] = list_files
    df_metafeatures.set_index("index", inplace=True)
    return df_metafeatures



if __name__ == "__main__":
    scores = []
    list_files = []
    list_X = []
    list_Y= []

    for i in tqdm(glob.glob("../../datasets/openml/version_3D/scores/seed_4*/*")):
        res = np.load(i)
        scores.append(res)
        list_files.append(i)
        list_X.append(np.load(i.replace("scores/seed_4", "seed_4").replace("y.", "x.")))
        list_Y.append(np.load(i.replace("scores/seed_4", "seed_4")))

    meta_hc = get_list_metafeatures(list_X, list_Y, "simple")
    meta_landmark = get_list_metafeatures(list_X, list_Y, "landmarking")

    meta_hc.to_csv("metafeatures_handcrafted.csv")
    meta_landmark.to_csv("metafeatures_landmarking.csv")
