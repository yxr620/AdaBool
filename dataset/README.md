# AdsFilter Dataset

The AdsFilter dataset used by the tests in this repository is hosted on Google Drive:

- Download link: https://drive.google.com/file/d/1WkSHuHe_B3gOBYCWkE8yxlTRTzTX7AeQ/view?usp=sharing

## How to install

1. Download the archive `adsfilter_dataset.tar.gz` from the link above.
2. Place it at the repository root (alongside this `dataset/` folder):

   ```
   <repo_root>/adsfilter_dataset.tar.gz
   ```

3. Extract it into `dataset/`:

   ```bash
   tar -xzf adsfilter_dataset.tar.gz -C dataset/
   ```

4. After extraction the layout should be:

   ```
   dataset/
   └── adsfilter/
       ├── filter.coo
       ├── query_vec_0.coo
       ├── query_vec_1.coo
       ├── query_vec_2.coo
       └── ...
   ```

Both `adsfilter_dataset.tar.gz` and `dataset/adsfilter/` are listed in `.gitignore` and will not be committed.

## How to use

Once the dataset is in place, the test programs under `test/` will pick it up automatically with their default arguments. See the top-level [README](../README.md) for build and run instructions.