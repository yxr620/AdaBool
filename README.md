# AdaBool

This anonymous repository contains the AdaBool core implementation for sparse Boolean filtering workloads together with a minimal CUDA test suite.

## Repository Scope

Only the AdaBool implementation and a small set of end-to-end tests are retained here. Unrelated benchmarks and auxiliary experiments were removed to keep the anonymous release focused and self-contained.

## Dataset

The AdsFilter dataset is hosted on Google Drive:
https://drive.google.com/file/d/1WkSHuHe_B3gOBYCWkE8yxlTRTzTX7AeQ/view?usp=sharing

Download `adsfilter_dataset.tar.gz`, place it at the repository root, and extract it into `dataset/`:

```bash
tar -xzf adsfilter_dataset.tar.gz -C dataset/
```

After extraction you should have `dataset/adsfilter/filter.coo` and `dataset/adsfilter/query_vec_*.coo`. See [dataset/README.md](dataset/README.md) for details. The archive and extracted directory are git-ignored.

## Quick Start

### AdsFilter Query Test

Build and run the full query sweep on the anonymized AdsFilter dataset:

```bash
cd test
./compile_adsfilter_q_all.sh
```

Default inputs:
- Filter file: `../dataset/adsfilter/filter.coo`
- Query directory: `../dataset/adsfilter`

Custom inputs:

```bash
./compile_adsfilter_q_all.sh <filter_file> <query_dir>
```

The executable is generated at `test/bin/adsfilter_q_all`.

### Small Batch Test

Build and run the small-batch query test:

```bash
cd test
./compile_adsfilter_batch.sh
```

Default inputs:
- Filter file: `../dataset/adsfilter/filter.coo`
- Query directory: `../dataset/adsfilter`
- Batch size: `4`

Custom inputs:

```bash
./compile_adsfilter_batch.sh <filter_file> <query_dir> <batch_size>
```

The executable is generated at `test/bin/adsfilter_batch`.

## Dataset Notes

The anonymized dataset is referred to as AdsFilter throughout this repository. See the [Dataset](#dataset) section above for download and setup instructions.

## Directory Structure

- `src/AdaBool/` - AdaBool core implementation
- `dataset/` - dataset notes and local dataset layout
- `test/` - minimal query test programs and build scripts
