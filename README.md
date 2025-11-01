# AdaBool: Adaptive Sparse Boolean Matrix Processing with Hybrid Data Representation

This repository implements AdaBool, an adaptive system for efficient sparse Boolean matrix processing.

## Quick Start

### Tencent Dataset Test

The project includes inverted index query tests on the Tencent dataset.

**Compile and run all query tests:**
```bash
cd test
./compile_tencent_q_all.sh
```

**Default parameters:**
- Filter file: `../dataset/tencent/filter.coo`
- Query directory: `../dataset/tencent`

**Custom paths:**
```bash
./compile_tencent_q_all.sh <filter_file> <query_dir>
```

The compiled executable will be located at `test/bin/tencent_q_all`.

## Directory Structure

- `src/AdaBool/` - Core implementation
- `dataset/tencent/` - Tencent dataset files (filter and query vectors)
- `test/` - Test programs and compilation scripts
