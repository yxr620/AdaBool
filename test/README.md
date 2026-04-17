# Test Directory

测试程序目录，仅保留 AdaBool 相关的最小查询测试。

## 脚本说明

### compile_adsfilter_q_all.sh
编译并运行 AdsFilter 数据集上的全部查询测试（密集和稀疏方法）。

**用法：**
```bash
# 使用默认路径
./compile_adsfilter_q_all.sh

# 自定义路径
./compile_adsfilter_q_all.sh <filter_file> <query_dir>
```

**默认参数：**
- filter_file: `../dataset/adsfilter/filter.coo`
- query_dir: `../dataset/adsfilter`

---

### compile_adsfilter_batch.sh
编译并运行小批量查询测试。

**用法：**
```bash
# 使用默认路径
./compile_adsfilter_batch.sh

# 自定义路径和批量大小
./compile_adsfilter_batch.sh <filter_file> <query_dir> <batch_size>
```

**默认参数：**
- filter_file: `../dataset/adsfilter/filter.coo`
- query_dir: `../dataset/adsfilter`
- batch_size: `4`

---

## 输出文件

编译后的可执行文件位于 `./bin/` 目录：
- `bin/adsfilter_q_all`
- `bin/adsfilter_batch`
