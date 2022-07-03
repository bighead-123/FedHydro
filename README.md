# FedHydro
## Description

### The source code of the FedHydro framework, which is built enviroment based on the framework at https://github.com/EngineerDDP/Parallel-SGD

## 提交者
### FedFRP相关提交者
  ./fed_hydro_3basins_t30_submit.py
  
  ./fed_hydro_6basins_submit.py
  
  ./fed_hydro_9basins_t30_submit.py
  
  ./fed_hydro_12basins_t30_210_submit.py

### FedSRP相关提交者:
  ./fedsrp_submit.py
  
  ./fedsrp_submit_01054200.py
  
  fedsrp_submit_01055000.py
  
  注：分别表示以流域01047000，流域01054200，流域01055000为数据稀缺流域时对应的提交者
  
## 数据集
### 数据集下载
  从XXX处下载CAMELS数据集，将其中的水文单元为01和03的气象及流量数据集放入./dataset/series_data/discharge_data 以及 ./dataset/series_data/forcing_data中

### 合并数据集
  **将多个流域的数据合并，生成符合框架使用规则的数据集**
  例如：将01单元中的6个流域数据
  
  


