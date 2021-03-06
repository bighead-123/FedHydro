# FedHydro
## Description

### The source code of the FedHydro framework, which builds enviroment based on the framework at https://github.com/EngineerDDP/Parallel-SGD

## 提交者
### FedFRP相关提交者
  - ./fedfrp_3basins_t30_submit.py
  - ./fedfrp_6basins_submit.py
  - ./fedfrp_9basins_t30_submit.py
  - ./fedfrp_12basins_t30_210_submit.py

### FedSRP相关提交者:
  - ./fedsrp_submit.py
  - ./fedsrp_submit_01054200.py
  - ./fedsrp_submit_01055000.py
  
  注：分别表示以流域01047000，流域01054200，流域01055000为数据稀缺流域时对应的提交者
  
## 数据集
### 数据集下载
  从https://ral.ucar.edu/solutions/products/camels 处下载CAMELS数据集，将其中的水文单元为01和03的气象及流量数据集放入./dataset/series_data/discharge_data以及 ./dataset/series_data/forcing_data中

### 合并数据集
  **将多个流域的数据合并，生成符合框架使用规则的数据集**
  
  例如：将01单元中的6个流域数据合并：
  
  - 执行./dataset/series_data/utils/generate_data.py，根据需要生成合并数据集（生成前注意修改数据集保存路径）
  - 将生成的数据集（txt格式）放入./dataset/series_data/下
  - 将./dataset/series_data/hydro_data_6basins.py中涉及到数据集路径的地方修改即可
  - 然后在./dataset/ 下创建对应的加载合并数据集的类（eg: hydro_data_6basins.py下的HydroDataSet2类）
  - 在提交者处会调用到加载合并数据集的类

  *注：./dataset/series_data/ 下有生成好的6个流域合并数据集，这可用于直接使用，其余的由于Github容量限制，可按照上述步骤生成合并数据集*

## 执行
- 搭建好分布环境（Docker搭建）
- 各个worker节点都执行./worker.py
- 提交者运行相应的提交者程序*_submit_*.py
- 执行完成后会在提交者上生成命名为形如Node-0-Retrieve（对应第一个节点）的结果文件
- 取出其中训练好的模型*.model进行测试
##
*如何搭建训练环境，具体可参考https://github.com/EngineerDDP/Parallel-SGD*

## 测试
- FedFRP相关实验在./dataset/code_test/hydro_lstm_test/save_script_.py 中
- FedSRP相关实验在./dataset/code_test/fedsrp/中.

  
  


