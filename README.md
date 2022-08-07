# Click-Through-Rate (CTR) prediction
## Model Design & Comparison for Recommendation Models
<!--https://latex.codecogs.com/eqneditor/editor.php-->
本專案將試著套用多種不同方法在不同資料集上。使用的方法主要分成兩類分別為第一類經典法及第二類以神經網路為主的相關方法。

在第一類的方法以下方方式執行
```r
python3 mf_main.py -dataname
```
```r
python3 fm_main.py -dataname
```

而第二類的方法以下方方式執行
```r
python3 deepctr_main.py -dataname -modelname 
```

### Dataset Intro
資料集共三種分別為: `movielens`, `yelp` 及 `douban_book`

- `movielens`:
  - User ID range: [1,943]
  - Item ID range: [1,1682]
  - User features: age, occupation
  - Item features: genre (multivalued)
- `yelp`:
  - User ID range: [1,16239]
  - Item ID range: [1,14284]
  - User features: compliment (multivalued)
  - Item features: city, category (multivalued)
- `douban_book`:
  - User ID range: [1,13024]
  - Item ID range: [1,22347]
  - User features: location, group (multivalued)
  - Item features: year, author, publisher

### Measure criteria
- `RMSE (real valued)`

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7BRMSE%7D%3D%5Csqrt%7B%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%28y_%7Bi%7D-%5Chat%7By%7D_%7Bi%7D%29%5E2%7D%7D">
  </p>

- `Recall@10 (binary)`:
Recall at 10 is the proportion of relevant items found in the top-10 recommendations.

  - Relevant item: Has a actual rating >= 3.5
  - Irrelevant item: Has a actual rating < 3.5

  <p align="center">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7BRecall@10%7D%3D%5Cfrac%7B%5Cmathrm%7BTotal%20%5C%2C%5C%2C%20number%20%5C%2C%5C%2C%20of%20%5C%2C%5C%2C%20recommended%20%5C%2C%5C%2C%20items%20%5C%2C%5C%2C%20@10%20%5C%2C%5C%2C%20that%20%5C%2C%5C%2C%20are%20%5C%2C%5C%2C%20relevant%7D%7D%7B%5Cmathrm%7BTotal%20%5C%2C%5C%2C%20number%20%5C%2C%5C%2C%20of%20%5C%2C%5C%2C%20relevant%20%5C%2C%5C%2C%20items%7D%7D">
  </p>

- `NDCG@10 (binary)`:NDCG is a measure of ranking quality.

  <p align="center">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7BNDCG@10%7D%3D%5Cfrac%7B%5Cmathrm%7BDCG@10%7D%7D%7B%5Cmathrm%7BiDCG@10%7D%7D">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7BDCG@10%7D%3D%5Csum_%7Bi%3D1%7D%5E%7B10%7D%7B%5Cfrac%7BI_%7B%5Cmathrm%7Bitem%20%5C%2C%5C%2Cis%20%5C%2C%5C%2C%20relevant%7D%7D%28%5Cmathrm%7Brecommended%20%5C%2C%5C%2C%20item%7D%20%5C%2C%5C%2C%20i%29%7D%7Blog_2%28i&plus;1%29%7D%7D">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7BiDCG@10%7D%3D%5Csum_%7Bi%3D1%7D%5E%7B10%7D%7B%5Cfrac%7B1%7D%7Blog_2%28i&plus;1%29%7D%7D">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?I_%7B%5C%7B%5Cmathrm%7Bcondition%7D%5C%7D%7D%28.%29%3A%5Cmathrm%7Bindicator%20%5C%2C%5C%2C%20function%7D">
  </p>

<!---
$$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}{(y_{i}-\hat{y}_{i})^2}}$$

$$\mathrm{Recall@10}=\frac{\mathrm{Total \,\, number \,\, of \,\, recommended \,\, items \,\, @10 \,\, that \,\, are \,\, relevant}}{\mathrm{Total \,\, number \,\, of \,\, relevant \,\, items}}$$

$$\mathrm{NDCG@10}=\frac{\mathrm{DCG@10}}{\mathrm{iDCG@10}}$$

$$\mathrm{DCG@10}=\sum_{i=1}^{10}{\frac{I_{\mathrm{item \,\,is \,\, relevant}}(\mathrm{recommended \,\, item} \,\, i)}{log_2(i+1)}}$$

$$\mathrm{iDCG@10}=\sum_{i=1}^{10}{\frac{1}{log_2(i+1)}}$$

$$I_{\{\mathrm{condition}\}}(.):\mathrm{indicator \,\, function}$$
--->

### Methods

#### Typical Methods

- `Matrix Factorization (MF)`:\
矩陣分解的推薦系統的核心概念認為用戶興趣主要被少數的因素所影響以及商品被選擇與否也是受到少數的因素影響。因此將評分矩陣(Rating Matrix)拆解，投射到低維度的矩陣的隱因子空間(Latent Factor Space)。主要運用奇異值分解法(Singular Value Decomposition, SVD)進行矩陣分解，將原本的評分矩陣拆解成使用者因子矩陣(User Factor Matrix)以及物品因子矩陣(Item Factor Matrix)。
- `Factorization Machine (FM)`:\
Factorization Machine在稀疏資料(Sparse Data)進行特徵交叉(Feature Interaction)並抽取出潛在因子(Latent Factor)，可在線性時間複雜度來進行訓練，且方便規模化。相較於簡易線性模型多考量了交互作用項，又比二階多項式迴歸(Degree-2 Polynomial Regression)更加具備泛化(Generalization)的能力。

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20y%28%5Cmathrm%7Bx%7D%29%20%26%3D%20%7Bw_0%7D%20&plus;%20%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_i%20x_i%7D%7D%20&plus;%20%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%7B%5Cleft%5Clangle%20%5Cmathrm%7Bv%7D_%7Bi%7D%2C%20%5Cmathrm%7Bv%7D_%7Bj%7D%20%5Cright%5Crangle%20x_i%20x_j%20%7D%7D%7D%20%5C%5C%20%26%3D%20%7Bw_0%7D%20&plus;%20%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cmathrm%7Bw%7D_i%20x_i%7D%7D%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%5B%20%5Cleft%5Clangle%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bx_i%20%5Cmathrm%7Bv%7D_i%7D%20%2C%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bx_i%20%5Cmathrm%7Bv%7D_i%7D%20%5Cright%5Crangle%20-%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cleft%5Clangle%20x_i%20%5Cmathrm%7Bv%7D_i%20%2C%20x_i%20%5Cmathrm%7Bv%7D_i%20%5Cright%5Crangle%7D%20%5Cright%5D%20%5C%5C%20%5Cend%7Baligned%7D">
  </p >

  since
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cleft%5Clangle%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bx_i%20%5Cmathrm%7Bv%7D_i%7D%20%2C%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bx_i%20%5Cmathrm%7Bv%7D_i%7D%5Cright%5Crangle%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cleft%5Clangle%20x_i%20%5Cmathrm%7Bv%7D_i%20%2C%20x_i%20%5Cmathrm%7Bv%7D_i%20%5Cright%5Crangle%7D%20&plus;%202%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%7B%5Cleft%5Clangle%20x_i%20%5Cmathrm%7Bv%7D_i%20%2C%20x_j%20%5Cmathrm%7Bv%7D_j%20%5Cright%5Crangle%7D%7D">
  </p >
  then
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%7B%5Cleft%5Clangle%20x_i%20%5Cmathrm%7Bv%7D_i%20%5C%2C%20%2C%20%5C%2C%20x_j%20%5Cmathrm%7Bv%7D_j%20%5Cright%5Crangle%7D%7D%3D%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%5B%20%5Cleft%5Clangle%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bx_i%20%5Cmathrm%7Bv%7D_i%7D%20%5C%2C%20%2C%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bx_i%20%5Cmathrm%7Bv%7D_i%7D%20%5Cright%5Crangle%20-%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cleft%5Clangle%20x_i%20%5Cmathrm%7Bv%7D_i%20%5C%2C%20%2C%20%5C%2C%20x_i%20%5Cmathrm%7Bv%7D_i%20%5Cright%5Crangle%7D%20%5Cright%5D">

  In another way, we can treat $\mathrm{v}_i, \mathrm{v}_j$ as output from multi-filed embedding layers.
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20y%28%5Cmathrm%7Bx%7D%29%20%26%3D%20w_0&plus;%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_ix_i%7D&plus;%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%7B%5Cleft%5Clangle%20%5Cmathrm%7Bv%7D_i%2C%20%5Cmathrm%7Bv%7D_j%20%5Cright%5Crangle%20x_ix_j%7D%7D%20%5C%5C%20%26%3D%20w_0&plus;%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_ix_i%7D&plus;%5Csum_%7Bk%3D1%7D%5E%7BK%7D%7B%5Csum_%7Bq%3Dk&plus;1%7D%5E%7BK%7D%7B%5Cleft%5Clangle%20%5Cmathrm%7BW%7D%5E%7B%28k%29%7D%20%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%2C%20%5Cmathrm%7BW%7D%5E%7B%28q%29%7D%20%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_q%3A%5Cmathrm%7Bend%7D_q%5D%20%5Cright%5Crangle%7D%7D%20%5C%5C%20%5Cend%7Baligned%7D">


<!--
$$\begin{aligned}
y(\mathrm{x}) 
&= {w_0} + {\sum_{i=1}^{n}{w_i x_i}} + {\sum_{i=1}^{n}{\sum_{j=i+1}^{n}{\left\langle \mathrm{v}_{i}, \mathrm{v}_{j} \right\rangle x_i x_j }}} \\
&= {w_0} + {\sum_{i=1}^{n}{\mathrm{w}_i x_i}} + \frac{1}{2} \left[ \left\langle \sum_{i=1}^{n}{x_i \mathrm{v}_i} , \sum_{i=1}^{n}{x_i \mathrm{v}_i} \right\rangle - \sum_{i=1}^{n}{\left\langle x_i \mathrm{v}_i , x_i \mathrm{v}_i \right\rangle} \right] \\
\end{aligned}
$$

$$\left\langle \sum_{i=1}^{n}{x_i \mathrm{v}_i} , \sum_{i=1}^{n}{x_i \mathrm{v}_i}\right\rangle = \sum_{i=1}^{n}{\left\langle x_i \mathrm{v}_i , x_i \mathrm{v}_i \right\rangle} + 2 \sum_{i=1}^{n}{\sum_{j=i+1}^{n}{\left\langle x_i \mathrm{v}_i , x_j \mathrm{v}_j \right\rangle}}$$

$$\sum_{i=1}^{n}{\sum_{j=i+1}^{n}{\left\langle x_i \mathrm{v}_i \, , \, x_j \mathrm{v}_j \right\rangle}}=\frac{1}{2} \left[ \left\langle \sum_{i=1}^{n}{x_i \mathrm{v}_i} \, ,\,  \sum_{i=1}^{n}{x_i \mathrm{v}_i} \right\rangle - \sum_{i=1}^{n}{\left\langle x_i \mathrm{v}_i \, , \, x_i \mathrm{v}_i \right\rangle} \right]$$

$$\begin{aligned}
y(\mathrm{x}) &= w_0+\sum_{i=1}^{n}{w_ix_i}+\sum_{i=1}^{n}{\sum_{j=i+1}^{n}{\left\langle \mathrm{v}_i, \mathrm{v}_j \right\rangle x_ix_j}} \\ 
&= w_0+\sum_{i=1}^{n}{w_ix_i}+\sum_{k=1}^{K}{\sum_{q=k+1}^{K}{\left\langle \mathrm{W}^{(k)} \mathrm{x}[\mathrm{start}_k:\mathrm{end}_k], \mathrm{W}^{(q)} \mathrm{x}[\mathrm{start}_q:\mathrm{end}_q] \right\rangle}} \\
\end{aligned}
$$

-->

#### NN-based Methods
- `FM-supported Neural Network (FNN)`:\
以 Factorization Machine(FM) 為基礎，將 FM 所產生的特徵向量，投入一個類神經網路中，以 Multi-Layer Perceptron(MLP) 代替內積來進行預測任務。

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28%5Cmathrm%7Bconcat%7D%28%5BW%5E%7B%28k%29%7D%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%20%5C%2C%5C%2C%20%5Cmathrm%7Bfor%7D%20%5C%2C%5C%2C%20k%3D1%2C2%2C...%2CK%5D%29%29">
  <br >
  <img src="model_figure/FNN.png" width="450">
  </p>

- `Product-based Neural Networks (IPNN, OPNN)`:\
比起 FNN ，在 MLP 的輸入加入每個 field 之間 inner/outer product 的特徵交叉。
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28%5Cmathrm%7Bcancat%7D%28%5B%5Cmathrm%7Bv%7D_1%2C%5Cmathrm%7Bv%7D_2%2C...%2C%5Cmathrm%7Bv%7D_K%2C%5Cmathrm%7Bp%7D%5D%29%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bp%7D%3D%5Cmathrm%7Bconcat%7D%28%5B%5Cmathrm%7Bflatten%7D%28%5Cmathrm%7Bp%7D_%7B1%2C2%7D%29%2C%5Cmathrm%7Bflatten%7D%28%5Cmathrm%7Bp%7D_%7B1%2C3%7D%29%2C%20...%2C%20%5Cmathrm%7Bflatten%7D%28%5Cmathrm%7Bp%7D_%7BK-1%2CK%7D%29%5D%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bp%7D_%7Bi%2Cj%7D%3D%5Cbegin%7Bcases%7D%20%5Clangle%20%5Cmathrm%7Bv%7D_i%2C%5Cmathrm%7Bv%7D_j%20%5Crangle%20%26%20%5Ctext%7Bif%20inner%20product%7D%20%5C%5C%20%5Cmathrm%7Bv%7D_i%20%5Cotimes%20%5Cmathrm%7Bv%7D_j%20%26%20%5Ctext%7Bif%20outer%20product%7D%20%5Cend%7Bcases%7D">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bv%7D_k%3DW%5E%7B%28k%29%7D%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%2C%20%5C%2C%5C%2C%20k%3D1%2C2%2C...%2CK">
  <br >
  <img src="model_figure/PNN.png" width="450">
  </p>

<!--
$$y(\mathrm{x})=\mathrm{MLP}(\mathrm{cancat}([\mathrm{v}_1,\mathrm{v}_2,...,\mathrm{v}_K,\mathrm{p}]))$$
$$\mathrm{p}=\mathrm{concat}([\mathrm{flatten}(\mathrm{p}_{1,2}),\mathrm{flatten}(\mathrm{p}_{1,3}), ..., \mathrm{flatten}(\mathrm{p}_{K-1,K})])$$
$$
\mathrm{p}_{i,j}=\begin{cases}
\langle \mathrm{v}_i,\mathrm{v}_j \rangle & \text{if inner product} \\
\mathrm{v}_i \otimes \mathrm{v}_j & \text{if outer product}
\end{cases}
$$
$$\mathrm{v}_k=W^{(k)}\mathrm{x}[\mathrm{start}_k:\mathrm{end}_k], \,\, k=1,2,...,K$$
-->


- `Product Network in Network (PIN)`:\
根據IPNN, OPNN進行延伸，比起加入每個field之間inner/outer product的特徵交叉在MLP的輸入，PIN考慮將兩兩field之間的特徵們輸入不同的子網路模型萃取特徵，最後在放入MLP。
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28%5Cmathrm%7Bcancat%7D%28%5B%5Cmathrm%7Bv%7D_1%2C%5Cmathrm%7Bv%7D_2%2C...%2C%5Cmathrm%7Bv%7D_K%2C%5Ctilde%7B%5Cmathrm%7Bp%7D%7D%5D%29%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Ctilde%7B%5Cmathrm%7Bp%7D%7D%3D%5Cmathrm%7Bconcat%7D%28%5B%5Ctilde%7B%5Cmathrm%7Bp%7D%7D_%7B1%2C2%7D%2C%5Ctilde%7B%5Cmathrm%7Bp%7D%7D_%7B1%2C3%7D%2C%20...%2C%20%5Ctilde%7B%5Cmathrm%7Bp%7D%7D_%7BK-1%2CK%7D%5D%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Ctilde%7B%5Cmathrm%7Bp%7D%7D_%7Bi%2Cj%7D%3D%5Cmathrm%7BsubMLP%7D_%7Bi%2Cj%7D%28%5Cmathrm%7Bconcat%7D%28%5B%5Cmathrm%7Bv%7D_i%2C%20%5Cmathrm%7Bv%7D_j%2C%20%5Cmathrm%7Bp%7D_%7Bi%2Cj%7D%5D%29%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bp%7D_%7Bi%2Cj%7D%3D%20%5Cmathrm%7Bv%7D_i%20%5Codot%20%5Cmathrm%7Bv%7D_j%20%5C%2C%5C%2C%5C%2C%5C%2C%20i%3Cj">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bv%7D_k%3DW%5E%7B%28k%29%7D%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%2C%20%5C%2C%5C%2C%20k%3D1%2C2%2C...%2CK">
  </p>

<!---
$$y(\mathrm{x})=\mathrm{MLP}(\mathrm{cancat}([\mathrm{v}_1,\mathrm{v}_2,...,\mathrm{v}_K,\tilde{\mathrm{p}}]))$$

$$\tilde{\mathrm{p}}=\mathrm{concat}([\tilde{\mathrm{p}}_{1,2},\tilde{\mathrm{p}}_{1,3}, ..., \tilde{\mathrm{p}}_{K-1,K}])$$

$$\tilde{\mathrm{p}}_{i,j}=\mathrm{subMLP}_{i,j}(\mathrm{concat}([\mathrm{v}_i, \mathrm{v}_j, \mathrm{p}_{i,j}]))$$

$$\mathrm{p}_{i,j}= \mathrm{v}_i \odot \mathrm{v}_j  \,\,\,\, i<j$$

$$\mathrm{v}_k=W^{(k)}\mathrm{x}[\mathrm{start}_k:\mathrm{end}_k], \,\, k=1,2,...,K$$
--->

- `Convolutional Click Prediction Model (CCPM)`:\
此模型最主要的特色是將卷積(Convolution)的概念納入CTR的預測模型。
  - 用法一：考慮每個人在不同的時間軸上會有不同的特徵(Features)，透過 *1*-D Conv搭配 *p*-max 池化層(*p*-max pooling)來擷取不同時間(Temporal)的資訊。 *p*-max 池化層主要在處理每個人在與Item的互動時間、次數並不一致而無法單純使用原始的最大池化層進行。
  - 用法二：沒有不同時間的互動關係，改把不同Filed的資訊併在一起，改成不同Filed的embedding vector在相同維度上進行卷積運算然後取 *p*-max。

  <p align="center">
  <img src="model_figure/CCPM.jpg" width="450">
  </p>


- `Wide & Deep (WD)`:\
WD模型最主要探討推薦系統模型的兩個挑戰:

  - Memorization(記憶): 模型是否可以記住歷史資料的互動組合進行預測?
  - Generalization(泛化): 模型可否能搓合出新的特徵組合增加預測結果的多樣性?
  
  在Memorization上可以透過Logistic regression來進行(下圖左邊的Wide models)。利用Logistic regression便可以學習各項特徵對於預測結果之間的關係(此部分的輸入特徵會需要額外人工的特徵工程 ex.cross-product transformation)。而在Generalization上可以透過Deep Network來擷取更高階的特徵組合(下圖右邊的Deep models)，進一步結合出新的特徵組合增加預測結果的多樣性。WD模型透過結合Wide models及Deep models來同時考慮Memorization以及Generalization(下圖中間的部分)。

  <p align="center">
    <img src="model_figure/Wide&Deep.png" width="750">
  </p>

- `Neural Factorization Machine (NFM)`:\
NFM模型主要是基於FM模型進行改良，在原始的FM模型中，它能充分考慮特徵之間的二階交互作用，但僅以線性的方式加在模型內，並不能無法考量特徵之間的非線性關係。因此NFM在這點上進行調整，並且充分?合了FM提取的二階線性特徵與神經網路模型提取高階非線性特徵。

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3Dw_0&plus;%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_ix_i%7D&plus;f%28%5Cmathrm%7Bx%7D%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?f%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28f_%7B%5Cmathrm%7BBI%7D%7D%28%5Cmathrm%7Bx%7D%29%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?f_%7B%5Cmathrm%7BBI%7D%7D%28%5Cmathrm%7Bx%7D%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%7Bx_i%20%5Cmathrm%7Bv%7D_i%20%5Codot%20x_j%20%5Cmathrm%7Bv%7D_j%7D%7D">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bv%7D_i%3A%5Cmathrm%7BEmbedding%20%5C%2C%5C%2C%20vector%20%5C%2C%5C%2C%20of%20%5C%2C%5C%2C%20features%20%5C%2C%5C%2C%7D%20i">
  <br >
  <img src="model_figure/NFM.png" width="450">  
  </p>

<!---
$$y(\mathrm{x})=w_0+\sum_{i=1}^{n}{w_ix_i}+f(\mathrm{x})$$
$$f(\mathrm{x})=\mathrm{MLP}(f_{\mathrm{BI}}(\mathrm{x}))$$
$$f_{\mathrm{BI}}(\mathrm{x})=\sum_{i=1}^{n}{\sum_{j=i+1}^{n}{x_i \mathrm{v}_i \odot x_j \mathrm{v}_j}}$$
$$\mathrm{v}_i:\mathrm{Embedding \,\, vector \,\, of \,\, features \,\,} i$$
--->

- `Attentional Factorization Machine (AFM)`:\
FM模型能夠考量特徵之間的二階交互作用，但是所有特徵的權重都是相同的，這樣的處理方式或許並不恰當因為不是所有的特徵都是有用的，當有無用的特徵進行組合又加到預測模型中其實會帶入噪聲(Noise)的干擾，降低FM的效果，因此AFM基於這樣的觀點引入Attention的機制，讓模型有能力自行調整特徵交互作用的重要程度。

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3Dw_0&plus;%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_ix_i%7D&plus;f%28%5Cmathrm%7Bx%7D%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?f%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28f_%7B%5Cmathrm%7BBIAtt%7D%7D%28%5Cmathrm%7Bx%7D%29%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?f_%7B%5Cmathrm%7BBIAtt%7D%7D%28%5Cmathrm%7Bx%7D%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%7Ba_%7Bi%2Cj%7D%20%5Ctimes%20%28x_i%20%5Cmathrm%7Bv%7D_i%20%5Codot%20x_j%20%5Cmathrm%7Bv%7D_j%29%7D%7D">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?a_%7Bi%2Cj%7D%3D%5Cfrac%7B%5Cmathrm%7Bexp%7D%28a_%7Bi%2Cj%7D%27%29%7D%7B%5Csum_%7B%28i%2Cj%29%7D%7B%5Cmathrm%7Bexp%7D%28a_%7Bi%2Cj%7D%27%29%7D%7D">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?a_%7Bi%2Cj%7D%27%3D%5Cmathrm%7BMLP%7D_%7B%5Cmathrm%7BAttention%7D%7D%28x_i%20%5Cmathrm%7Bv%7D_i%20%5Codot%20x_j%20%5Cmathrm%7Bv%7D_j%29">
  <br >
  <img src="model_figure/AFM.png" width="450">
  </p>

<!---
$$y(\mathrm{x})=w_0+\sum_{i=1}^{n}{w_ix_i}+f(\mathrm{x})$$
$$f(\mathrm{x})=\mathrm{MLP}(f_{\mathrm{BIAtt}}(\mathrm{x}))$$
$$f_{\mathrm{BIAtt}}(\mathrm{x})=\sum_{i=1}^{n}{\sum_{j=i+1}^{n}{a_{i,j} \times (x_i \mathrm{v}_i \odot x_j \mathrm{v}_j)}}$$
$$a_{i,j}=\frac{\mathrm{exp}(a_{i,j}')}{\sum_{(i,j)}{\mathrm{exp}(a_{i,j}')}}$$
$$a_{i,j}'=\mathrm{MLP}_{\mathrm{Attention}}(x_i \mathrm{v}_i \odot x_j \mathrm{v}_j)$$
--->

- `Deep Factorization Machine (DeepFM)`:\
DeepFM模型可以視為WD模型的改進版，在WD模型使用Logistic regression來學習各項特徵對於預測結果之間的關係，而在推薦系統當中往，資料往會是高維且稀疏的，若Logistic regression需要考慮二階交互作用項容易導致訓練結果不正確，因此將其替換Factorization machine(FM)恰好能解決這樣的問題，並且如此一來也能夠省去WD模型在Wide part需要額外的特徵工程，都可用相同的Embedding vector來作為輸入。
  <p align="center">
    <img src="model_figure/DeepFM.png" width="450">
  </p>

- `Deep Crossing (DCN)`:\
在Wide & Deep模型實現了同時考慮Memorization及Generalization。但在Wide最大的問題就是需要人工的特徵工程來設計特徵之間叉乘。DeepFM雖然在Wide的部分以FM進行，但也僅限組合特徵的二階交互關係。因此Deep Crossing的核心概念在於讓模型自動去學習特徵之間的高階交互關係。在下圖的左半邊照舊是一個Deep Network主要處理Generalization使模型在預測上考量組合的多樣性，而在右邊為Cross Network。Cross Network主要流程如下:

  <p align="center">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bx%7D_%7Bl&plus;1%7D%3D%28%5Cmathrm%7Bx%7D_%7Bl%7D%5Cotimes%5Cmathrm%7Bx%7D_%7Bl%7D%29%5Cmathrm%7Bw%7D_%7Bl%7D&plus;b_%7Bl%7D&plus;%5Cmathrm%7Bx%7D_%7Bl%7D%20%5C%2C%5C%2C%5C%2C%20l%3D0%2C1%2C...L">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cotimes%3A%20%5Cmathrm%7Bout%20%5C%2C%5C%2C%20product%7D">
  <br >
  <img src="model_figure/DCN.png" width="450">
  </p>

<!---
$$\mathrm{x}_{l+1}=(\mathrm{x}_{l}\otimes\mathrm{x}_{l})\mathrm{w}_{l}+b_{l}+\mathrm{x}_{l} \,\,\, l=0,1,...L$$

$$\otimes: \mathrm{out \,\, product}$$
--->

- `xDeepFM`
  <p align="center">
    <img src="model_figure/xDeepFM.png" width="450">
  </p>

