# DLG assignment3: Model Design & Comparison for Recommendation Models
<!--https://latex.codecogs.com/eqneditor/editor.php-->
本作業將試著套用多種不同方法在不同資料集上。使用的方法主要分成兩類分別為第一類經典法及第二類以神經網路為主的相關方法。
在第二類的方法以下方方式執行
```r
python3 deepctr_main.py -dataname -modelname 
```

## Dataset Intro
資料集共三種分別為: `movielens`, `yelp` 及 `douban_book`

<table align="center">
    <thead>
        <tr>
            <th colspan=2> </th>
            <th align="center">Movielens</th>
            <th align="center">Yelp</th>
            <th align="center">Douban Book</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2> ID-range </td>
            <td>User</td>
            <td align="center"> [1,943] </td>
            <td align="center"> [1,16239] </td>
            <td align="center"> [1,13024] </td>
        </tr>
        <tr>
            <td>Item</td>
            <td align="center"> [1,1682] </td>
            <td align="center"> [1,14284] </td>
            <td align="center"> [1,22347] </td>
        </tr>
    </tbody>
</table>

## Measure criteria
- `RMSE`
- `Recall@10`
- `NDCG@10`

## Result

<table align="center">
    <thead>
        <tr>
            <th colspan=2 rowspan=2> </th>
            <th align="center" colspan=3>Movielens</th>
            <th align="center" colspan=3>Yelp</th>
            <th align="center" colspan=3>Douban Book</th>
        </tr>
        <tr>
            <th align="center">RMSE</th> <th align="center">Recall@10</th> <th align="center">NDCG@10</th>
            <th align="center">RMSE</th> <th align="center">Recall@10</th> <th align="center">NDCG@10</th>
            <th align="center">RMSE</th> <th align="center">Recall@10</th> <th align="center">NDCG@10</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=10> Typical </td>
            <td>UCF-s</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>UCF-p</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>ICF-s</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>ICF-p</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>MF</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>FM</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>BPR-MF</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>BPR-FM</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>GBDT+LR</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>XGB+LR</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td rowspan=10> NN-based </td>
            <td>FNN</td>
            <td> 0.5371 </td> <td> 0.0420 </td> <td> 0.0227 </td>
            <td> 0.5482 </td> <td> 0.1121 </td> <td> 0.00007 </td>
            <td> 0.3591 </td> <td> 0.0285 </td> <td> 0.00021 </td>
        </tr>
        <tr>
            <td>IPNN</td>
            <td> 0.5352 </td> <td> 0.0533 </td> <td> 0.0326 </td>
            <td> 0.5443 </td> <td> 0.1120 </td> <td> 0.00005 </td>
            <td> 0.3579 </td> <td> 0.0283 </td> <td> 0.00014 </td>
        </tr>
        <tr>
            <td>OPNN</td>
            <td> 0.5349 </td> <td> 0.0489 </td> <td> 0.0269 </td>
            <td> 0.5455 </td> <td> 0.1120 </td> <td> 0.00005 </td>
            <td> 0.3614 </td> <td> 0.0283 </td> <td> 0.00019 </td>
        </tr>
        <tr>
            <td>PIN</td>
            <td> 0.5348 </td> <td> 0.0516 </td> <td> 0.0320 </td>
            <td> 0.5467 </td> <td> 0.1121 </td> <td> 0.00006 </td>
            <td> 0.3581 </td> <td> 0.0283 </td> <td> 0.00016 </td>
        </tr>
        <tr>
            <td>CCPM</td>
            <td> 0.5361 </td> <td> 0.0338 </td> <td> 0.0120 </td>
            <td> 0.5466 </td> <td> 0.1121 </td> <td> 0.00009 </td>
            <td> 0.3641 </td> <td> 0.0284 </td> <td> 0.00018 </td>
        </tr>
        <tr>
            <td>NeuMF</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>WD</td>
            <td> 0.5357 </td> <td> 0.0391 </td> <td> 0.0176 </td>
            <td> 0.5474 </td> <td> 0.1121 </td> <td> 0.00007 </td>
            <td> 0.3596 </td> <td> 0.0283 </td> <td> 0.00007 </td>
        </tr>
        <tr>
            <td>DeepCorss</td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
            <td> x </td> <td> x </td> <td> x </td>
        </tr>
        <tr>
            <td>NFM</td>
            <td> 0.5396 </td> <td> 0.0335 </td> <td> 0.0104 </td>
            <td> 0.5438 </td> <td> 0.1120 </td> <td> 0.00007 </td>
            <td> 0.3649 </td> <td> 0.0285 </td> <td> 0.00018 </td>
        </tr>
        <tr>
            <td>DeepFM</td>
            <td> 0.5369 </td> <td> 0.0416 </td> <td> 0.0182 </td>
            <td> 0.5473 </td> <td> 0.1121 </td> <td> 0.00007 </td>
            <td> 0.3586 </td> <td> 0.0283 </td> <td> 0.00023 </td>
        </tr>
        <tr>
            <td rowspan=2> Recent NN-based </td>
            <td>xDeepFM</td>
            <td> 0.5359 </td> <td> 0.0431 </td> <td> 0.0163 </td>
            <td> 0.5423 </td> <td> 0.1120 </td> <td> 0.00003 </td>
            <td> 0.3580 </td> <td> 0.0283 </td> <td> 0.00013 </td>
        </tr>
        <tr>
            <td>AFM</td>
            <td> 0.5421 </td> <td> 0.0333 </td> <td> 0.0109 </td>
            <td> 0.5470 </td> <td> 0.1122 </td> <td> 0.00007 </td>
            <td> 0.3615 </td> <td> 0.0285 </td> <td> 0.00023 </td>
        </tr>
    </tbody>
</table>

## Methods

### Typical Methods
- `Collaborative Filtering (CF)`:協同過濾中主要分做 User-based 與 Item-based 兩種，而評估相似程度的方法中常見的有 Cosine Similarity 與 Pearson Correlation Coefficient 兩種，因此考慮不同種的協同過濾與不同的相似程度評估準則共有4種方法。分別為: User-based & Cosine Similarity(UCF-s)、Item-based & Cosine Similarity(ICF-s)、User-based & Pearson Correlation(UCF-p)、Item-based & Cosine Similarity(ICF-p)。
- `Matrix Factorization (MF)`:矩陣分解的推薦系統的核心概念認為用戶興趣主要被少數的因素所影響以及商品被選擇與否也是受到少數的因素影響。因此將評分矩陣(Rating Matrix)拆解，投射到低維度的矩陣的隱因子空間(Latent Factor Space)。主要運用奇異值分解法(Singular Value Decomposition, SVD)進行矩陣分解，將原本的評分矩陣拆解成使用者因子矩陣(User Factor Matrix)以及物品因子矩陣(Item Factor Matrix)。
- `Factorization Machine (FM)`:Factorization Machine在稀疏資料(Sparse Data)進行特徵交叉(Feature Interaction)並抽取出潛在因子(Latent Factor)，可在線性時間複雜度來進行訓練，且方便規模化。相較於簡易線性模型多考量了交互作用項，又比二階多項式迴歸(Degree-2 Polynomial Regression)更加具備泛化(Generalization)的能力。

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20y%28%5Cmathrm%7Bx%7D%29%20%26%3D%20w_0&plus;%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_ix_i%7D&plus;%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Csum_%7Bj%3Di&plus;1%7D%5E%7Bn%7D%7B%5Clangle%20%5Cmathrm%7Bv%7D_i%2C%20%5Cmathrm%7Bv%7D_j%20%5Crangle%20x_ix_j%7D%7D%20%5C%5C%20%26%3D%20w_0&plus;%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_ix_i%7D&plus;%5Csum_%7Bk%3D1%7D%5E%7BK%7D%7B%5Csum_%7Bq%3Dk&plus;1%7D%5E%7BK%7D%7B%5Clangle%20%5Cmathrm%7BW%7D%5E%7B%28k%29%7D%20%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%2C%20%5Cmathrm%7BW%7D%5E%7B%28q%29%7D%20%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_q%3A%5Cmathrm%7Bend%7D_q%5D%20%5Crangle%7D%7D%20%5Cend%7Baligned%7D">
  <br >
  </p>

<!---
$$\begin{aligned}
y(\mathrm{x}) &= w_0+\sum_{i=1}^{n}{w_ix_i}+\sum_{i=1}^{n}{\sum_{j=i+1}^{n}{\langle \mathrm{v}_i, \mathrm{v}_j \rangle x_ix_j}} \\ 
&= w_0+\sum_{i=1}^{n}{w_ix_i}+\sum_{k=1}^{K}{\sum_{q=k+1}^{K}{\langle \mathrm{W}^{(k)} \mathrm{x}[\mathrm{start}_k:\mathrm{end}_k], \mathrm{W}^{(q)} \mathrm{x}[\mathrm{start}_q:\mathrm{end}_q] \rangle}}
\end{aligned}
  $$
--->
  
### NN-based Methods
- `FM-supported Neural Network (FNN)`:
以Factorization Machine為基礎，將FM所產生的特徵向量，投入一個類神經網路中，以MLP(Multi Layers Perceptron)代替內積來進行預測任務。

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28%5Cmathrm%7Bconcat%7D%28%5BW%5E%7B%28k%29%7D%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%20%5C%2C%5C%2C%20%5Cmathrm%7Bfor%7D%20%5C%2C%5C%2C%20k%3D1%2C2%2C...%2CK%5D%29%29">
  <br >
  <img src="model_figure/FNN.png" width="450">
  </p>

- `Product-based Neural Networks (IPNN, OPNN)`:
比起FNN，在MLP的輸入加入每個field之間inner/outer product的特徵交叉。
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


- `Product Network in Network (PIN)`:
根據IPNN, OPNN進行延伸，比起加入每個field之間inner/outer product的特徵交叉在MLP的輸入，PIN考慮將兩兩field之間的特徵們輸入不同的子網路模型萃取特徵，最後在放入MLP。
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28%5Cmathrm%7Bcancat%7D%28%5B%5Cmathrm%7Bv%7D_1%2C%5Cmathrm%7Bv%7D_2%2C...%2C%5Cmathrm%7Bv%7D_K%2C%5Ctilde%7B%5Cmathrm%7Bp%7D%7D%5D%29%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Ctilde%7B%5Cmathrm%7Bp%7D%7D%3D%5Cmathrm%7Bconcat%7D%28%5B%5Ctilde%7B%5Cmathrm%7Bp%7D%7D_%7B1%2C2%7D%2C%5Ctilde%7B%5Cmathrm%7Bp%7D%7D_%7B1%2C3%7D%2C%20...%2C%20%5Ctilde%7B%5Cmathrm%7Bp%7D%7D_%7BK-1%2CK%7D%5D%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Ctilde%7B%5Cmathrm%7Bp%7D%7D_%7Bi%2Cj%7D%3D%5Cmathrm%7BsubMLP%7D%28%5Cmathrm%7Bconcat%7D%28%5B%5Cmathrm%7Bv%7D_i%2C%20%5Cmathrm%7Bv%7D_j%2C%20%5Cmathrm%7Bp%7D_%7Bi%2Cj%7D%5D%29%29">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bp%7D_%7Bi%2Cj%7D%3D%20%5Cmathrm%7Bv%7D_i%20%5Codot%20%5Cmathrm%7Bv%7D_j%20%5C%2C%5C%2C%5C%2C%5C%2C%20i%3Cj">
  <br >
  <img src="https://latex.codecogs.com/gif.latex?%5Cmathrm%7Bv%7D_k%3DW%5E%7B%28k%29%7D%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%2C%20%5C%2C%5C%2C%20k%3D1%2C2%2C...%2CK">
  </p>

<!---
$$y(\mathrm{x})=\mathrm{MLP}(\mathrm{cancat}([\mathrm{v}_1,\mathrm{v}_2,...,\mathrm{v}_K,\tilde{\mathrm{p}}]))$$

$$\tilde{\mathrm{p}}=\mathrm{concat}([\tilde{\mathrm{p}}_{1,2},\tilde{\mathrm{p}}_{1,3}, ..., \tilde{\mathrm{p}}_{K-1,K}])$$

$$\tilde{\mathrm{p}}_{i,j}=\mathrm{subMLP}(\mathrm{concat}([\mathrm{v}_i, \mathrm{v}_j, \mathrm{p}_{i,j}]))$$

$$\mathrm{p}_{i,j}= \mathrm{v}_i \odot \mathrm{v}_j  \,\,\,\, i<j$$

$$\mathrm{v}_k=W^{(k)}\mathrm{x}[\mathrm{start}_k:\mathrm{end}_k], \,\, k=1,2,...,K$$
--->

- `Convolutional Click Prediction Model (CCPM)`
<p align="center">
  <img src="model_figure/CCPM.jpg" width="450">
</p>

- `Neural Matrix Factorization (NeuMF)`

- `Wide & Deep (WD)`
<p align="center">
  <img src="model_figure/Wide&Deep.png" width="450">
</p>

- `Deep Crossing`

- `Neural Factorization Machine (NFM)`
<p align="center">
  <img src="model_figure/NFM.png" width="450">
</p>

- `Deep Factorization Machine (DeepFM)`
<p align="center">
  <img src="model_figure/DeepFM.png" width="450">
</p>

- `xDeepFM`
<p align="center">
  <img src="model_figure/xDeepFM.png" width="450">
</p>

- `Attentional Factorization Machine (AFM)`
<p align="center">
  <img src="model_figure/AFM.png" width="450">
</p>

