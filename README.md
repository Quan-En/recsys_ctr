# Click-Through-Rate (CTR) prediction
## Model Design & Comparison for Recommendation Models
<!--https://latex.codecogs.com/eqneditor/editor.php-->
���M�ױN�յۮM�Φh�ؤ��P��k�b���P��ƶ��W�C�ϥΪ���k�D�n�����������O���Ĥ@���g��k�βĤG���H���g�������D��������k�C

�b�Ĥ@������k�H�U��覡����
```r
python3 mf_main.py -dataname
```
```r
python3 fm_main.py -dataname
```

�ӲĤG������k�H�U��覡����
```r
python3 deepctr_main.py -dataname -modelname 
```

### Dataset Intro
��ƶ��@�T�ؤ��O��: `movielens`, `yelp` �� `douban_book`

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
�x�}���Ѫ����˨t�Ϊ��֤߷����{���Τῳ��D�n�Q�ּƪ��]���Ҽv�T�H�ΰӫ~�Q��ܻP�_�]�O����ּƪ��]���v�T�C�]���N�����x�}(Rating Matrix)��ѡA��g��C���ת��x�}�����]�l�Ŷ�(Latent Factor Space)�C�D�n�B�Ω_���Ȥ��Ѫk(Singular Value Decomposition, SVD)�i��x�}���ѡA�N�쥻�������x�}��Ѧ��ϥΪ̦]�l�x�}(User Factor Matrix)�H�Ϊ��~�]�l�x�}(Item Factor Matrix)�C
- `Factorization Machine (FM)`:\
Factorization Machine�b�}�����(Sparse Data)�i��S�x��e(Feature Interaction)�é���X��b�]�l(Latent Factor)�A�i�b�u�ʮɶ������רӶi��V�m�A�B��K�W�ҤơC�۸���²���u�ʼҫ��h�Ҷq�F�椬�@�ζ��A�S��G���h�����j�k(Degree-2 Polynomial Regression)��[��ƪx��(Generalization)����O�C

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
�H Factorization Machine(FM) ����¦�A�N FM �Ҳ��ͪ��S�x�V�q�A��J�@�������g�������A�H Multi-Layer Perceptron(MLP) �N�����n�Ӷi��w�����ȡC

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28%5Cmathrm%7Bconcat%7D%28%5BW%5E%7B%28k%29%7D%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%20%5C%2C%5C%2C%20%5Cmathrm%7Bfor%7D%20%5C%2C%5C%2C%20k%3D1%2C2%2C...%2CK%5D%29%29">
  <br >
  <img src="model_figure/FNN.png" width="450">
  </p>

- `Product-based Neural Networks (IPNN, OPNN)`:\
��_ FNN �A�b MLP ����J�[�J�C�� field ���� inner/outer product ���S�x��e�C
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
�ھ�IPNN, OPNN�i�橵���A��_�[�J�C��field����inner/outer product���S�x��e�bMLP����J�APIN�Ҽ{�N���field�������S�x�̿�J���P���l�����ҫ��Ѩ��S�x�A�̫�b��JMLP�C
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
���ҫ��̥D�n���S��O�N���n(Convolution)�������ǤJCTR���w���ҫ��C
  - �Ϊk�@�G�Ҽ{�C�ӤH�b���P���ɶ��b�W�|�����P���S�x(Features)�A�z�L *1*-D Conv�f�t *p*-max ���Ƽh(*p*-max pooling)���^�����P�ɶ�(Temporal)����T�C *p*-max ���Ƽh�D�n�b�B�z�C�ӤH�b�PItem�����ʮɶ��B���ƨä��@�P�ӵL�k��¨ϥέ�l���̤j���Ƽh�i��C
  - �Ϊk�G�G�S�����P�ɶ����������Y�A��⤣�PFiled����T�֦b�@�_�A�令���PFiled��embedding vector�b�ۦP���פW�i����n�B��M��� *p*-max�C

  <p align="center">
  <img src="model_figure/CCPM.jpg" width="450">
  </p>


- `Wide & Deep (WD)`:\
WD�ҫ��̥D�n���Q���˨t�μҫ�����ӬD��:

  - Memorization(�O��): �ҫ��O�_�i�H�O����v��ƪ����ʲզX�i��w��?
  - Generalization(�x��): �ҫ��i�_��b�X�X�s���S�x�զX�W�[�w�����G���h�˩�?
  
  �bMemorization�W�i�H�z�LLogistic regression�Ӷi��(�U�ϥ��䪺Wide models)�C�Q��Logistic regression�K�i�H�ǲߦU���S�x���w�����G���������Y(����������J�S�x�|�ݭn�B�~�H�u���S�x�u�{ ex.cross-product transformation)�C�ӦbGeneralization�W�i�H�z�LDeep Network���^���󰪶����S�x�զX(�U�ϥk�䪺Deep models)�A�i�@�B���X�X�s���S�x�զX�W�[�w�����G���h�˩ʡCWD�ҫ��z�L���XWide models��Deep models�ӦP�ɦҼ{Memorization�H��Generalization(�U�Ϥ���������)�C

  <p align="center">
    <img src="model_figure/Wide&Deep.png" width="750">
  </p>

- `Neural Factorization Machine (NFM)`:\
NFM�ҫ��D�n�O���FM�ҫ��i���}�A�b��l��FM�ҫ����A����R���Ҽ{�S�x�������G���椬�@�ΡA���ȥH�u�ʪ��覡�[�b�ҫ����A�ä���L�k�Ҷq�S�x�������D�u�����Y�C�]��NFM�b�o�I�W�i��վ�A�åB�R��?�X�FFM�������G���u�ʯS�x�P���g�����ҫ����������D�u�ʯS�x�C

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
FM�ҫ�����Ҷq�S�x�������G���椬�@�ΡA���O�Ҧ��S�x���v�����O�ۦP���A�o�˪��B�z�覡�γ\�ä����]�����O�Ҧ����S�x���O���Ϊ��A���L�Ϊ��S�x�i��զX�S�[��w���ҫ������|�a�J���n(Noise)���z�Z�A���CFM���ĪG�A�]��AFM���o�˪��[�I�ޤJAttention������A���ҫ�����O�ۦ�վ�S�x�椬�@�Ϊ����n�{�סC

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
DeepFM�ҫ��i�H����WD�ҫ�����i���A�bWD�ҫ��ϥ�Logistic regression�ӾǲߦU���S�x���w�����G���������Y�A�Ӧb���˨t�η����A��Ʃ��|�O�����B�}�����A�YLogistic regression�ݭn�Ҽ{�G���椬�@�ζ��e���ɭP�V�m���G�����T�A�]���N�����Factorization machine(FM)��n��ѨM�o�˪����D�A�åB�p���@�Ӥ]����٥hWD�ҫ��bWide part�ݭn�B�~���S�x�u�{�A���i�άۦP��Embedding vector�ӧ@����J�C
  <p align="center">
    <img src="model_figure/DeepFM.png" width="450">
  </p>

- `Deep Crossing (DCN)`:\
�bWide & Deep�ҫ���{�F�P�ɦҼ{Memorization��Generalization�C���bWide�̤j�����D�N�O�ݭn�H�u���S�x�u�{�ӳ]�p�S�x�����e���CDeepFM���M�bWide�������HFM�i��A���]�ȭ��զX�S�x���G���椬���Y�C�]��Deep Crossing���֤߷����b�����ҫ��۰ʥh�ǲ߯S�x�����������椬���Y�C�b�U�Ϫ����b����¬O�@��Deep Network�D�n�B�zGeneralization�ϼҫ��b�w���W�Ҷq�զX���h�˩ʡA�Ӧb�k�䬰Cross Network�CCross Network�D�n�y�{�p�U:

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

