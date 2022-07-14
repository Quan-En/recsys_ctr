# DLG assignment3: Model Design & Comparison for Recommendation Models
<!--https://latex.codecogs.com/eqneditor/editor.php-->
���@�~�N�յۮM�Φh�ؤ��P��k�b���P��ƶ��W�C�ϥΪ���k�D�n�����������O���Ĥ@���g��k�βĤG���H���g�������D��������k�C
�b�ĤG������k�H�U��覡����
```r
python3 deepctr_main.py -dataname -modelname 
```

## Dataset Intro
��ƶ��@�T�ؤ��O��: `movielens`, `yelp` �� `douban_book`

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
- `Collaborative Filtering (CF)`:��P�L�o���D�n���� User-based �P Item-based ��ءA�ӵ����ۦ��{�ת���k���`������ Cosine Similarity �P Pearson Correlation Coefficient ��ءA�]���Ҽ{���P�ت���P�L�o�P���P���ۦ��{�׵����ǫh�@��4�ؤ�k�C���O��: User-based & Cosine Similarity(UCF-s)�BItem-based & Cosine Similarity(ICF-s)�BUser-based & Pearson Correlation(UCF-p)�BItem-based & Cosine Similarity(ICF-p)�C
- `Matrix Factorization (MF)`:�x�}���Ѫ����˨t�Ϊ��֤߷����{���Τῳ��D�n�Q�ּƪ��]���Ҽv�T�H�ΰӫ~�Q��ܻP�_�]�O����ּƪ��]���v�T�C�]���N�����x�}(Rating Matrix)��ѡA��g��C���ת��x�}�����]�l�Ŷ�(Latent Factor Space)�C�D�n�B�Ω_���Ȥ��Ѫk(Singular Value Decomposition, SVD)�i��x�}���ѡA�N�쥻�������x�}��Ѧ��ϥΪ̦]�l�x�}(User Factor Matrix)�H�Ϊ��~�]�l�x�}(Item Factor Matrix)�C
- `Factorization Machine (FM)`:Factorization Machine�b�}�����(Sparse Data)�i��S�x��e(Feature Interaction)�é���X��b�]�l(Latent Factor)�A�i�b�u�ʮɶ������רӶi��V�m�A�B��K�W�ҤơC�۸���²���u�ʼҫ��h�Ҷq�F�椬�@�ζ��A�S��G���h�����j�k(Degree-2 Polynomial Regression)��[��ƪx��(Generalization)����O�C

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
�HFactorization Machine����¦�A�NFM�Ҳ��ͪ��S�x�V�q�A��J�@�������g�������A�HMLP(Multi Layers Perceptron)�N�����n�Ӷi��w�����ȡC

  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y%28%5Cmathrm%7Bx%7D%29%3D%5Cmathrm%7BMLP%7D%28%5Cmathrm%7Bconcat%7D%28%5BW%5E%7B%28k%29%7D%5Cmathrm%7Bx%7D%5B%5Cmathrm%7Bstart%7D_k%3A%5Cmathrm%7Bend%7D_k%5D%20%5C%2C%5C%2C%20%5Cmathrm%7Bfor%7D%20%5C%2C%5C%2C%20k%3D1%2C2%2C...%2CK%5D%29%29">
  <br >
  <img src="model_figure/FNN.png" width="450">
  </p>

- `Product-based Neural Networks (IPNN, OPNN)`:
��_FNN�A�bMLP����J�[�J�C��field����inner/outer product���S�x��e�C
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
�ھ�IPNN, OPNN�i�橵���A��_�[�J�C��field����inner/outer product���S�x��e�bMLP����J�APIN�Ҽ{�N���field�������S�x�̿�J���P���l�����ҫ��Ѩ��S�x�A�̫�b��JMLP�C
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

