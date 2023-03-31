# DeepPoIs: Points of Interest based on Neural Network’s Key Recovery in Side-Channel Analysis through Occlusion

This work contains the repository of the paper <br>
**"DeepPoIs: Points of Interest based on Neural Network’s Key Recovery in Side-Channel Analysis through Occlusion" by Trevor Yap, Shivam Bhasin and Stjepan.**



### 1. Key Guessing Occlusion (KGO)
- In this repository, we apply Key Guessing Occluion (KGO) `Key_Guess_Occlusion.py` to obtain the DeepPoIs.
We have incoorporated 1-KGO as the last round of the KGO algorithm stated as Algorithm 1 in the paper. 
1-KGO allow us to see how much each DeepPoIs contribute in recovering the key via the GE value. 

  - The algorithm is defined as `KGO` in  `Key_Guess_Occlusion.py` 



### 2. Visualization of DeepPoIs
- We visualize DeepPoIs via CPA in `Key_Guess_Occlusion_cpa.py` and `Key_Guess_Occlusion_cpa_2.py`.
  - `Key_Guess_Occlusion_cpa_2.py` consist of the visualization the DeepPoIs of Chipwhisperer (CW) dataset.
  - `Key_Guess_Occlusion_cpa.py` consist of the visualization of ASCAD, ASCAD_variable and AES_HD.



### 3. Usage of Attribution-based methods
- We have also included the attribution methods of Saliency Map (aka Gradient), Layer-wise
Relevance Propagation (LRP) and 1-Occlusion. We use [INNvestigate](https://github.com/albermax/innvestigate) to apply Saliency Map and LRP
  - In order to use it, please fix the follow bug stated [here](https://github.com/albermax/innvestigate/issues/177#issuecomment-627918737). We also have to replace the Batch Normalization with a Dense layer to work another bug. 


### 4. Exploiting the DeepPoIs as a feature selection tool for Template Attack.
- In `Key_Guess_Occlusion_ta.py`, we compare the different classical feature selection techniques and attribution-based techniques used in side-channel analysis 
with the KGO.
  - Classical feature selection tools: SOSD,SOST, first-order CPA, multivariate second-order CPA.
  - Explainability methods: Saliency Map, LRP, 1-Occlusion.


### 5. Dataset.
The dataset we used are the following: 
  - [Chipwhisperer](https://github.com/newaetech/chipwhisperer-datasets) (CW): we have zipped it in the dataset file.
  - [ASCADv1](https://github.com/ANSSI-FR/ASCAD): both fixed key (synchronized, desync50, desync100) and random key. We denoted it as ASCADf and ASCADr respectively.
  - [AES_HD_ext](http://aisylabdatasets.ewi.tudelft.nl/): We denoted it as AES_HD in the paper.

