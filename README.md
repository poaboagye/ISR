# Interpretable Debiasing of Vectorized Language Representations with Iterative Orthogonalization

Official Repository for the implemention of **Iterative Subspace Rectification (ISR)** from our ICLR 2023 [paper](https://openreview.net/pdf?id=TkQ1sxd9P4):

[**Interpretable Debiasing of Vectorized Language Representations with Iterative Orthogonalization**](https://openreview.net/pdf?id=TkQ1sxd9P4). Prince Osei Aboagye, Yan Zheng, Jack Shunn, Chin-Chia Michael Yeh, Junpeng Wang, Zhongfang Zhuang, Huiyuan Chen, Liang Wang, Wei Zhang, Jeff Phillips.


We propose a new mechanism to augment a word vector embedding representation that offers improved bias removal while retaining the key information—resulting in improved interpretability of the representation. Rather than removing the information associated with a concept that may induce bias, our proposed method identifies two concept subspaces and makes them orthogonal. The resulting representation has these two concepts uncorrelated. Moreover, because they are orthogonal, one can simply apply a rotation on the basis of the representation so that the resulting subspace corresponds with coordinates. This explicit encoding of concepts to coordinates works because they have been made fully orthogonal, which previous approaches do not achieve. Furthermore, we show that this can be extended to multiple subspaces. As a result, one can choose a subset of concepts to be represented transparently and explicitly while the others are retained in the mixed but extremely expressive format of the representation.

## Dependencies

* Python 
* NumPy 
* [GluonNLP](https://nlp.gluon.ai/install/install-more.html)

## How To Apply Iterative Subspace Rectification (ISR) on Two Concept Subspaces

Given a biased embedding (static or contextual embedding), you can apply ISR to debias a pair of concepts with the following command:

### Bespoke Word Lists (Appendix F)

* Use the Bespoke Word Lists in the ```Bespoke_Word_Lists``` directory to Reproduce the ISR results in Table 3 of the Paper. To debias Gender (M/F)  and Pleasant/Unpleasant subspace, you should run the following:

```py

python ISR_TwoConcepts.py --iterations 10 --embedType "glove" --preTrainFile "glove.6B.300d" --top_vocab 200000 \
--X_file "Bespoke_Word_Lists/gen_male.json" --Y_file "Bespoke_Word_Lists/gen_female.json" \
--A_file "Bespoke_Word_Lists/career.json" --B_file "Bespoke_Word_Lists/family.json" \
--output_file  debiased_emb.vec

```

