# Team information

Student 1: 01529038, Preinsperger Christopher:

Student 2: 1129115, Kaufmann Thomas:

Student 3: 11777780, Kowarsch Florian:


* Task 2 Etractive QA im Report beschreiben --> Thomas
* Task 1 Testssetscores berechnen --> Thomas    ==> DONE
* Task 1 Conv-Model findings --> Christopher
* Task 1 TK-Model findings --> Florian      ==> DONE
* Task 1 KNRM findings --> Thomas           ==> DONE
* Task 1: lookup training losses in wandb for the table
* Task 1: one line about the hyper params for the different models
# Report

## Part 1

### Training and Evaluation
    todo explain overall training and evaluation process, findings etc.
    todo explain how we implemented early stopping
    
We recorded and visualized our training runs with wandb.

### KNRM
Compared to the more sophisticated approaches described below, implementing KNRM was in general relatively straight forward. 
KNRM consists in general only of very few parameters, thus opens up rather limited degree of freedom for customizations. 
The main challenges were essentially to properly handle numerical issues due to the log-sum of each query word's feature vectors. 
We used pytorch's clap function to ensure a minimum value of 10^(-10), followed by proper masking of paddings to avoid introducing errors.
In addition, some manual tests with different hyper-parameters were conducted, however, parameters listed in the paper turned out to be (more or less) best. 

### CONV-KNRM
todo describe problems and solutions implementing CONV-KNRM

### TK
The biggest challenge implementing model TK was to correctly apply every implementation detail and little tricks.
For instance, without running into troubles I would have never considered to pre-initialize weights of the learning-to-rank layer with small values.
Also differences that are not explicitly mentioned in the paper (like applying tanh on the cosine-similarity matrix), had quite an impact on the resulting performance and could only be observed by trial and error. 
As always after bigger adaption hyperparameters like learning rate or weight decay have be reevaluated. This process is definitely more difficult and time consuming than expected.
All this mentioned difficulties intensify itself by the needed time to train the network. Every little tweak to test, every hypothesis to modify the network needs at least 45min of training time to be answered. I am aware that this time consumption is minimal compared to other way bigger models, however it is still more demanding than the development of a classical non NN re-ranker or a very small NN.


### FK
Since the aim of the TK-Model is to provide an efficient and lightweight neural re-ranking model we considered the computational efficient Fourier-transformation layers of [FNET]((https://arxiv.org/pdf/2105.03824.pdf)) as legitimate extension to this approach. 
A common problem of self attention as used in transformers is its O(n^2) runtime and memory requirement.
Whereas other efficient transformer adaptations (such as Linformers or Performers)  deploy mathematical tricks to reduce the memory and time requirements of attention computation, FNET does not use any attention mechanism at all. Instead it "mixes" the tokens of a sequence by an unparameterized Fourier-Transformation and further processes the mixings with feedforward layers in order to learn to select the desired components of the mix.
According to the [FNET Paper](https://arxiv.org/pdf/2105.03824.pdf) replacing the self attention layers of a transformer with 
a standard Fourier Transformation can achieve but to 92% of BERT performance while running up to seven times faster on GPUs.
We therefore just replaced the stacked self-attention in the TK-Model with stacked Fourier-Transformation-Transformer blocks. 
With this setup we achieved a MRR@10 on MSMARCO test set of 0.22. It uses the same amount of FNET Layers as self-attention layers used in the TK Model.
Since our TK Model achieves MRR@10 of 0.24 on the same test set we meet the expectations of the FNET authors by reaching ~91-92% of performance compared to using self-attention while saving about 23% of GPU memory requirement.
Despite this interesting results one has to question if more efficient models than Model TK are actual needed.


### Results

The implemented re-ranking models were trained with the MSMARCO training set consisting of 2.4e^6 samples and evaluated against the coarse grained MSMARCO and the more fine-grained FIRA labelsets.
Training was conducted with batch sizes of 128 and hyper-parameters were chosen depending on the particular model at hand. 
For KNRM, a learning rate of 0.001 was selected in accordance to the paper and weight decay was set to 0.01. 
For ... < fill me in> ...

The following table compares training loss and MRR@10 on validation and test sets for the implemented models. 

In general, it can be observed that model complexity and thus the expressiveness goes in accordance with its quality in terms of the MRR@10 statistic.
todo add some interpretation...


| Model | Test-Set | Training Batch Size |  Training Loss    |  Validation MRR@10 | Test MRR@10  |
|-------:|------:|----:|----:|----:|----:|
| KNRM      | MSMARCO | 128 | 0.5313  | 0.189  | 0.193 | 
| Conv-KNRM | MSMARCO | 128 | -       | 0.213  | 0.207 |
| TK        | MSMARCO | 128 | -       | 0.232  | 0.231 |
| FK        | MSMARCO | 128 | -       | 0.230  | 0.220 | 
|------------------|------------------|------------------|------------------|------------------|------------------|
| KNRM      | FIRA  | 128 | - | 0.619 | 0.608 | 
| Conv-KNRM | FIRA  | 128 | - | 0.613 | 0.607 |
| TK        | FIRA  | 128 | - | 0.659 | 0.647 |
| FK        | FIRA  | 128 | - | 0.667 | 0.664 | 



## Part 2

    todo describe used model for extractive qa; which reranker model is used; results on fira etc.
    - generate result files from our best reranking model
    - evaluate qa results and write report
    - describe findings while implementing part 2

## Part 3

For Part 3 we were interested in analyzing the differences between TK and FK model. How does the Fourier-Transformation token mixing affect the output and similarities scores compared to self attention token mixing. Are there spotable differences at all? Can the differences in performance be identified as the ability to capture the whole context of a token set? If yes can this be explained by the differences in the similarity matrix?
### Setup
In order to compare and visualize divergent predictions of both models we reused the [Neural IR-Explorer](https://github.com/sebastian-hofstaetter/neural-ir-explorer). In the following list we describe the steps we have taken to execute this plan:

* analyzed code of neural-ir-explorer and the reuse possibility 
* requested additional files needed by neural-ir-explorer from Mr. Hoftsätter (thank you!)
* implemented secondary data logging for the re-ranking models
* installed, configured, built & run neural-ir-explorer
* implemented python script to select query-doc pairs with the greatest difference in score of both models
* adapted neural-ir-explorer to view only the queries in the clusters that are actual present in the secondary output
* adapted neural-ir-explorer to enable side-by-side document comparison between different runs

Further our semi-automated evaluation process looks as followed:
1. generate secondary output for both models
2. filter output for divergent results of both models (max delta between scores) & save them as filtered secondary output
3. visualize filtered secondary output of both models in the Neural IR-Explorer
4. manually select interesting query-document pairs and compare results of both models in side-by-side view
5. qualitatively interpret observations

All modifications we made to the Neural-IR-Explorer can be viewed on GitHub in this [Fork](https://github.com/CaRniFeXeR/neural-ir-explorer).
### Visualizations
The query *"___ is the ability of cardiac pacemaker cells to spontaneously initiate an electrical impulse without being stimulated."* contains the query-document pairs with the greatest difference in score between both models. Therefore we started our analysis with this query.

The following figure shows the result of TK Model on the left side and the result of FK Model on the right side.
You can see that FK lags to identify some significant words related to the current query (e.g. *heart*), while TK mostly identified important words. It is very interesting that TK associates the first occurrence of *heart* with *being*, while the second occurrence, which is located next to *cardiac* is related to *cardiac*.

![_](documents\report_heart_query_overall.png)

In the following figure we can clearly see that FK does not associates *cardiac* with obvious choose *heart*.

![_](documents\report_heart_query_overall_cardiac.png)

Further examples indicate that FK lags in finding contextual similarities in the document. For instance, for the word *being* TK correctly finds similarities to *body*, *is* and *heart*, while FK only associates it with *cardiac*.

![_](documents\report_heart_query_overall_being.png)

But there are also examples for which FK clearly catches the context of the query better than TK. In the following figure we see that FK associates *pacemaker* with *implanted*, *impulses* and *pacemaker* while TK only indicates *pacemaker* in the document.

![_](documents\report_heart_query_overall_pacemaker.png)

Another good comparison is served by a document form the query "*how long does it take for the earth to make one revolution?*". For instance, as shown in the following figure, model FK seems to over weight similarities with *how*:

![_](documents\report_revolution_query_how.png)

In the same query-doc pair we can see an example how FK seems to capture certain relationships, but weights them less strongly that TK. TK shows similarity from *long* in the query to *long*, *hour* (*day* and *cycles*) in the document while FK actual also captures this relationships but indicates them with much lower similarity.

![_](documents\report_revolution_query_long.png)

### Conclusion

In conclusion it is interesting that token mixing in FNET style works at all. We are glad that our reproduction of this concept resulted in the same performance expectation as stated by the authors of FNET. Although it does not reach the performance of model TK, which uses self attention, it clearly shows that the Fourier-Transformation token mixing is beneficial for context-aware reranking, since it outperforms the non-context-aware approach KNRM and performs similar to the local context approach CONV-KNRM.
The are many aspects that we could have done differently in the process of analyzing and visualizing the comparison of both models.
In retro perspective choosing the greatest score delta to extract interesting query-doc pairs may not be the perfect choice, since it is normal that the score ranges can differ from model to model. For re-ranking only the relative scores between query-doc pairs are relevant not the actual score value. Maybe it would have been more appropriate to select the queries with greatest rank differences between the two model for comparison. However, as seen above, the score delta yielded good examples in which both models greatly differ and allowed to interpret this differences meaningful.
Another point is that this analysis is only done qualitatively. To provide more assured statements it could be quantality computed if the rank difference is greater for query-doc pairs that rely more on contextualized understanding (e.g. where it is less likely that exact words of the query occur in relevant documents).


For further work it would be interesting to compare the FK model performance with TK model under the use of more training data and with varying number of FNET-layers.

