# Team information

Student 1: 01529038, Preinsperger Christopher:

Student 2: 1129115, Kaufmann Thomas:

Student 3: 11777780, Kowarsch Florian:

# Report

## Part 1

### Training and Evaluation
There are a multitude of measures that we have taken to ensure a sufficiently well trained model.
- We trained our models for multiple epochs
- We used a custom early stopping implementation to end the training process if the following criteria were met
- - The number of iterations must not exceed 100000
- - A minimum loss decrease of 0.001 has to achieved
- - Looking at the last 40 losses a minimum standard deviation of 0.01 has to be reached
- The model has been evaluated using a certain interval
- after every model evaluation the early stopping criteria were checked

For testing/evaluation we used the models to rank the documents in the test dataset
and than calculated the following performance metrics
- MRR@10
- nDCG@10
but used MRR@10 as our most important performance indicator.

To improve the training process we implemented a custom early stopping solution.
The consists of the so called early stopping watcher which has the following important features:
- It stops the training process if some criteria are (not) met
- It allows you to add custom criteria to be checked periodically
- It allows your to set the "patience" of the watcher defining how often certain criteria can (not) be met in a row before the watcher stops the training process

We recorded and visualized our training runs with [wandb](https://wandb.ai/).

### KNRM
Compared to the more sophisticated approaches described below, implementing KNRM was in general relatively straight forward. 
KNRM consists in general only of very few parameters, thus opens up rather limited degree of freedom for customizations. 
The main challenges were essentially to properly handle numerical issues due to the log-sum of each query word's feature vectors. 
We used pytorch's clap function to ensure a minimum value of 10^(-10), followed by proper masking of paddings to avoid introducing errors.
In addition, some manual tests with different hyper-parameters were conducted, however, parameters listed in the paper turned out to be (more or less) best. 

### CONV-KNRM
The convolutional knrm builds upon the previously described knrm uses n-grams of different length instead of the plain word embeddings.
During implementation I followed the described architecture of Dai et al. very strictly and only ran into problems regarding the usage
of pytorch. I used pytorch for the first time and played around with it quite a lot.
The biggest issues I had were related to wrong tensor dimension and exceeding the available memory.
To be able to further process the word embeddings using the convolutional layer I had to switch dimensions.
I also tried to limit the input for each layer to only one tensor, therefore using the tensor stack method extensively.
Unfortunately this caused some memory problems especially when trying to train the model on a GPU. Because of this I reverted the layer
inputs back to using multiple inputs.

### TK
The biggest challenge implementing model TK was to correctly apply every implementation detail and little tricks.
For instance, without running into troubles I would have never considered to pre-initialize weights of the learning-to-rank layer with small values.
Also differences that are not explicitly mentioned in the paper (like applying tanh on the cosine-similarity matrix), had quite an impact on the resulting performance and could only be observed by trial and error. 
Furthermore I refused to use AllenNLP's StackedSelfAttention Module and aimed to implement this part on my own. However it took quiet a lot attempts until my implementation was error free and achieved similar good results as using AllenNLP's version. So in practice I would not recommend to reimplement such things on your own, nevertheless it was a learningfull and interesting experience.
As always after bigger adaption hyperparameters like learning rate or weight decay have been reevaluated. This process is definitely more difficult and time consuming than expected.
All this mentioned difficulties intensify itself by the needed time to train the network. Every little tweak to test, every hypothesis to modify the network needs at least 45min of training time to be answered. I am aware that this time consumption is minimal compared to other way bigger models, however it is still more demanding than the development of a classical non NN re-ranker or a very small NN.


### FK
Since the aim of the TK-Model is to provide an efficient and lightweight neural re-ranking model we considered the computational efficient Fourier-transformation layers of [FNET]((https://arxiv.org/pdf/2105.03824.pdf)) as legitimate extension of this approach. 
A common problem of self attention as used in transformers is its O(n^2) runtime and memory requirement.
Whereas other efficient transformer adaptations (such as [Linformers](https://arxiv.org/pdf/2006.04768.pdf) or [Performers](https://arxiv.org/pdf/2009.14794.pdf))  deploy mathematical tricks to reduce the memory and time requirements of attention computation, FNET does not use any attention mechanism at all. Instead it "mixes" the tokens of a sequence by an unparameterized Fourier-Transformation and further processes these mixings with feedforward layers in order to learn to select the desired components of the mix.
According to the [FNET Paper](https://arxiv.org/pdf/2105.03824.pdf) replacing the self attention layers of a transformer with 
a standard Fourier Transformation can achieve up to 92% of BERT performance while running up to seven times faster on GPUs.
We therefore just replaced the stacked self-attention in the TK-Model with stacked Fourier-Transformation-Transformer blocks. 
With this setup we achieved a MRR@10 on MSMARCO test set of 0.22. It uses the same amount of FNET Layers as self-attention layers used in the TK Model.
Since our TK Model achieves MRR@10 of 0.24 on the same test set we meet the expectations of the FNET authors by reaching ~91-92% of performance compared to using self-attention while saving about 23% of GPU memory requirement.
Despite this interesting results one has to question if more efficient models than Model TK are actual needed.


### Results

The implemented re-ranking models were trained with the MSMARCO training set consisting of 2.4e^6 samples and evaluated against the coarse grained MSMARCO and the more fine-grained FIRA labelsets.
Training was conducted with batch sizes of 128 and hyper-parameters were chosen depending on the particular model at hand. 
The following hyper-parameters were chosen for the respective models.

| Model | Learning Rate | Weight Decay |
| -----:|-------:|-------:|
| KNRM | 0.001 | 0.01 | 
| Conv-KNRM | 0.001 | 1e-15 |
| TK | 0.0001 | 1e-16 | 
| FK | 0.0001 | 0.0001 |

The following table compares training loss and MRR@10 on validation and test sets for the implemented models.
In general, it can be observed that model complexity and thus the expressiveness goes in accordance with its quality in terms of the MRR@10 statistic.
Even more, some decay can be observed between validation and test set, which is in this degree not unusual. 


| Model | Test-Set | Training Batch Size |  Training Loss    |  Validation MRR@10 | Test MRR@10  |
|-------:|------:|----:|----:|----:|----:|
| KNRM      | MSMARCO | 128 | 0.5313  | 0.189  | 0.193 | 
| Conv-KNRM | MSMARCO | 128 | 0.1034  | 0.213  | 0.207 |
| TK        | MSMARCO | 128 | 0.8973  | 0.232  | 0.231 |
| FK        | MSMARCO | 128 | 0.9553  | 0.230  | 0.220 | 
|------------------|------------------|------------------|------------------|------------------|------------------|
| KNRM      | FIRA  | 128 | - | 0.619 | 0.608 | 
| Conv-KNRM | FIRA  | 128 | - | 0.613 | 0.607 |
| TK        | FIRA  | 128 | - | 0.659 | 0.647 |
| FK        | FIRA  | 128 | - | 0.667 | 0.664 | 



## Part 2
The second part of assignment was about transformers and an extractive qa task. 
We started with the pure transformer model without re-ranking and decided on the distilled BERT based model "distilbert-base-uncased-distilled-squad". 
We used configurations for sequence and question lengths in accordance to part 1 and implemented some helper classes to invoke the provided F1 and exact metrics.   
Even though we decided on a distilled model, processing the entire test set took quite a while. 
We set the device to a GPU in the pipeline constructor appropriately and investigated the internals of the model/pipeline to find ways to gain speedup. 
First, we tried to pass batches of sequence question tuples and even set size of the threadpool > 1 (which made things even worse) without observing a significant speedup. 
In the end, to process the entire test set of approx. 53000 entries took about 24h. 

As our novel FK model showed its slight advantages on the FIRA labels, it was selected for our re-ranking phase in this part of the exercise.
We took a snapshot of the best model obtained from the last section's evaluation phase and implemented some snippets to transform the MSMARCO test set into an input file compatible with the fira.qrels.qa-tuples.tsv data set format.
During the mapping of document to reference answer, we observed that our re-ranking procedure may not be optimal, since for quite a few documents no reference answers could be obtained. 
As suggested in the TUWEL discussion board, we counted these cases with a score of zero, leading to quite a performance shift towards this direction.   

The following table briefly compares average F1 and exact scores as well as their standard deviations for both QA models. 
There are several important remarks to be made: 
1) the sizes of the testsets differs significantly due to the recommended procedure to compile the testset. 
2) due to the re-ranking it is quite frequent that the select top1 document does not have a reference answer. As mentioned, in this case we score it with 0.0. 

| Model | Test-Set | Testset Size | F1 | Exact | No Reference |
|-------------:|----:|----:|------:|------:|----:|
| Distilbert QA         | FIRA qa-tuples                    | 52606 | 0.371 (+- 0.32)   | 0.108 (+- 0.311)  | -     |
| FK + Distilbert QA    | MSMARCO testset + FIRA answers    | 1992  | 0.150 (+- 0.28)   | 0.055 (+- 0.228)  | 1240  |    

In general, it can be observed that the Distilbert QA model actually behaves quite well (although we do not have any baseline here). 
For the model in the context of a "simulated" re-ranking pipeline with re-ranking, on the other, a significant decrease in quality can be observed, obviously due to the relative high number of more than 60% where the reference was missing and a score of 0.0 was enforced.
When not considering these cases, apparently the F1 and exact score increase to a level similar to the pure transformer based model.    

## Part 3

For Part 3 we were interested in analyzing the differences between TK and FK model. How does the Fourier-Transformation token mixing affect the output and similarities scores compared to self attention token mixing. Are there spotable differences at all? Can the differences in performance be identified as the ability to capture the whole context of a token set? If yes can this be explained by the differences in the similarity matrix?
### Setup
In order to compare and visualize divergent predictions of both models we reused the [Neural IR-Explorer](https://github.com/sebastian-hofstaetter/neural-ir-explorer). In the following list we describe the steps we have taken to execute this plan:

* analyzed code of neural-ir-explorer and the reuse possibility 
* requested additional files needed by neural-ir-explorer from Mr. Hofts??tter (thank you!)
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

![_](./documents/viz_report_heart_query_overall.png)

In the following figure we can clearly see that FK does not associates *cardiac* with obvious choose *heart*.

![_](./documents/viz_report_heart_query_overall_cardiac.png)

Further examples indicate that FK lags in finding contextual similarities in the document. For instance, for the word *being* TK correctly finds similarities to *body*, *is* and *heart*, while FK only associates it with *cardiac*.

![_](./documents/viz_report_heart_query_overall_being.png)

But there are also examples for which FK clearly catches the context of the query better than TK. In the following figure we see that FK associates *pacemaker* with *implanted*, *impulses* and *pacemaker* while TK only indicates *pacemaker* in the document.

![_](./documents/viz_report_heart_query_overall_pacemaker.png)

Another good comparison is served by a document form the query "*how long does it take for the earth to make one revolution?*". For instance, as shown in the following figure, model FK seems to over weight similarities with *how*:

![_](./documents/viz_report_revolution_query_how.png)

In the same query-doc pair we can see an example how FK seems to capture certain relationships, but weights them less strongly that TK. TK shows similarity from *long* in the query to *long*, *hour* (*day* and *cycles*) in the document while FK actual also captures this relationships but indicates them with much lower similarity. This is especially interesting since the FK actual has a lower alpha value of 0.73 compared to TK 0.82 indicating that model FK learned to integrate the contextualization 
![_](./documents/viz_report_revolution_query_long.png)

### Conclusion

In conclusion it is interesting that token mixing in FNET style works at all. We are glad that our reproduction of this concept resulted in the same performance expectation as stated by the authors of FNET. Although it does not reach the performance of model TK, which uses self attention, it clearly shows that the Fourier-Transformation token mixing is beneficial for context-aware reranking, since it outperforms the non-context-aware approach KNRM and performs similar to the local context approach CONV-KNRM.
The are many aspects that we could have done differently in the process of analyzing and visualizing the comparison of both models.
In retro perspective choosing the greatest score delta to extract interesting query-doc pairs may not be the perfect choice, since it is normal that the score ranges can differ from model to model. For re-ranking only the relative scores between query-doc pairs are relevant not the actual score value. Maybe it would have been more appropriate to select the queries with greatest rank differences between the two model for comparison. However, as seen above, the score delta yielded good examples in which both models greatly differ and allowed to interpret this differences meaningful.
Another point is that this analysis is only done qualitatively. To provide more assured statements it could be quantality computed if the rank difference is greater for query-doc pairs that rely more on contextualized understanding (e.g. where it is less likely that exact words of the query occur in relevant documents).


For further work it would be interesting to compare the FK model performance with TK model under the use of more training data and with varying number of FNET-layers.

