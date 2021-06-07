# Team information

Student 1 Matrikelnummer + Name:
Student 2 Matrikelnummer + Name:
Student 3 Matrikelnummer + Name:

# Report

## Part 1

### KNRM
    todo describe problems and solutions implementing KNRM

### CONV-KNRM
todo describe problems and solutions implementing CONV-KNRM

### TK
    todo describe problems and solutions implementing TK
    choosing the correct hyperparameters learning rate)
    knowing specific weight tricks --> weight initialization
    choosing correct time to stop

### FK
    todo describe FK model
Since the aim of the TK-Model is to provide an efficient and lightweight neural re-ranking model we considered the computational efficient Fourier-transformation layers of [FNET]((https://arxiv.org/pdf/2105.03824.pdf)) as legitimate extension to this approach. 
A common problem of self attention as used in transformers is its O(n^2) runtime and memory requirement.
Whereas other efficient transformer adaptations (such as Linformers or Performers)  deploy mathematical tricks to reduce the memory and time requirements of attention computation, FNET does not use any attention mechanism at all. Instead it "mixes" the tokens of a sequence by an unparameterized Fourier-Transformation and further processes the mixings with feedforward layers in order to learn to select the desired components of the mix.
According to the [FNET Paper](https://arxiv.org/pdf/2105.03824.pdf) replacing the self attention layers of a transformer with 
a standard Fourier Transformation can achieve but to 92% of BERT performance while running up to seven times faster on GPUs.
We therefore just replaced the stacked self-attention in the TK-Model with stacked Fourier-Transformation-Transformer blocks.
- aim of tk --> effiecent 
- problem of self attention --> run time
- solution of fnet --> linear mixing with unparemetric fouretransformations
- fnet obvious choose to replace contextualization
()


### Results

| Model | Test-Set | Batches |  Training Loss    |  Validation MRR@10 | Test MRR@10  | Comment |
|-------:|------:|----:|----:|----:|----:|----|
| KNRM | MSMARCO | 12323 | 0.123 | 0.23  | 0.23 | super nice run |


## Part 2

    todo describe used model for extractive qa; which reranker model is used; results on fira etc.
    - generate result files from our best reranking model
    - evaluate qa results and write report

## Part 3

    todo visualize tk model vs. fk model in neural ir-explorer and tell a story
    - get neural ir-explorer running
    - generate secondary results for tk and fk model
    - explore with the ir-explorer and search for interesting differences (e.g.: where fk model failed to watch for context)

