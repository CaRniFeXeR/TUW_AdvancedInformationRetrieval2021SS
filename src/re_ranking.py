from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader
prepare_environment(Params({})) # sets the seeds to be fixed

import torch

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from data_loading import *
from model_knrm import *
from model_conv_knrm import *
from model_tk import *

# change paths to your data directory
config = {
    "vocab_directory": "data/allen_vocab_lower_10",
    "pre_trained_embedding": "data/glove.42B.300d.txt",
    "model": "tk",
    "train_data": "data/triples.train.tsv",
    "validation_data": "data/tuples.validation.tsv",
    "test_data":"data/tuples.test.tsv",
    "onGPU" : False,
    "train_batch_size": 32
}

#
# data loading
#

vocab = Vocabulary.from_files(config["vocab_directory"])
tokens_embedder = Embedding(vocab=vocab,
                           pretrained_file= config["pre_trained_embedding"],
                           embedding_dim=300,
                           trainable=True,
                           padding_index=0)
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# recommended default params for the models (but you may change them if you want)
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11)
elif config["model"] == "conv_knrm":
    model = Conv_KNRM(word_embedder, n_grams=3, n_kernels=11, conv_out_dim=128)
elif config["model"] == "tk":
    model = TK(word_embedder, n_kernels=11, n_layers = 2, n_tf_dim = 300, n_tf_heads = 10)


# todo optimizer, loss 

print('Model',config["model"],'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)

#
# train
#

_triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_triple_reader = _triple_reader.read(config["train_data"])
_triple_reader.index_with(vocab)
loader = PyTorchDataLoader(_triple_reader, batch_size=config["train_batch_size"])

#activate training mode on model
model.train(mode = True)

#loss = max(0, s_nonrel - s_rel + 1) .... called marginrankingloss
marginRankingLoss = torch.nn.MarginRankingLoss(margin=1, reduction='mean') #.cuda(cuda_device)
#since we always want a "big" distance between unrelevant and releveant documents we can use Ones for each pair as targetValue
targetValues = torch.ones(config["train_batch_size"]) #.cuda(cuda_device)

#todo set learningrate and weight decay
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(2):

    trainLossList = []

    for batch in Tqdm.tqdm(loader):
        #todo include GPU support
        
        #first forward pass query + relevant doc
        output_score_relevant = model(batch["query_tokens"]["tokens"], batch["doc_pos_tokens"]["tokens"])
        #second forward pass query + non-relevant doc
        output_score_unrelevant = model(batch["query_tokens"]["tokens"], batch["doc_neg_tokens"]["tokens"])

        current_batch_size = batch["query_tokens"]["tokens"]["tokens"].shape[0]
        if current_batch_size != config["traning_batch_size"]:
            targetValues = torch.ones(current_batch_size) #.cuda(cuda_device)
        
        batch_loss = marginRankingLoss(output_score_relevant, output_score_unrelevant, targetValues)
        batch_loss.backward()
        optimizer.step()
        current_loss = batch_loss.detach().numpy()
        trainLossList.append(current_loss)
        print(f"current loss: {current_loss:.3f}")

    meanLoss = np.mean(trainLossArray)
    stdLoss = np.std(trainLossArray)
    print('epoch {}\n train loss: {:.3f} ± {:.3f}'.format(epoch +1, meanLoss, stdLoss))


#
# eval (duplicate for validation inside train loop - but rename "loader", since
# otherwise it will overwrite the original train iterator, which is instantiated outside the loop)
#

_tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_tuple_reader = _tuple_reader.read(config["test_data"])
_tuple_reader.index_with(vocab)
eval_loader = PyTorchDataLoader(_tuple_reader, batch_size=128)

#set model in eval mode
model.train(mode = False)

for batch in Tqdm.tqdm(eval_loader):
    # todo test loop 

    # todo evaluation
    pass
