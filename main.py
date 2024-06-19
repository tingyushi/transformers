import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
import torch.nn as nn
import random
import numpy as np
import gc
from itertools import product
import time

import utilities
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Token_Embedding, TransformerLayers, Classifier, LMGenerator
from transformer import LearnableAbsolutePositionalEmbedding

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

    

# evaluate classifier (both loss and accuracy)
@torch.no_grad()
def compute_classifier_loss_accuracy(model, dataloader, loss_fn):
    model.eval()

    loss = 0
    correct = 0
    for xb, yb in dataloader:
        # move to device
        xb, yb = xb.to(device), yb.to(device)
        
        # forward 
        pred_logits = model(xb)
        loss += loss_fn(pred_logits, yb).item()
        correct += (pred_logits.argmax(1) == yb).type(torch.float).sum().item()

    loss /= len(dataloader)
    acc = correct / len(dataloader.dataset)

    model.train()

    return loss, acc


@torch.no_grad()
def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def stop_running():
    import sys ; sys.exit()


# training encoder classifier
def train_cls(classifier, train_CLS_loader, test_CLS_loader, n_epochs, verbose, learning_rate):
  
    # for the classification  task, you will train for a fixed number of epochs like this:
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(),lr=learning_rate)
    for epoch in range(n_epochs):

        print(f"\n========== Training Epoch {epoch + 1} ==========")
        
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)

            # forward
            pred_logits = classifier(xb)

            # calculate loss
            loss = loss_fn(pred_logits, yb)

            # back prop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        print(f"========== Finished Epoch {epoch + 1} ==========")
        
        if verbose:
            
            # eval on the training dataset
            train_loss, train_acc = compute_classifier_loss_accuracy(model=classifier, 
                                                                     dataloader=train_CLS_loader, 
                                                                     loss_fn=loss_fn)

            # eval on the testing dataset
            test_loss, test_acc = compute_classifier_loss_accuracy(model=classifier, 
                                                                   dataloader=test_CLS_loader, 
                                                                   loss_fn=loss_fn)

            print(f"========== Evaluation after Epoch {epoch + 1} ==========")
            print(f"Training Loss = { train_loss:.4f}")
            print(f"Training Accuracy = { 100 * train_acc:.2f} %")
            print(f"Testing Loss = { test_loss:.4f}")
            print(f"Testing Accuracy = { 100 * test_acc:.2f} %\n")
            
        else:
            
            if epoch == n_epochs - 1:
                
                # eval on the training dataset
                train_loss, train_acc = compute_classifier_loss_accuracy(model=classifier, 
                                                                         dataloader=train_CLS_loader, 
                                                                         loss_fn=loss_fn)

                # eval on the testing dataset
                test_loss, test_acc = compute_classifier_loss_accuracy(model=classifier, 
                                                                       dataloader=test_CLS_loader, 
                                                                       loss_fn=loss_fn)

                print(f"========== Evaluation after Epoch {epoch + 1} ==========")
                print(f"Training Loss = { train_loss:.4f}")
                print(f"Training Accuracy = { 100 * train_acc:.2f} %")
                print(f"Testing Loss = { test_loss:.4f}")
                print(f"Testing Accuracy = { 100 * test_acc:.2f} %\n")



# training decoder generator
def train_generator(generator, train_dataloader, test_wbush_dataloader, 
                    test_hbush_dataloader, test_obama_dataloader, learning_rate):
    
    # define loss function
    optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
    
    for i, (xb, yb) in enumerate(train_dataloader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        # forward
        loss = generator(xb, yb)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % eval_interval == 0:
            print(f"\n========== Evaluating after step {i + 1} ==========")

            train_perp = compute_perplexity(decoderLMmodel=generator, 
                                            data_loader=train_dataloader, 
                                            eval_iters=eval_iters)
            print(f"Training Set perplexity = {train_perp:.4f}")

            obama_perp = compute_perplexity(decoderLMmodel=generator, 
                                            data_loader=test_obama_dataloader, 
                                            eval_iters=eval_iters)
            print(f"Obama Testing Set perplexity = {obama_perp:.4f}")

            hbush_perp = compute_perplexity(decoderLMmodel=generator, 
                                            data_loader=test_hbush_dataloader, 
                                            eval_iters=eval_iters)
            print(f"HBush Testing Set perplexity = {hbush_perp:.4f}")

            wbush_perp = compute_perplexity(decoderLMmodel=generator, 
                                            data_loader=test_wbush_dataloader, 
                                            eval_iters=eval_iters)
            print(f"WBush Testing Set perplexity = {wbush_perp:.4f}")

            print(f"========== Finished step {i + 1} Evaluation ==========\n")
        


# do sanity check, generate attention maps
def sanity_check(tokenizer, model, sentence: str, folder_path, showImage = True):

    # generate sentence code
    sentence_code = tokenizer.encode(sentence)
    
    print(f"\nSentence used for sanity check: {sentence}\n")
    
    print(f"\nCheck out each token in the sentence\n")
    for idx, singleCode in enumerate(sentence_code):
        print(f"{idx} --- {tokenizer.decode([singleCode])}")
 
    santity_checker = utilities.Utilities(tokenizer=tokenizer, 
                                          model=model)
    
    santity_checker.sanity_check(sentence=sentence, 
                                 block_size=block_size, 
                                 folder_path=folder_path, 
                                 showImage=showImage)


# hyperparameter tuning for decoder generator
def generator_hptuning(generator, train_dataloader, test_wbush_dataloader, 
                       test_hbush_dataloader, test_obama_dataloader, learning_rate, include_scheduler):
    
    # define loss function
    optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
    
    if include_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for i, (xb, yb) in enumerate(train_dataloader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        # forward
        loss = generator(xb, yb)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if include_scheduler:
            scheduler.step()
        

        if (i + 1) % eval_interval == 0:
            print(f"Finished Training Step {i+1}")

        if (i + 1) == max_iters:
            print(f"=============== Final Evaluation ===============")

            train_perp = compute_perplexity(decoderLMmodel=generator, 
                                            data_loader=train_dataloader, 
                                            eval_iters=eval_iters)
            print(f"Training Set perplexity = {train_perp:.4f}")

            obama_perp = compute_perplexity(decoderLMmodel=generator, 
                                            data_loader=test_obama_dataloader, 
                                            eval_iters=eval_iters)
            print(f"Obama Testing Set perplexity = {obama_perp:.4f}")

            hbush_perp = compute_perplexity(decoderLMmodel=generator, 
                                            data_loader=test_hbush_dataloader, 
                                            eval_iters=eval_iters)
            print(f"HBush Testing Set perplexity = {hbush_perp:.4f}")

            wbush_perp = compute_perplexity(decoderLMmodel=generator, 
                                            data_loader=test_wbush_dataloader, 
                                            eval_iters=eval_iters)
            print(f"WBush Testing Set perplexity = {wbush_perp:.4f}")
    
    return train_perp, obama_perp, hbush_perp, wbush_perp



# evaluate the generator with the optimal hyperparameters
def optimal_generator_hptuning(generator, train_dataloader, test_dataloader, learning_rate, include_scheduler):
    
    # define loss function
    optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
    
    if include_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for i, (xb, yb) in enumerate(train_dataloader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        # forward
        loss = generator(xb, yb)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if include_scheduler:
            scheduler.step()
        
        if (i + 1) % eval_interval == 0:
            print(f"Finished Training Step {i+1}")
        
    perp = compute_perplexity(decoderLMmodel=generator, 
                              data_loader=test_dataloader, 
                              eval_iters=eval_iters)

    return perp


def main():

    torch.manual_seed(seed)

    # process argument
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--part1', action='store_true', help='Run Part1 of the Assignment')
    parser.add_argument('--part2', action='store_true', help='Run Part2 of the Assignment')
    parser.add_argument('--sparseAttention', action='store_true', help='Run Part3(Architecture Exploration) of the Assignment')
    parser.add_argument('--hptuning', action='store_true', help='Fine-tuning hyper-parameters')
    parser.add_argument('--optimalhp', action='store_true', help='Run Optimal Hyper-parameter combination')

    args = parser.parse_args()

    provided_args = sum([args.part1, args.part2, args.sparseAttention, args.hptuning, args.optimalhp])
    if provided_args == 0:
        parser.error('You must provide Command Line Arguments, please check README')
        stop_running()


    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)


    # sentence for testing attention map
    sentence = "The Republican nominee, John McCain, has worn the uniform of our country with bravery and distinction, and for that we owe him our gratitude and respect."
    
    global n_embd, batch_size, block_size, learning_rate, n_head, n_layer, n_input, n_output, n_hidden, epochs_CLS
    global max_iters, eval_interval, eval_iters

    if args.part1:
        
        # load data
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

        # create token embedder
        token_embedder = Token_Embedding(vocab_size=tokenizer.vocab_size, 
                                         d_model=n_embd)
        token_embedder.to(device)
        
        # create position embedder
        positional_embedder = LearnableAbsolutePositionalEmbedding(block_size=block_size, d_model=n_embd)
        positional_embedder.to(device)
        
        # create transformer blocks
        encoder = TransformerLayers(d_model=n_embd, n_heads=n_head, ff_hidden_size=int(4 * n_embd),
                                    n_single_blocks=n_layer, dropout_p=0.5, 
                                    token_embedder=token_embedder, 
                                    positional_embedder=positional_embedder, needMask=False, 
                                    preLN=True, attentionMode=['full', 2])
        encoder.to(device)
        
        # create output head  
        output_head = nn.Sequential(nn.Linear(n_embd, n_hidden), 
                                    nn.ReLU(), 
                                    nn.Linear(n_hidden, n_output))
        output_head.to(device)
        
        # create classifier
        classifier = Classifier(transformer_blocks=encoder, output_head=output_head)
        classifier.to(device)

        print("\n=============== Training Part1 Transformer Encoder Classifier ===============\n")
        train_cls(classifier=classifier, 
                  train_CLS_loader=train_CLS_loader, 
                  test_CLS_loader=test_CLS_loader, 
                  n_epochs=epochs_CLS,
                  verbose = True, 
                  learning_rate=learning_rate)
        print("\n=============== Finished Training ===============\n")

        print("\n=============== Doing Part1 Sanity Check ===============\n")
        folder_path = "part1_attention_maps"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        sanity_check(tokenizer=tokenizer,
                     model=encoder,
                     sentence=sentence,
                     folder_path=folder_path)
        print(f"=============== Images Stored in {folder_path} folder ===============\n")

        gc.collect()



    if args.part2:
        # load data
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_hbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_wbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_obama.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_obama_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)

        # create token embedder
        token_embedder = Token_Embedding(vocab_size=tokenizer.vocab_size, 
                                         d_model=n_embd)
        token_embedder.to(device)
        
        # create position embedder
        positional_embedder = LearnableAbsolutePositionalEmbedding(block_size=block_size, d_model=n_embd)
        positional_embedder.to(device)

        # create transformer blocks
        decoder = TransformerLayers(d_model=n_embd, n_heads=n_head, ff_hidden_size=int(4 * n_embd),
                                    n_single_blocks=n_layer, dropout_p=0.4,
                                    token_embedder=token_embedder, 
                                    positional_embedder=positional_embedder, needMask=True, 
                                    preLN=False, attentionMode=['full', 2])
        positional_embedder.to(device)
        
        
        # output head
        output_head = nn.Linear(n_embd, tokenizer.vocab_size)
        output_head.to(device)

        # LM generator 
        generator = LMGenerator(transformer_blocks=decoder, 
                                output_head=output_head, 
                                loss_fn=nn.CrossEntropyLoss())
        generator.to(device)

        print("\n=============== Training Part2 Transformer Decoder Generator ===============\n")
        train_generator(generator=generator, 
                        train_dataloader=train_LM_loader, 
                        test_hbush_dataloader=test_hbush_loader, 
                        test_wbush_dataloader=test_wbush_loader, 
                        test_obama_dataloader=test_obama_loader,
                        learning_rate=learning_rate)
        print("\n=============== Finished Training ===============\n")

        
        print("\n=============== Doing Part2 Sanity Check ===============\n")
        folder_path = "part2_attention_maps"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        sanity_check(tokenizer=tokenizer, 
                     model=decoder, 
                     sentence=sentence, 
                     folder_path=folder_path)
        print(f"=============== Images Stored in {folder_path} folder ===============\n")

        gc.collect()


    if args.sparseAttention:

        # load data
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)


        for mode in ['atrous', 'local', 'sparse']: 
            # create token embedder
            token_embedder = Token_Embedding(vocab_size=tokenizer.vocab_size, 
                                            d_model=n_embd)
            token_embedder.to(device)
            
            # create position embedder
            positional_embedder = LearnableAbsolutePositionalEmbedding(block_size=block_size, d_model=n_embd)
            positional_embedder.to(device)
            
            # create transformer blocks
            encoder = TransformerLayers(d_model=n_embd, n_heads=n_head, ff_hidden_size=int(4 * n_embd),
                                        n_single_blocks=n_layer, dropout_p=0.5, 
                                        token_embedder=token_embedder, 
                                        positional_embedder=positional_embedder, needMask=False, 
                                        preLN=True, attentionMode=[mode, 2])
            encoder.to(device)
            
            
            # create output head  
            output_head = nn.Sequential(nn.Linear(n_embd, n_hidden), 
                                        nn.ReLU(), 
                                        nn.Linear(n_hidden, n_output))
            output_head.to(device)
            
            # create classifier
            classifier = Classifier(transformer_blocks=encoder, output_head=output_head)
            classifier.to(device)
            
            print(f"\n=============== Training Transformer Encoder Classifier with {mode} Attention ===============\n")
            train_cls(classifier=classifier, 
                    train_CLS_loader=train_CLS_loader, 
                    test_CLS_loader=test_CLS_loader, 
                    n_epochs=epochs_CLS, 
                    verbose = False, 
                    learning_rate=learning_rate)
            print("\n=============== Finished Training ===============\n")

            print(f"\n=============== Doing Sanity Check for {mode} Attention ===============\n")
            folder_path = f"part3_attention_maps_{mode}_attention"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            sanity_check(tokenizer=tokenizer, 
                        model=encoder, 
                        sentence=sentence, 
                        folder_path=folder_path, 
                        showImage=False)
            print(f"=============== Images Stored in {folder_path} folder ===============\n")

            gc.collect()



    if args.hptuning: 

        # load data
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_hbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_wbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_obama.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_obama_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)

        lrs = [5e-4, 1e-3, 2e-3]
        n_embds = [32, 64, 128] 
        n_heads = [2, 4, 8]
        dropout_probs = [0.2, 0.4, 0.6, 0.8]
        include_scheduler = [True, False]

        hyperparameters_ = [lrs, n_embds, n_heads, dropout_probs, include_scheduler]
        hyperparameters = list(product(*hyperparameters_))

        print(f"\n{len(hyperparameters)} number of settings to try in total\n")

        # store metrics
        train_perps, obama_perps, hbush_perps, wbush_perps = [], [], [], []
        
        starttime = time.time()
        
        for idx, hyperparameter in enumerate(hyperparameters):
            print(f"Setting {idx + 1}")
            lr, n_embd, n_head, dropout_p, include_scheduler = hyperparameter
            print(f"LR = {lr} | n_embd = {n_embd} | n_head = {n_head} | dropout_p = {dropout_p} | include_scheduler = {include_scheduler} ")


            # create token embedder
            token_embedder = Token_Embedding(vocab_size=tokenizer.vocab_size, 
                                            d_model=n_embd)
            token_embedder.to(device)
        
            # create position embedder
            positional_embedder = LearnableAbsolutePositionalEmbedding(block_size=block_size, d_model=n_embd)
            positional_embedder.to(device)

            # create transformer blocks
            decoder = TransformerLayers(d_model=n_embd, n_heads=n_head, ff_hidden_size=int(4 * n_embd),
                                        n_single_blocks=n_layer, dropout_p=dropout_p,
                                        token_embedder=token_embedder, 
                                        positional_embedder=positional_embedder, needMask=True, 
                                        preLN=False, attentionMode=['full', 2])
            positional_embedder.to(device)
        
            # output head
            output_head = nn.Linear(n_embd, tokenizer.vocab_size)
            output_head.to(device)

            # LM generator 
            generator = LMGenerator(transformer_blocks=decoder, 
                                    output_head=output_head, 
                                    loss_fn=nn.CrossEntropyLoss())
            generator.to(device)

            train_perp, obama_perp, hbush_perp, wbush_perp = generator_hptuning(generator=generator, 
                                                                                train_dataloader=train_LM_loader, 
                                                                                test_hbush_dataloader=test_hbush_loader, 
                                                                                test_wbush_dataloader=test_wbush_loader, 
                                                                                test_obama_dataloader=test_obama_loader,
                                                                                learning_rate=lr,
                                                                                include_scheduler=include_scheduler)

            train_perps.append(train_perp)
            obama_perps.append(obama_perp)
            hbush_perps.append(hbush_perp)
            wbush_perps.append(wbush_perp)

            print() ; print()    

        endtime = time.time()
        print(f"\n Finished in {( (endtime - starttime) / 60):.4f} min \n")

        train_setting = train_perps.index( min(train_perps) ) + 1
        obama_setting = obama_perps.index( min(obama_perps) ) + 1
        hbush_setting = hbush_perps.index( min(hbush_perps) ) + 1
        wbush_setting = wbush_perps.index( min(wbush_perps) ) + 1

        print(f"Setting {train_setting} has the lowest training perplexity")
        print(f"Setting {obama_setting} has the lowest Obama perplexity")
        print(f"Setting {hbush_setting} has the lowest HBush perplexity")
        print(f"Setting {wbush_setting} has the lowest WBush perplexity")


    if args.optimalhp:

        # load data
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_hbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_wbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_obama.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        test_obama_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)


        lrs = [2e-3, 1e-3, 1e-3, 1e-3]
        n_embds = [128, 128, 128, 128] 
        n_heads = [4, 4, 8, 2]
        dropout_probs = [0.2, 0.2, 0.2, 0.2]
        include_scheduler = [False, True, True, True]
        datasets_names = ["Training Set", "Obama Testing Set", "HBush Testing Set", "WBush Testing Set" ]
        dataloaders = [train_LM_loader, test_obama_loader, test_hbush_loader, test_wbush_loader]


        hyperparameters = zip(lrs, n_embds, n_heads, dropout_probs, include_scheduler, datasets_names, dataloaders)


        for idx, hyperparameter in enumerate(hyperparameters):

            lr, n_embd, n_head, dropout_p, include_scheduler, datasets_name, dataloader = hyperparameter

            print(f"\nBest Hyperparameter Combination for {datasets_name}")
            print(f"LR = {lr} | n_embd = {n_embd} | n_head = {n_head} | dropout_p = {dropout_p} | include_scheduler = {include_scheduler} ")

            # create token embedder
            token_embedder = Token_Embedding(vocab_size=tokenizer.vocab_size,
                                             d_model=n_embd)
            token_embedder.to(device)
        
            # create position embedder
            positional_embedder = LearnableAbsolutePositionalEmbedding(block_size=block_size, d_model=n_embd)
            positional_embedder.to(device)

            # create transformer blocks
            decoder = TransformerLayers(d_model=n_embd, n_heads=n_head, ff_hidden_size=int(4 * n_embd),
                                        n_single_blocks=n_layer, dropout_p=dropout_p,
                                        token_embedder=token_embedder, 
                                        positional_embedder=positional_embedder, needMask=True, 
                                        preLN=False, attentionMode=['full', 2])
            positional_embedder.to(device)
        
            # output head
            output_head = nn.Linear(n_embd, tokenizer.vocab_size)
            output_head.to(device)

            # LM generator 
            generator = LMGenerator(transformer_blocks=decoder, 
                                    output_head=output_head, 
                                    loss_fn=nn.CrossEntropyLoss())
            generator.to(device)

            # calculate perplexity
            perp = optimal_generator_hptuning(generator=generator, 
                                              train_dataloader=train_LM_loader,
                                              test_dataloader=dataloader,
                                              learning_rate=lr,
                                              include_scheduler=include_scheduler)

            print(f"Best Perplexity for {datasets_name} = {perp:.4f}")

            print()


if __name__ == "__main__":
    main()