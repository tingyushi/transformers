
import matplotlib.pyplot as plt
import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, folder_path, showImage = True):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1}")
            
            # Save the plot
            plt.savefig(os.path.join(folder_path, f"attention_map_{j + 1}.png"))
            
            if showImage:
                # Show the plot
                plt.show()
            else:
                plt.close()


def count_parameters(model):
    total = 0
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()} --- {torch.tensor(param.size()).prod().item()}")
        total += torch.tensor(param.size()).prod().item()
    print(f"========== Total Number of Parameters = {total} ==========")


# given a batch of token encodings, convert to sentences
def token_encode_to_sentences(tokenizer, sen_batch):
    assert len(sen_batch.shape) == 2
    batch_size , _ = sen_batch.shape
    sentences = []
    for i in range(batch_size):
        one_sen_encoding = sen_batch[i]
        sentences.append(tokenizer.decode( one_sen_encoding.to(torch.int64).numpy() ))
    return sentences