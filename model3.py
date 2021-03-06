import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.utils import load_state_dict_from_url
from torch.nn.utils.rnn import pack_padded_sequence

class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.clone().detach()
    def close(self):
        self.hook.remove()

# This is a simple model that returns that last fully-connected layer of a Resnet 18 CNN      
class EncoderCNN(resnet.ResNet):
    def __init__(self):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2])
        state_dict = load_state_dict_from_url(resnet.model_urls['resnet18'], progress=True)
        self.load_state_dict(state_dict)
        self.activation = SaveFeatures(list(self.children())[-1])
     
    def __call__(self, inputs):
        super().__call__(inputs)
        return self.activation.features
    
    # Pass a list of ints representing the modules of the encoder for which you want to extract features
    def create_forward_hooks(self, layer_list):
        modules = list(self.modules())
        self.activations = {i: SaveFeatures(modules[i]) for i in layer_list}
    
    # Pass the int value of the layer for which you want a feature map.  Only call this after passing 
    # inputs to the encoder and you will get the output associated with those inputs
    def extract_layer_features(self, layer):
        return self.activations[layer].features
    
    def close_forward_hooks(self):
        for activation in self.activations.values():
            activation.close()

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids