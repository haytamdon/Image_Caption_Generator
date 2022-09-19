import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

class CNNModel(nn.Module):
    def __init__(self, embedding_size = 256):
        super(CNNModel, self).__init__()
        resnet = models.resnet152(pretrained=True) #Load from pretained resnet 152 model
        module_list = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*module_list)
        self.linear = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)
        
    def forward(self, input_images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            res_output = self.resnet(input_images)
        res_output = res_output.reshape(res_output.size(0), -1)
        output = self.batch_norm(self.linear(res_output))
        return output

class LSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_sequence_len=20):
        """Set the hyper-parameters and build the layers."""
        super(LSTMModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_sequence_len = max_sequence_len
        
    def forward(self, input_features, capts, lens):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embedding_layer(caps)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(embeddings, lens, batch_first=True) 
        hidden_variables, _ = self.lstm_layer(lstm_input)
        model_outputs = self.linear(hidden_variables[0])
        return model_outputs
    
    def sample(self, input_features, lstm_states=None):
        """Generate captions for given image features using greedy search."""
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)
        for i in range(self.max_sequence_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs, lstm_states)          # hiddens: (batch_size, 1, hidden_size)
            model_outputs = self.linear(hidden_variables.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted_outputs = model_outputs.max(1)                        # predicted: (batch_size)
            sampled_indices.append(predicted_outputs)
            lstm_inputs = self.embedding_layer(predicted_outputs)                       # inputs: (batch_size, embed_size)
            lstm_inputs = lstm_inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_indices = torch.stack(sampled_indices, 1)                # sampled_ids: (batch_size, max_sequence_length)
        return sampled_indices
    
class CONV_LSTM_Model(nn.Module):
    def __init__(self, cnn_embedding, lstm_embedding, lstm_hidden_layer_size, vocabulary_size, lstm_num_layers):
        super(CONV-LSTM-Model, self).__init__()
        self.encoder = CNNModel(cnn_embedding)
        self.decoder = LSTMModel(lstm_embedding, lstm_hidden_layer_size, vocabulary_size, lstm_num_layers)
    def forward(self, input_images, caps, lens):
        cnn_outputs = self.encoder(input_images)
        lstm_outputs = self.decoder(cnn_outputs, caps, lens)