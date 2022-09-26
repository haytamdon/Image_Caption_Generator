import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from models.models import *
import onnx

def model_scripting(model):
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, 'scripted_CONV_LSTM.pt')

def model_tracing(model):
    input_img = torch.ones(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, input_img)
    torch.jit.save(scripted_model, 'traced_CONV_LSTM.pt')

def export_to_onnx(model):
    input_img = torch.ones(1, 3, 224, 224)
    torch.onnx.export(model, input_img, "CONV_LSTM.onnx")
    
#Optional


if __name__ == "__main__":
    model_path = '../checkpoints/best-model-parameters.pth'
    with open('./data_dir/train_vocab.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    model = CONV_LSTM_Model(256, 256, 512, len(train_vocab), 1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    
    