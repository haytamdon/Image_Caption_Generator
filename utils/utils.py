import torch
from torch.utils.tensorboard import SummaryWriter

def load_from_checkpoint(model, PATH):
    model.load_state_dict(torch.load(PATH))
    
def log_tensorboard(writer ,loss, epoch):
    writer.add_scalar('Train_Loss', train_loss, epoch)