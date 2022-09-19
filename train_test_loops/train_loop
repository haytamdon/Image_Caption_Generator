import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import numpy as np

def train_func(model ,data_loader, device, loss_fn, optimizer, epoch):
    model.train()
    loop = tqdm(data_loader)
    total_num_steps = len(data_loader)
    for i, (imgs, caps, lens) in enumerate(loop):

        # Set mini-batch dataset
        imgs = imgs.to(device)
        caps = caps.to(device)
        tgts = pack_padded_sequence(caps, lens, batch_first=True)[0]

        # Forward, backward and optimize
        outputs = model(imgs, caps, lens)
        loss = loss_fn(outputs, tgts)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

        # Print log info
        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, 5, i, total_num_steps, loss.item(), np.exp(loss.item()))) 

        # Save the model checkpoints
        if (i+1) % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(
                'models_dir/', 'cnn_lstm_model-{}-{}.ckpt'.format(epoch+1, i+1)))
                
    return loss