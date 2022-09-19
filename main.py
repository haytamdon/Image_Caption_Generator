from dataset.dataset import *
from models.models import *
from preprocessing.text_preprocessing import *
from preprocessing.image_preprocessing import *
from train_test_loops.train_loop import *
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import mlflow
from utils.utils import *

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("pytorch_experiment")
    train_vocab = build_vocabulary(json='data_dir/annotations/captions_train2014.json', threshold=4)
    vocab_path = './data_dir/train_vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    train_image_path = './data_dir/train2014/'
    train_image_output_path = './data_dir/train_resized_images/'
    if not os.path.exists(train_image_output_path):
        os.makedirs(train_image_output_path)
    image_shape = [256, 256]
    reshape_images(train_image_path, train_image_output_path, image_shape)
    val_image_path = './data_dir/val2014/'
    val_image_output_path = './data_dir/val_resized_images/'
    if not os.path.exists(val_image_output_path):
        os.makedirs(val_image_output_path)
    reshape_images(val_image_path, val_image_output_path, image_shape)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists('models_dir/'):
        os.makedirs('models_dir/')
        
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    
    train_dataloader = get_loader(train_image_output_path, 
                                    'data_dir/annotations/captions_train2014.json', train_vocab, 
                                    transform, 128,
                                    shuffle=True, num_workers=2) 
    
    model = CONV_LSTM_Model(256, 256, 512, len(train_vocab), 1).to(device)
    loss_criterion = nn.CrossEntropyLoss()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    
    writer = SummaryWriter('../tensorboard_runs/Logger')
    
    weights_path = "../checkpoints"
    os.makedirs(weights_path, exist_ok=True)
    
    least_loss = 100000
    for epoch in range(5):
        with mlflow.start_run():
            loss = train_func(model ,train_dataloader, device, loss_criterion, optimizer, epoch)
            params = {"loss_function": 'CrossEntropyLoss', "optimizer": 'Adam', 
                    "cnn_embedding": 256, "lstm_embedding": 256, "lstm_hidden_layer_size": 512, 
                    "vocabulary_size": len(train_vocab), "lstm_num_layers": 1,
                    "learning_rate": 0.001, "epoch": epoch}
            mlflow.log_params(params)
            mlflow.log_metric("loss", loss)
            mlflow.pytorch.log_model('cnn_lstm_model-{}'.format(epoch+1), artifact_path)
            if least_loss>= loss:
                least_loss = loss
                torch.save(model.state_dict(), '../checkpoints/best-model-parameters.pth')
                mlflow.register_model(model, "Conv_LSTM")
            log_tensorboard(writer, loss, epoch)