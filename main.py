from dataset.dataset import *
from models.models import *
from preprocessing.text_preprocessing import *
from preprocessing.image_preprocessing import *
from train_test_loops.train_loop import *

if __name__ == "__main__":
    vocab = build_vocabulary(json='data_dir/annotations/captions_train2014.json', threshold=4)
    vocab_path = './data_dir/vocabulary.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    image_path = './data_dir/train2014/'
    image_output_path = './data_dir/resized_images/'
    image_shape = [256, 256]
    reshape_images(image_path, output_path, image_shape)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists('models_dir/'):
        os.makedirs('models_dir/')
        
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    
    