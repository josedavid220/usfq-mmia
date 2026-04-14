import torch
from utils import get_pennfudan_dataloaders, visualize_predictions
from models import ObjectCountingModel

def evaluate_model(model_path='object_counting_model.pth', batch_size=2, threshold=0.35):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, val_loader, category_names = get_pennfudan_dataloaders(batch_size)
    
    model = ObjectCountingModel(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, _ in val_loader:
            images = [image.to(device) for image in images]
            
            predictions = model(images)
            for img, pred in zip(images, predictions):

                counts = {}
                for label, score in zip(pred['labels'], pred['scores']):
                    if score >= threshold:  # Threshold confianza
                        label_name = category_names[label]
                        counts[label_name] = counts.get(label_name, 0) + 1
                
                print("Object counts:", counts)
                
                
                visualize_predictions(img.cpu(), pred, category_names, threshold=threshold)
            
            break  

evaluate_model()