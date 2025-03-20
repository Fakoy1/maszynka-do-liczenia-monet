# Instalacja potrzebnych bibliotek
!pip install torch torchvision tqdm opencv-python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from google.colab import files, drive

# Montowanie Google Drive (opcjonalnie, do zapisywania modelu)
drive.mount('/content/drive')

# Wczytanie obrazów treningowych
print("Wczytaj zdjęcia monet. Zalecane jest min. 8 zdjęć różnych monet.")
uploaded = files.upload()

# Utworzenie listy ścieżek do wczytanych obrazów
image_paths = []
for filename in uploaded.keys():
    image_path = os.path.join('/content', filename)
    image_paths.append(image_path)
    print(f"Wczytano: {image_path}")


# Funkcja pomocnicza do wyświetlania obrazów
def show_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('on')  # Pokaż osie aby odczytać koordynaty
    plt.title(f"Obraz: {os.path.basename(image_path)}")
    plt.show()
    return img.shape

# Wyświetl obrazy, aby określić koordynaty monet
image_shapes = {}
for path in image_paths:
    print(f"Obraz: {path}")
    shape = show_image(path)
    image_shapes[path] = shape
    print("Zapisz koordynaty bounding box (x1, y1, x2, y2) i klasę monety")
    print("Klasy monet: 1=1zł, 2=2zł, 3=5zł, 4=10gr, 5=20gr, 6=50gr, 7=1gr, 8=5gr")
    print("-------------------------------")

# Ręczne wprowadzenie etykiet (przykładowe dane - musisz je zamienić na właściwe)
# W prawdziwym scenariuszu użyj narzędzia do anotacji jak LabelImg
labels = []

# Przykład jak mogłyby wyglądać twoje etykiety (zamień na właściwe wartości!)
for i, path in enumerate(image_paths):
    # Przykładowa etykieta - musisz zmienić te wartości na podstawie własnych obserwacji
    if i % 8 == 0:
        class_id = 1  # 1 zł
        box = [50, 50, 150, 150]  # [x1, y1, x2, y2] w pikselach
    elif i % 8 == 1:
        class_id = 2  # 2 zł
        box = [60, 60, 160, 160]
    elif i % 8 == 2:
        class_id = 3  # 5 zł
        box = [70, 70, 170, 170]
    elif i % 8 == 3:
        class_id = 4  # 10 gr
        box = [80, 80, 180, 180]
    elif i % 8 == 4:
        class_id = 5  # 20 gr
        box = [90, 90, 190, 190]
    elif i % 8 == 5:
        class_id = 6  # 50 gr
        box = [100, 100, 200, 200]
    elif i % 8 == 6:
        class_id = 7  # 1 gr
        box = [110, 110, 210, 210]
    else:
        class_id = 8  # 5 gr
        box = [120, 120, 220, 220]
    
    labels.append({"boxes": [box], "labels": [class_id]})

print("Utworzono przykładowe etykiety. W rzeczywistym projekcie powinieneś zdefiniować je na podstawie swoich obrazów.")



# Definicja datasetu
class CoinDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            print(f"Failed to load image: {self.image_paths[idx]}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        if idx >= len(self.labels):
            print(f"Warning: Label index {idx} out of bounds. Using default label.")
            label = {"boxes": np.array([[0, 0, 10, 10]]), "labels": np.array([0])}
        else:
            label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {"image": image, 
                "boxes": torch.tensor(label["boxes"], dtype=torch.float32),
                "labels": torch.tensor(label["labels"], dtype=torch.long)}

# Przygotowanie transformacji
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Utworzenie datasetu i dataloadera
train_ds = CoinDataset(image_paths, labels, transform)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# Definicja urządzenia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {DEVICE}")

# Model do rozpoznawania monet
class CoinDetector(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_bbox = nn.Linear(256, 4)  # bounding box (x1, y1, x2, y2)
        self.fc_class = nn.Linear(256, num_classes)  # klasy monet + tło
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        bbox = self.fc_bbox(x)
        class_logits = self.fc_class(x)
        
        return {"boxes": bbox, "labels": class_logits}

# Inicjalizacja modelu
model = CoinDetector().to(DEVICE)

# Definicja funkcji straty
class CoinLoss(nn.Module):
    def __init__(self, lambda_bbox=1.0, lambda_class=1.0):
        super().__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.class_loss = nn.CrossEntropyLoss()
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class
        
    def forward(self, preds, targets):
        bbox_loss = self.bbox_loss(preds["boxes"], targets["boxes"].squeeze(1))
        class_loss = self.class_loss(preds["labels"], targets["labels"].squeeze(1))
        total_loss = self.lambda_bbox * bbox_loss + self.lambda_class * class_loss
        return total_loss

# Funkcje do wartości monet
def calculate_coin_value(coin_class):
    value_map = {
        1: 100,   # 1 złoty = 100 groszy
        2: 200,   # 2 złote = 200 groszy
        3: 500,   # 5 złotych = 500 groszy
        4: 10,    # 10 groszy
        5: 20,    # 20 groszy
        6: 50,    # 50 groszy
        7: 1,     # 1 grosz
        8: 5      # 5 groszy
    }
    return value_map.get(coin_class, 0)



# Funkcja trenująca
def train_model(model, train_loader, epochs=15, lr=0.001):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CoinLoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["image"].to(DEVICE)
            targets = {
                "boxes": batch["boxes"].to(DEVICE),
                "labels": batch["labels"].to(DEVICE)
            }
            
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Zapisz najlepszy model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "/content/coin_detector_best.pth")
            print(f"Zapisano najlepszy model z loss: {best_loss:.4f}")
    
    # Zapisz ostatni model
    torch.save(model.state_dict(), "/content/coin_detector_last.pth")
    print("Trenowanie zakończone!")
    
    # Opcjonalnie zapisz na Google Drive
    try:
        torch.save(model.state_dict(), "/content/drive/MyDrive/coin_detector.pth")
        print("Model zapisany na Google Drive")
    except:
        print("Nie udało się zapisać modelu na Google Drive")

# Trenowanie modelu
print("Rozpoczynanie treningu...")
train_model(model, train_loader, epochs=15)


# Funkcja do przetwarzania nowego obrazu
def process_image(image_path, model):
    model.eval()
    
    # Wczytanie i przygotowanie obrazu
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    h, w = image.shape[:2]
    
    # Przygotowanie do sieci
    image_resized = cv2.resize(image, (224, 224))
    image_tensor = transform(image_resized).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Pobranie wyników
    pred_boxes = predictions["boxes"][0].cpu().numpy()
    pred_class = torch.argmax(predictions["labels"][0]).item()
    
    # Skalowanie bounding boxów do oryginalnego rozmiaru
    pred_boxes[0] = pred_boxes[0] * w / 224
    pred_boxes[1] = pred_boxes[1] * h / 224
    pred_boxes[2] = pred_boxes[2] * w / 224
    pred_boxes[3] = pred_boxes[3] * h / 224
    
    # Obliczenie wartości monety
    coin_value = calculate_coin_value(pred_class)
    
    # Wyświetlenie wyników
    result_image = original_image.copy()
    x1, y1, x2, y2 = map(int, pred_boxes)
    
    # Konwersja wartości na format tekstowy
    if coin_value >= 100:
        text = f"{coin_value//100} zł"
    else:
        text = f"{coin_value} gr"
    
    # Rysowanie na obrazie
    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(result_image, f"Klasa: {pred_class}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result_image, f"Wartość: {text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Wyświetlenie wyniku
    plt.figure(figsize=(10, 8))
    plt.imshow(result_image)
    plt.title(f"Rozpoznano: {text}")
    plt.axis('off')
    plt.show()
    
    return result_image, coin_value

# Wczytanie nowego obrazu do testów
print("Wczytaj zdjęcie monety do rozpoznania:")
test_uploaded = files.upload()
test_image_path = list(test_uploaded.keys())[0]
test_image_path = os.path.join('/content', test_image_path)

# Wczytanie najlepszego modelu
model.load_state_dict(torch.load("/content/coin_detector_best.pth"))
model.to(DEVICE)

# Testowanie modelu
result_image, coin_value = process_image(test_image_path, model)

# Wyświetlenie wartości
if coin_value >= 100:
    zlote = coin_value // 100
    grosze = coin_value % 100
    print(f"Rozpoznana wartość: {zlote} zł {grosze} gr")
else:
    print(f"Rozpoznana wartość: {coin_value} gr")