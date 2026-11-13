import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import train_loader, test_loader
from src.models.base_cnn import MetalDefectCNN
from src.config import cnn_epoch
def train_model():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System Check: Executing on {device.type.upper()}")


    model = MetalDefectCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    num_epochs = cnn_epoch
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
   
        model.train() 
        running_train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
          
            outputs = model(images)
            loss = criterion(outputs, labels)
           
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)


        model.eval() 
        running_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
    
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
        avg_val_loss = running_val_loss / len(test_loader)
        val_accuracy = 100.0 * correct_predictions / total_samples
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "../artifacts/best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Checkpoint saved: {save_path}")

if __name__ == "__main__":
    train_model()
    
    
    