import argparse
import torch
from unet import UNet
from NewTrain import train_model, get_args
import torchsummary

def train_sequential_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sizes = [[350, 350], [500, 500], [750, 664], [1276, 664]]
    
    args = get_args()
    
    best_model = None
    for i, size in enumerate(sizes):
        print(f"Training model with dataset size: {size}")
        
        # Initialize model
        if args.TempDim:
            model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        elif args.BGS_input:
            model = UNet(n_channels=2, n_classes=args.classes, bilinear=args.bilinear)
        else:
            model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
        
        model = model.to(device=device)
        model.to(memory_format=torch.channels_last)
        torchsummary.summary(model, input_size=(model.n_channels, size[0], size[1]))

        # Load the weights from the best model of the previous stage, if available
        if best_model:
            model.load_state_dict(best_model)
            print(f'Loaded model weights from the previous stage')
        
        # Train the model
        best_model = train_model(
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_checkpoint=True,
            img_scale=args.scale,
            rotation=args.TrainRotation,
            verticalFlip=args.verticalFlip,
            contrast=args.contrast,
            TempDim=args.TempDim,
            ModLoss=args.ModLoss,
            BGS_input=args.BGS_input,
            GaussianNoise=args.GaussianNoise,
            SpeckleNoise=args.SpeckleNoise,
            patience=args.patience,
            early_stopping_patience=args.early_stopping_patience,
            dataset_size=size
        )
        
        # Save the best model for the current stage
        # model_save_path = f'model_stage_{i+1}.pth'
        # torch.save(best_model.state_dict(), model_save_path)
        # print(f'Model for stage {i+1} saved to {model_save_path}')

if __name__ == '__main__':
    train_sequential_models()

