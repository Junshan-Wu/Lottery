import argparse

def get_parameters():
    parser = argparse.ArgumentParser(description="Hyperparameters for ResNet18 training")

    # Training related
    parser.add_argument('--num_epoch', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2500, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for optimizer')
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='Proportion of training data used for validation (0 means no validation split)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'exponential', 'none'],
                        help='Type of learning rate scheduler')
    parser.add_argument('--use_relay_train', type=int, default=0, choices=[0, 1],
                        help='Use relay training framework for full training (0/1)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'sigma', 'tanh'],
                        help='Activation function (relu/sigmoid/tanh)')
    # Model related
    parser.add_argument('--model_save_dir', type=str, default='./model_weights/', help='Directory to save model weights')

    # Device related
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training and testing')

    # Data related
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for dataset storage')
    parser.add_argument('--use_cutout', type=int, default=0, choices=[0,1], help='Use Cutout augmentation (0/1)')
    # Warm-up related
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warm-up epochs at the start of training (0 means no warm-up)')

    return parser.parse_args()

if __name__ == "__main__":
    params = get_parameters()
    print(params)
