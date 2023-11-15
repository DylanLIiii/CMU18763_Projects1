import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def plot_loss_curves(log_dir, model_type):
    """
    Plots the training and validation loss curves from TensorBoard logs.
    :param log_dir: Directory where TensorBoard logs are stored.
    :param model_type: Type of the model (e.g., 'MLP', 'Transformer') for title.
    """
    writer = SummaryWriter(log_dir=log_dir)
    all_data = writer.all_scalars()

    # Extracting loss data
    train_loss = all_data['Loss/Train']
    val_loss = all_data['Loss/Validation']

    epochs = range(1, len(train_loss['value']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss['value'], 'r-', label='Training Loss')
    plt.plot(epochs, val_loss['value'], 'b-', label='Validation Loss')
    plt.title(f'{model_type} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    log_dir = "path_to_tensorboard_log_directory"  # Update with actual log directory
    model_type = "MLP"  # Update with the model type you used
    plot_loss_curves(log_dir, model_type)
