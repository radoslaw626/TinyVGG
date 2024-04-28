
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)


  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  


def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:

    model_path = Path(model_path)
    assert model_path.exists(), "Model file does not exist"
    assert model_path.suffix in ['.pt', '.pth'], "Model file should be a '.pt' or '.pth' file"

    print(f"[INFO] Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path))

    return model


def plot_loss_curves(results: Dict[str, List[float]]):
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  accuracy = results["train_acc"]
  test_accuracy = results["test_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label="train_accuracy")
  plt.plot(epochs, test_accuracy, label="test_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend();
