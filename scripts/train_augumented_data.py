
"""
Trains a PyTorch image classification model with trivial_augumented transformer.
"""

import os
import json
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer

NUM_EPOCHS = 70
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"


simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
    ])

trivial_augumented_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
    ])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=trivial_augumented_transform,
    test_transform=simple_transform,
    batch_size=BATCH_SIZE
)

model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

start_time = timer()

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       epochs=NUM_EPOCHS,
                       device=device)

with open('results/augumented_data_model_results.json', 'w') as f:
    json.dump(results, f)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")


utils.save_model(model=model,
                 target_dir="models",
                 model_name="augumented_data_tinyvgg_model.pt")
