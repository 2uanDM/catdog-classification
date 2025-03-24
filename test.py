from src.utils.dataset import CatDogDataset

data = CatDogDataset(
    split_ratio=0.8,
    data_dir="data",
)

train_dataset = data.from_mode(mode="train")
val_dataset = data.from_mode(mode="val")

print(len(train_dataset))
print(len(val_dataset))

print(val_dataset[0])
print(train_dataset[1])
