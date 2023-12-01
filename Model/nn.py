import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data_root = Path("/home/deepcat/Documents/Course/DS/kkdata3")


class Dataset(Dataset):
    def __init__(self, source_data, target_data):
        self.src = torch.from_numpy(source_data).type(torch.long)
        self.tgt = torch.from_numpy(target_data).type(torch.long)
        self.length = len(source_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


class TestDataset(Dataset):
    def __init__(self, source_data):
        self.src = torch.from_numpy(source_data).type(torch.long)
        self.length = len(source_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.src[idx]


# collect kkdata
def collect_data(data_root):
    train_source = pd.read_parquet(data_root / "label_train_source.parquet")
    train_target = pd.read_parquet(data_root / "label_train_target.parquet")
    test_source = pd.read_parquet(data_root / "label_test_source.parquet")
    meta_song = pd.read_parquet(data_root / "meta_song.parquet")

    # sort data
    train_source.sort_values(["session_id", "listening_order"], inplace=True)
    train_target.sort_values(["session_id", "listening_order"], inplace=True)
    # test_source.sort_values(["session_id", "listening_order"], inplace=True)

    # map song_id to song_index to save memory and speed up
    meta_song["song_index"] = meta_song.index
    train_source = train_source.merge(
        meta_song[["song_id", "song_index"]], on="song_id", how="left"
    )
    train_target = train_target.merge(
        meta_song[["song_id", "song_index"]], on="song_id", how="left"
    )
    test_source = test_source.merge(
        meta_song[["song_id", "song_index"]], on="song_id", how="left"
    )
    del train_source["song_id"]
    del train_target["song_id"]
    del test_source["song_id"]

    return train_source, train_target, test_source, meta_song


# filter duplicate song id
def filterDuplicate(df_src, df_tgt):
    df_src.reset_index(drop=True, inplace=True)
    df_tgt.reset_index(drop=True, inplace=True)

    # find duplicate song_id
    duplicates_mask = df_src.duplicated()

    # filter duplicate song_id
    src_filtered = df_src[~duplicates_mask].sort_values("song_index").sort_index()
    tgt_filtered = df_tgt[~duplicates_mask].sort_values("song_index").sort_index()

    if tgt_filtered.shape[0] != src_filtered.shape[0]:
        raise ValueError("src and tgt shape not match")

    return src_filtered, tgt_filtered


# return n+1 column song id
def getTrainData(df, n=2):
    df = df.copy()
    # gen n song id be the dataset
    for i in range(1, n + 1):
        df[f"next{i}_song_id"] = df["song_index"].shift(-i)

    # check if last song id is in the same session
    df[f"next{n}_session_id"] = df["session_id"].shift(-n)
    df = df.query(f"session_id == next{n}_session_id")

    # only get the song_id and next1_song_id, next2_song_id, next3_song_id... column
    df = df[["song_index"] + [f"next{i}_song_id" for i in range(1, n + 1)]]
    return df


# dataframe to numpy
def df2np(df):
    return df.values


# get testX dataset
def getTestX(df, n=2):
    trainX = pd.DataFrame()
    # if n == 3:, get listening_order : 18, 19, 20
    for i in range(n, 0, -1):
        _ = df.query(f"listening_order == {20-i+1}")[
            ["session_id", "song_index"]
        ].set_index("session_id")
        if trainX.empty:
            trainX = _
        else:
            trainX = trainX.join(_, lsuffix=f"_{n-i-1}")

    trainX = trainX.rename(columns={trainX.columns[-1]: f"song_id_{n-1}"})

    # trainX keep session_id
    trainX.reset_index(inplace=True)
    return trainX.values


# Define the neural network
class NNmodel(nn.Module):
    def __init__(self, VOCAB_SIZE, embedding_dim, output_dim):
        super(NNmodel, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        # self.conv1d = nn.Conv1d(
        #     in_channels=embedding_dim, out_channels=32, kernel_size=3
        # )

        self.fc = nn.Linear(20, 5)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.unemb_fc = nn.Linear(embedding_dim, VOCAB_SIZE)

    def forward(self, x):
        embedded = self.embedding(x)
        x = embedded.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        out = self.unemb_fc(x)
        return out


# numpy to csv
def np2csv(data_array, column_names, filename):
    # Check if the number of columns in data_array matches the length of column_names
    if data_array.shape[1] != len(column_names):
        raise ValueError(
            "Number of columns in data_array must match the length of column_names."
        )

    # Create a list with column names and data_array
    combined_data = [column_names] + data_array.tolist()

    # Save the combined list to a CSV file
    np.savetxt(filename, combined_data, delimiter=",", fmt="%s", header="", comments="")

    print(f"CSV file '{filename}' saved successfully.")


# validation
def validate(model, val_loader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_val_loss = 0

    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            # Forward pass
            output = model(src_batch)
            val_loss = criterion(output.permute(0, 2, 1), tgt_batch)
            total_val_loss += val_loss.item()

            # Calculate accuracy
            predictions = torch.argmax(output, dim=2)
            correct = (predictions == tgt_batch).sum().item()
            total_correct += correct
            total_samples += tgt_batch.numel()

    average_val_loss = total_val_loss / len(val_loader)
    accuracy = total_correct / total_samples

    return average_val_loss, accuracy


# inference
def inference(model, test_loader, device):
    model.eval()
    predicted_indices_list = []

    with torch.no_grad():
        for batch in test_loader:
            # Extract relevant data from the batch
            session_id = batch[:, 0].reshape(-1, 1)
            src = batch[:, 1:]

            # Convert test data to torch tensor
            src = src.type(torch.long).to(device)

            # Forward pass
            output = model(src)

            # Get the index with maximum probability using argmax
            predicted_indices = torch.argmax(output, dim=2).cpu().numpy()

            # Merge session_id and predicted_indices
            predicted_indices_with_session_id = np.hstack(
                (session_id, predicted_indices)
            )
            predicted_indices_list.append(predicted_indices_with_session_id)

    # Concatenate the results from all batches
    predicted_indices = np.concatenate(predicted_indices_list, axis=0)

    return predicted_indices


def main():
    train_source, train_target, test_source, meta_song = collect_data(data_root)

    src = getTrainData(train_source, 19)
    tgt = getTrainData(train_target, 4)

    src, tgt = filterDuplicate(src, tgt)
    src = df2np(src)
    tgt = df2np(tgt)

    test_src = getTestX(test_source, n=20)

    # split train and validation
    src_train, src_val, tgt_train, tgt_val = train_test_split(
        src, tgt, test_size=0.2, random_state=42
    )

    # size = 1000
    # src_train = src_train[:size]
    # src_val = src_val[:size]
    # tgt_train = tgt_train[:size]
    # tgt_val = tgt_val[:size]
    # test_src = test_src[:size]

    # Create a PyTorch dataset
    train_dataset = Dataset(src_train, tgt_train)
    val_dataset = Dataset(src_val, tgt_val)
    test_dataset = TestDataset(test_src)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # define hyperparameters
    VOCAB_SIZE = meta_song["song_index"].max() + 1
    embedding_dim = 64
    output_dim = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NNmodel(VOCAB_SIZE, embedding_dim, output_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0
    num_epochs = 20
    accuracy_list = []
    loss_list = []

    # load model
    model.load_state_dict(torch.load("model.pth"))

    # Inference
    predicted_indices = inference(model, test_loader, device)
    np2csv(
        predicted_indices,
        ["session_id", "top1", "top2", "top3", "top4", "top5"],
        "submission.csv",
    )

    # Train the model
    for epoch in range(num_epochs):
        # Training
        model.train()
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.permute(0, 2, 1), tgt)
            loss.backward()
            optimizer.step()

        # Validation
        average_val_loss, accuracy = validate(model, val_loader, criterion, device)

        accuracy_list.append(accuracy)
        loss_list.append(average_val_loss)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "model.pth")

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss}, Accuracy: {accuracy}"
        )
        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label="Validation Loss")
        plt.ylabel("CrossEntropy Loss")
        plt.title("Training Loss Curve")
        plt.savefig("loss.png")
        # Close the figure to prevent it from being displayed
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(accuracy_list, label="Validation Accuracy")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.savefig("accuracy.png")
        # Close the figure to prevent it from being displayed
        plt.close()

    # Inference
    predicted_indices = inference(model, test_loader, device)
    np2csv(
        predicted_indices,
        ["session_id", "top1", "top2", "top3", "top4", "top5"],
        "submission.csv",
    )


if __name__ == "__main__":
    main()
