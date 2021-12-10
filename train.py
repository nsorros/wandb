import csv

from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
import typer
import torch

wandb.init(project="test-wandb", entity="nsorros")


class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        texts = []
        tags = []
        with open(data_path) as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                texts.append(row["tweet"])
                tags.append(row["hate_speech"])

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.X = self.tokenizer(texts, padding=True)["input_ids"]

        self.label_encoder = LabelEncoder()
        self.Y = self.label_encoder.fit_transform(tags)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])


class Model(torch.nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size,
        hidden_size,
        num_layers,
        bidirectional,
        num_classes,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = torch.nn.LSTM(
            embedding_size,
            hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, num_classes
        )

    def forward(self, x):
        emb_out = self.embedding(x)
        hout = self.lstm(emb_out)[0][:, -1, :]
        out = self.linear(hout)
        return out


def train(
    data_path,
    model_path,
    learning_rate: float = 1e-2,
    epochs: int = 5,
    batch_size: int = 32,
    vocabulary_size: int = 30000,
    embedding_size: int = 200,
    hidden_size: int = 100,
    num_layers: int = 2,
    bidirectional: bool = True,
    num_classes: int = 8,
):
    wandb.config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "vocabulary_size": vocabulary_size,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
    }
    dataset = ToxicityDataset(data_path)
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Model(
        vocabulary_size,
        embedding_size,
        hidden_size,
        num_layers,
        bidirectional,
        num_classes,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        batches = tqdm(
            data, desc=f"Epoch {epoch:2d}/{epochs:2d}"
        )
        for batch in tqdm(batches):
            inputs, labels = batch

            optimizer.zero_grad()

            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            batches.set_postfix({"loss": "{:.5f}".format(loss.item() / len(batch))})
            wandb.log({"loss": loss.item()})

    torch.save(model, model_path)


if __name__ == "__main__":
    typer.run(train)
