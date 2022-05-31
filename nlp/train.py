from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from bert import Dataset
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix, classification_report


# Classification model
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs, save_model=False, save_dir='weight.pt'):
    """
    Training function using given model.
    :param model: BERT model for question classification.
    :param train_data: Training data from train_questions.csv
    :param val_data: Validation data from train_questions.csv
    :param learning_rate: Learning rate: 1e-6
    :param epochs: Epochs: 5
    :param save_model: Boolean value for saving model or not.
    :param save_dir: Path to save model.
    :return: None
    """
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

        if save_model:
            torch.save(model.state_dict(), save_dir)


def evaluate(model, test_data):
    """
    Evaluation function after training for BERT model.
    :param model: BERT model for evaluation.
    :param test_data: df_test data for testing accuracy and other metrics of model.
    :return: None
    """
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    y_true, y_pred = [], []
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            pred = output.argmax(dim=1)
            y_true.extend(test_label.tolist())
            y_pred.extend(pred.tolist())

            acc = (pred == test_label).sum().item()
            total_acc_test += acc

    report = classification_report(y_true, y_pred)
    print("Classification Report:", )
    print(report)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    print(cm)
    cm_df = pd.DataFrame(cm, columns=[0, 1, 2], index=[0, 1, 2])
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    heatmap(cm_df, cmap='Blues', annot=True)
    plt.savefig('model/confusion_matrix.png')
    plt.show()
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


# Creating new models with its current date value.
def get_current_date():
    from datetime import datetime
    return datetime.today().strftime('%Y_%m_%d')


if __name__ == '__main__':
    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6
    current_date = get_current_date()

    df = pd.read_csv('csv_data/train_questions.csv')
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])
    # train(model, df_train, df_val, LR, EPOCHS, True, save_dir=f'model/bert_{current_date}.pt')
    # print('Train finished successfully..')

    model.load_state_dict(torch.load('model/bert.pt'))  # GPU
    # model.load_state_dict(torch.load('model/bert.pt', map_location=torch.device('cpu')))

    evaluate(model, df_test)
    print('Evaluation finished successfully..')