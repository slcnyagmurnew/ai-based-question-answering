import torch
from train import BertClassifier
from bert import tokenizer, labels
from utils import clean_question, list_to_string, find_objects

# model = torch.load('model/bert.pt')


def predict(model, text):
    text_dict = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    mask = text_dict['attention_mask'].to(device)
    input_id = text_dict['input_ids'].squeeze(1).to(device)

    with torch.no_grad():
        output = model(input_id, mask)
        label_id = output.argmax(dim=1).item()
        for key in labels.keys():
            if labels[key] == label_id:
                print(text, ' => ', key, '#', label_id)
                return key, label_id


# if __name__ == '__main__':
#     model = BertClassifier()
#     model.load_state_dict(torch.load('model/bert.pt', map_location=torch.device('cpu')))
#     # model.load_state_dict(torch.load('model/bert.pt'))  with GPU
#
#     text = 'What is this, can you tell me?'
#     text2 = 'What color is this shirt, how can i buy it?'
#     text3 = 'How many people are there?'
#     model.eval()
#     predict(model, text=text)
#     predict(model, text=text2)
#     predict(model, text=text3)
#
#     # cleaned_text = clean_question(text)
#     # obj = find_objects(list_to_string(cleaned_text))
#     # print(obj)
