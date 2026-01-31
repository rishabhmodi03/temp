
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

device = 'cpu'
    
model=None
tokenizer=None



def predict(text, model, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.sigmoid(outputs.logits)  # Convert logits to probabilities
    return predictions


model_name = "ifmain/ModerationBERT-En-02"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=18)

# Device configuration
model.to(device)
new_text = "hello"
predictions = predict(new_text, model, tokenizer)

# Define the categories
categories = ['harassment', 'harassment_threatening', 'hate', 'hate_threatening',
          'self_harm', 'self_harm_instructions', 'self_harm_intent', 'sexual',
          'sexual_minors', 'violence', 'violence_graphic', 'self-harm',
          'sexual/minors', 'hate/threatening', 'violence/graphic',
          'self-harm/intent', 'self-harm/instructions', 'harassment/threatening']

# Convert predictions to a dictionary
category_scores = {categories[i]: predictions[0][i].item() for i in range(len(categories))}
sorted_scores = dict(sorted(category_scores.items(), key=lambda x: x[1], reverse=True))

print(sorted_scores)

with open('a.txt','w+') as file:
  file.write(str(sorted_scores))

