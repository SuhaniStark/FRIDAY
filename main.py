import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

model_name = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  
epochs = 10
batch_size = 16
learning_rate = 2e-5
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch in dataloader:
        input_ids, labels = batch
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

user_input = input("Victory: ")

input_ids = tokenizer.encode(user_input, return_tensors="pt")

output = model.generate(input_ids, max_length=100,   
 num_beams=4, temperature=0.7)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("FRIDAY:", generated_text)
