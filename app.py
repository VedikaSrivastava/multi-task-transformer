from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-Task Model Definition (DO NOT change this model architecture/class)
class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', pooling_strategy='mean', num_classification_labels=4, num_ner_labels=5):
        """
        Multi-task model that extends a Sentence Transformer to support:
          - Task A: Sentence Classification
          - Task B: Named Entity Recognition
        """
        super(MultiTaskSentenceTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        if pooling_strategy not in ['mean', 'cls']:
            raise ValueError("Unsupported pooling strategy. Choose 'mean' or 'cls'.")
        self.pooling_strategy = pooling_strategy
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classification_labels)
        self.ner_head = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, sentences=None, inputs=None):
        """
        Forward pass for either raw sentences (for classification) or pre-tokenized inputs (for NER).
        """
        if inputs is None:
            if sentences is None:
                raise ValueError("Either sentences or inputs must be provided.")
            inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        # Ensure inputs are on the proper device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.transformer(**inputs)
        token_embeddings = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        if self.pooling_strategy == 'cls':
            sentence_embeddings = token_embeddings[:, 0, :]
        else:
            sentence_embeddings = torch.mean(token_embeddings, dim=1)

        classification_logits = self.classifier(sentence_embeddings)
        ner_logits = self.ner_head(token_embeddings)

        return {
            'classification_logits': classification_logits,
            'ner_logits': ner_logits,
            'inputs': inputs
        }

# Initialize, load weights, and prepare the model
model = MultiTaskSentenceTransformer()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Label mappings
classification_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
ner_labels = {0: "O", 1: "PER", 2: "ORG", 3: "LOC", 4: "MISC"}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    example_texts = [
        "Fears for T N pension after talks Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.",
        "The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com) SPACE.com - TORONTO, Canada -- A second team of rocketeers competing for the $36.10 million Ansari X Prize, a contest for privately funded suborbital space flight, has officially announced the first launch date for its manned rocket",
        "Ky. Company Wins Grant to Study Peptides (AP) AP - A company founded by a chemistry researcher at the University of Louisville won a grant to develop a method of producing better peptides, which are short chains of amino acids, the building blocks of proteins.",
    ]
    example_text_ner = ["cricket leicestershire take over at top after innings victory"]

    if request.method == "POST":
        input_text = request.form.get("input_text")
        task = request.form.get("task")
        if input_text:
            with torch.no_grad():
                if task == "classification":
                    outputs = model(sentences=[input_text])
                    logits = outputs["classification_logits"]
                    pred = torch.argmax(logits, dim=1).item()
                    result = f"Predicted Classification: {classification_labels[pred]}"
                elif task == "ner":
                    inputs_data = model.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
                    outputs = model(inputs=inputs_data)
                    ner_logits = outputs["ner_logits"]
                    preds = torch.argmax(ner_logits, dim=2)
                    tokens = model.tokenizer.convert_ids_to_tokens(inputs_data["input_ids"][0])
                    ner_result = " ".join([f"{tok}:{ner_labels[pred]}" for tok, pred in zip(tokens, preds[0].tolist())])
                    result = f"NER Output: {ner_result}"
                elif task == "embedding":
                    inputs_data = model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                    inputs_data = {k: v.to(device) for k, v in inputs_data.items()}
                    outputs_transformer = model.transformer(**inputs_data)
                    token_embeddings = outputs_transformer.last_hidden_state  # (1, seq_length, hidden_size)
                    if model.pooling_strategy == 'cls':
                        sentence_embeddings = token_embeddings[:, 0, :]
                    else:
                        sentence_embeddings = torch.mean(token_embeddings, dim=1)
                    padding_mask = inputs_data["attention_mask"]

                    # Convert embeddings and padding mask to lists for display purposes
                    embeddings_list = sentence_embeddings.cpu().detach().numpy().tolist()
                    padding_list = padding_mask.cpu().detach().numpy().tolist()

                    # Get the shape/dimensions of the sentence embeddings
                    embedding_shape = sentence_embeddings.shape  # (batch_size, hidden_dim)
                    result = (
                        f"Sentence Embeddings (Shape: {embedding_shape}):\n{embeddings_list}\n\n"
                        f"Padding Mask:\n{padding_list}"
                    )
    return render_template("index.html", result=result, example_texts=example_texts, example_texts_ner=example_text_ner)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
