# Multi-Task Sentence Transformer: Implementation & Evaluation

## Overview
This project implements a multi-task sentence transformer that supports two key NLP tasks: sentence classification and named entity recognition (NER). The solution builds upon a pre-trained transformer (["sentence-transformers/all-mpnet-base-v2"](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)) to encode sentences into fixed-length embeddings and then extends the architecture with task-specific heads. Additionally, a complete training loop is provided alongside a Flask web demo for real-time inference.

## Task 1: Sentence Transformer Implementation
### Model Design and Considerations
- **Base Model:** pre-trained `"sentence-transformers/all-mpnet-base-v2"` model from Hugging Face due to its strong performance in generating high-quality sentence embeddings.
- **Pooling Strategy:** Two pooling strategies were considered:
  - **Mean Pooling:** Computes the average of token embeddings, offering a robust summary of the entire sentence.
  - **CLS Pooling:** Uses the first token's embedding as the sentence representation.
  
  For implementation, **mean pooling** was chosen to capture an aggregate representation over all tokens.

### Code Snippet

```python
class SentenceTransformer(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', pooling_strategy='mean'):
        """
        Initializes the Sentence Transformer.

        Args:
            model_name (str): Pre-trained transformer model identifier.
            pooling_strategy (str): Strategy to pool token embeddings into a sentence embedding.
        """
        super(SentenceTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        if pooling_strategy not in ['mean', 'cls']:
            raise ValueError("Unsupported pooling strategy. Choose 'mean' or 'cls'.")
        self.pooling_strategy = pooling_strategy

    def forward(self, sentences):
        """
        Forward pass to obtain sentence embeddings.
        """
        # Tokenize input
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

        # Forward pass
        outputs = self.transformer(**inputs)
        token_embeddings = outputs.last_hidden_state

        if self.pooling_strategy == 'cls':
            sentence_embeddings = token_embeddings[:, 0, :]
        else:
            sentence_embeddings = torch.mean(token_embeddings, dim=1)

        return sentence_embeddings
```
*for complete code check [code.ipynb](code.ipynb) `Task 1`. For viewing it as docker build follow [Instruction to build and run docker image](readme.md#instruction-to-build-and-run-docker-image)*

### Sample Evaluation
The model was tested on several sample sentences, including:
- "Highly intelligent, dogs have the capability of expressing their joy and happiness by wagging their tails."
- "They are known to be the most loyal of animals. Dogs can sense your pain and can be your best friend."

For each sample, the model successfully generated a fixed-length embedding, with the embedding shape confirming the expected dimensions (`torch.Size([1, 768])`).

## Task 2: Multi-Task Learning Expansion
### Architecture Enhancements
The sentence transformer was extended to support two tasks by adding two distinct heads:
- **Task A – Sentence Classification:**  
  - **Objective:** Classify sentences into one of four classes (e.g., World, Sports, Business, Sci/Tech).  
  - **Dataset:** AG News dataset (accessed via Hugging Face datasets) was used, where the label mapping is defined as `{0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}`. This is the original set of labells.
    - Due to the large size of the dataset, only 10% of the official train set was used for training purposes with a 80-20 train-validation split.
  
- **Task B – Named Entity Recognition (NER):**  
  - **Objective:** Perform token-level entity recognition.  
  - **Dataset:** CoNLL2003 dataset was used with a reduced set of NER labels `{0: "O", 1: "PER", 2: "ORG", 3: "LOC", 4: "MISC"}`.
    - Official train and validation data split were used for training and validation.

- **Architecture Modifications and Considereations**
  - **Shared Components:**  
    The transformer encoder and tokenizer are shared across both tasks.  
  - **Task-Specific Heads:**  
    A linear layer for sentence classification aggregates the sentence embedding (obtained via mean pooling) while another linear layer processes the token-level embeddings for NER.
  - **Loss Functions:**  
    - **Classification:** Cross-entropy loss.
    - **NER:** Cross-entropy loss with an `ignore_index` (set to -100) to handle padded tokens.

### Code Snippet

```python
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
```
*for complete code check [code.ipynb](code.ipynb) `Task 2 - Final Implementation`. For viewing it as docker build follow [Instruction to build and run docker image](readme.md#instruction-to-build-and-run-docker-image)*

### Sample Evaluation
Section [Task 4](writeup.md#task-4-training-loop-implementation-bonus) offers in-depth details on both the implementation and the resulting outcomes. It clearly outlines our training strategy and the performance achieved.

## Task 3: Training Considerations
### Freezing Strategies & Transfer Learning

1. **Freezing the Entire Network:**  
   - **Pros:** Leverages pre-trained features entirely; minimal computational cost.
   - **Cons:** No task-specific fine-tuning; might not capture nuances of new tasks.
   - **Rationale:** This strategy is especially useful when data is scarce, as it preserves the powerful general representations learned during pre-training and avoids the risk of overfitting on a small dataset. However, by not adapting any layers to the specific tasks, the model may fail to learn the unique features needed to excel in the target domain, potentially limiting performance improvements.

2. **Freezing Only the Transformer Backbone:**  
   - **Pros:** Keeps robust pre-trained language representations intact while fine-tuning task-specific heads.
   - **Cons:** Limits potential improvements that could be gained by adjusting the deeper layers for specific tasks.
   - **Rationale:** This approach allows the model to retain its general language understanding while focusing training efforts on the additional layers that directly address task-specific requirements, offering a good balance between stability and adaptation. However since the backbone remains static, any subtle features that could be extracted from re-tuning the model to the specific task domain are lost, potentially resulting in suboptimal performance on tasks that deviate from the pre-training domain.

3. **Freezing One of the Task-Specific Heads:**  
   - **Scenario:** When one task (e.g., classification) has a large, high-quality dataset and the other (e.g., NER) is limited.
   - **Pros:** Allows the model to specialize on the weaker task while preserving a strong head on the well-represented task.
   - **Rationale:** This strategy targets resource allocation effectively—by keeping the head for the abundant task intact, the model avoids unnecessary re-training, and it directs learning capacity toward improving the performance of the under-represented task. The imbalance in learning dynamics might lead to conflicts during training if gradients from the fine-tuning process inadvertently affect other parts of the network, necessitating a well-planned training schedule.
 
### Transfer Learning Senerio for Current Task
- **Pre-trained Model:**  
  Pick a pre-trained model like `"all-mpnet-base-v2"` because it is known for its strong performance in generating high-quality embeddings, making it an ideal foundation for both sentence classification and NER tasks.
- **Layer Adjustments:**  
  Initially, the transformer backbone is frozen and only the classification and NER heads are trained. If performance plateaus, selective unfreezing of the later transformer layers is considered.<br>
  *Why: Starting with a frozen backbone minimizes the risk of overfitting and leverages established representations; gradually unfreezing layers later allows for fine-tuning that can capture more task-specific nuances when needed.*
- **Rationale:**  
  This approach strikes a balance between leveraging pre-trained language understanding and adapting to task-specific nuances. The method ensures that we benefit from the robustness of the pre-trained model while also maintaining the flexibility to adapt when the standard representations are insufficient for capturing the intricacies of our target tasks.

## Task 4: Training Loop Implementation (BONUS)
### Training Loop Design
A basic model has been implemented to showcase multi-task learning.
The training loop alternates between the two tasks to ensure balanced updates:
- **Dataset Handling:**  
  - **AG News Dataset:** A subset (10% of the training data) was used with an 80/20 train/validation split.
    - *Dataset size:*
        - Training set size: 9600
        - Validation set size: 2400
  - **CoNLL2003 NER Dataset:** Loaded with tokenization adjustments to align subword tokens with reduced label mappings.
    - *Dataset size:*
        - Training set size: 14041
        - Validation set size: 3250
  - **Preprocessing:**
    - For classification, sentences are tokenized with padding and truncation to produce uniform input shapes.
    - For NER, special care is taken to align subword tokens with their corresponding labels, ensuring that token-level predictions can be correctly mapped back to the original tokens.
    - This preprocessing step converts raw text into tensor representations, which are then fed into the transformer model.
- **Model Architecture**
  - **Shared Transformer Backbone:** The model uses a shared pre-trained transformer as the base for encoding input sentences. This ensures that both tasks benefit from high-quality semantic embeddings.
  - **Task-Specific Heads:** 
    - For **classification**, a linear layer takes the pooled sentence embedding (using mean pooling by default) and outputs logits corresponding to the predefined classes
    - For **NER**, a separate linear layer processes the token-level embeddings from the transformer to generate logits for each token in the sequence.
    - These heads are designed to operate concurrently within the same forward pass, providing outputs for both tasks
  - **Loss Computation:**  
    Both classification and NER losses are computed separately and summed for backpropagation.
- **Mixed Precision Training** 
  - Optimization with AMP: To maximize GPU efficiency and maintain numerical stability, mixed precision training is implemented using PyTorch’s torch.cuda.amp package.
    - A gradient scaler is used to prevent underflow during backpropagation.
    - This approach speeds up training and reduces memory consumption while maintaining accuracy.
- **Checkpointing:** Early stopping was implemented with patience ( of 3 epochs) to save the best model based on validation loss.
- **Metrics:**  
  - **Classification:** Evaluated using accuracy (via sklearn’s `accuracy_score`).
  - **NER:** Token-level accuracy computed by comparing predictions against true labels (ignoring special tokens).

### Training Loop - Code Snippet
```python
for epoch in range(num_epochs):
    model.train()
    total_cls_loss = 0.0
    total_ner_loss = 0.0
    steps = 0

    agnews_iter = cycle(train_agnews_loader)
    conll_iter = cycle(train_conll_loader)

    scaler = torch.amp.GradScaler('cuda')
    for _ in tqdm(range(num_steps), desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # Classification step
            batch_cls = next(agnews_iter)
            texts = batch_cls['text']
            cls_labels = batch_cls['label'].to(device)
            outputs_cls = model(sentences=texts)
            cls_logits = outputs_cls['classification_logits']
            loss_cls = classification_loss_fn(cls_logits, cls_labels)

            # NER step
            batch_ner = next(conll_iter)
            outputs_ner = model(inputs=batch_ner)
            ner_logits = outputs_ner['ner_logits']
            ner_labels = batch_ner['labels'].to(device)
            loss_ner = ner_loss_fn(ner_logits.view(-1, model.ner_head.out_features), ner_labels.view(-1))

            loss = loss_cls + loss_ner

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update loss trackers and step counter
        total_cls_loss += loss_cls.item()
        total_ner_loss += loss_ner.item()
        steps += 1

    avg_cls_loss = total_cls_loss / steps
    avg_ner_loss = total_ner_loss / steps
    train_cls_loss_history.append(avg_cls_loss)
    train_ner_loss_history.append(avg_ner_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Classification Loss: {avg_cls_loss:.4f}, Train NER Loss: {avg_ner_loss:.4f}")


    # Validation step
    model.eval()
    total_val_cls_loss = 0.0
    total_val_ner_loss = 0.0
    val_steps = 0

    # Validate Classification Task
    for batch in val_agnews_loader:
        texts = batch['text']
        cls_labels = torch.tensor(batch['label']).to(device)
        outputs = model(sentences=texts)
        cls_logits = outputs['classification_logits']
        loss_cls = classification_loss_fn(cls_logits, cls_labels)
        total_val_cls_loss += loss_cls.item()
        val_steps += 1

    # Validate NER Task
    for batch in val_conll_loader:
        outputs = model(inputs=batch)
        ner_logits = outputs['ner_logits']
        ner_labels = batch['labels'].to(device)
        loss_ner = ner_loss_fn(ner_logits.view(-1, model.ner_head.out_features),
                               ner_labels.view(-1))
        total_val_ner_loss += loss_ner.item()
        val_steps += 1

    avg_val_cls_loss = total_val_cls_loss / (len(val_agnews_loader))
    avg_val_ner_loss = total_val_ner_loss / (len(val_conll_loader))
    val_cls_loss_history.append(avg_val_cls_loss)
    val_ner_loss_history.append(avg_val_ner_loss)

    print(f"Epoch {epoch+1} - Val Classification Loss: {avg_val_cls_loss:.4f}, "
          f"Val NER Loss: {avg_val_ner_loss:.4f}")

    # Early stopping check
    avg_val_loss = avg_val_cls_loss + avg_val_ner_loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
        torch.save(best_model_state, "best_model.pth")  # Save best model to disk.
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("\nTraining complete.")
model.load_state_dict(best_model_state)
``` 
*for complete code check [code.ipynb](code.ipynb) `Task 2 - Final Implementation`. For viewing it as docker build follow [Instruction to build and run docker image](readme.md#instruction-to-build-and-run-docker-image)* 

### Results
- **AG News Test Accuracy:** The model achieved a test accuracy of **92.11 %** on the AG News official test dataset split.
- **CoNLL2003 NER Test Token Accuracy:** The token accuracy for NER was approximately **97.90 %**.
- **Visualization:**  
    ![loss_plot](loss_plot.png)

- **Classification Task Result Examples from Test Set**
  ```
  Text: Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.
  True Label: Business | Predicted Label: Business
  --------------------------------------------------------------------------------
  Text: The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com) SPACE.com - TORONTO, Canada -- A second\team of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately funded suborbital space flight, has officially announced the first\launch date for its manned rocket.
  True Label: Sci/Tech | Predicted Label: Sci/Tech
  --------------------------------------------------------------------------------
  Text: Ky. Company Wins Grant to Study Peptides (AP) AP - A company founded by a chemistry researcher at the University of Louisville won a grant to develop a method of producing better peptides, which are short chains of amino acids, the building blocks of proteins.
  True Label: Sci/Tech | Predicted Label: Sci/Tech
  --------------------------------------------------------------------------------
  ```
  *Classification labels `{0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}`*
- **NER Task Result Examples from Test Set**
  ```
  NER Sample:
  <s>:O cricket:O -:O leicestershire:ORG take:O over:O at:O top:O after:O innings:O victory:O .:O </s>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O
  --------------------------------------------------------------------------------
  NER Sample:
  <s>:O london:LOC 1996:O -:O 08:O -:O 30:O </s>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O
  --------------------------------------------------------------------------------
  NER Sample:
  <s>:O west:MISC indian:MISC all:O -:O round:O ##er:O phil:PER simmons:PER took:O four:O for:O 38:O on:O friday:O as:O leicestershire:ORG beat:O somerset:ORG by:O an:O innings:O and:O 39:O runs:O in:O two:O days:O to:O take:O over:O at:O the:O head:O of:O the:O county:MISC championship:MISC .:O </s>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O <pad>:O
  --------------------------------------------------------------------------------
  ```
  *NER labels `{0: "O", 1: "PER", 2: "ORG", 3: "LOC", 4: "MISC"}`*

## Demo Application
A simple Flask-based web application was developed for demonstration purposes. Users can:
- **Classify Sentences:** Get predictions for the sentence classification task.
- **Perform NER:** View token-level entity predictions.
- **Display Embeddings:** See the fixed-length sentence embeddings along with the corresponding padding mask.

Please follow [Instruction to build and run docker image](readme.md#instruction-to-build-and-run-docker-image) to run the demo.

## Conclusion & Future Directions
- **Key Decisions:**  
  The decision to use a shared transformer backbone with task-specific heads allowed for efficient multi-task learning. The chosen freezing strategies and transfer learning approach helped balance computational efficiency with task performance.
- **Insights:**  
  - Pre-trained models like `"all-mpnet-base-v2"` offer powerful representations that benefit diverse NLP tasks.
  - Alternating between tasks in the training loop supports balanced learning across tasks.
- **Future Work:**  
  Potential next steps include exploring dynamic task weighting, experimenting with alternative pooling strategies, and expanding the dataset size for further improvements.

## Environment Setup
The repository includes a `requirements.txt` file for environment replication. Additionally, the project has been containerized using Docker to streamline deployment and testing.

---

Feel free to reach out if you need additional details or further clarifications on any part of the implementation.
