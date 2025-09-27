# LLM_Based_English_to_Hindi_Language_Translation
Fine-Tuning GPT-2 for English-to-Hindi Translation
This repository contains a complete PyTorch project for fine-tuning a pre-trained GPT-2 model on the task of translating English sentences into Hindi. The project uses an instruction-based fine-tuning approach, where the model learns to follow a specific prompt format to generate the translation.

The repository also includes the foundational code for building a GPT model from scratch, which serves as an excellent learning resource for understanding the underlying architecture of transformers.

Demo Output
The following is an example of the model's output after fine-tuning. It learns to follow the instruction and generate a response in Hindi.

🚀 Features
Instruction Fine-Tuning: The model is trained to understand and respond to a specific instruction format, making it more robust than simple sequence-to-sequence learning.

Hugging Face Integration: Leverages the transformers library to load the pre-trained openai-community/gpt2 model and tokenizer.

Custom Data Pipeline: Includes a custom PyTorch Dataset and collate_fn to efficiently process and pad the training data for batching.

Complete Training & Evaluation: A full training loop with an optimizer (AdamW), loss calculation, and periodic evaluation on a test set.

Text Generation: A generation function to perform inference and see the model's translation capabilities live after training.

GPT from Scratch: Includes a self-contained implementation of the GPT architecture, including Multi-Head Attention, Layer Normalization, and Transformer Blocks for educational purposes.

⚙️ How It Works
The project follows a systematic process for fine-tuning:

Data Loading: A CSV file containing English sentences and their Hindi translations is loaded using pandas.

Prompt Formatting: Each data pair is formatted into a standardized instruction prompt. This teaches the model the context of the task. The format is:

### Instruction:
Translate the given English sentence to Hindi.

### Input:
[English Sentence Here]

### Response:
[Hindi Sentence Here]<|endoftext|>
Tokenization: The formatted text is tokenized using the GPT-2 tokenizer. The entire string (instruction, input, and response) is converted into a sequence of token IDs.

Batching: The custom InstructionDataset and custom_collate_fn prepare the data by creating batches of input and target tensors. It handles padding to ensure all sequences in a batch have the same length.

Fine-Tuning: The pre-trained GPT-2 model is trained on these batches. The model's task is to predict the next token in the sequence. By training on the combined prompt and response, it learns to generate the Response part when given the Instruction and Input.

Inference: After training, we provide the model with a prompt containing only the Instruction and Input. The model then generates the Response part token by token.

📋 Technology Stack & Dependencies
PyTorch: The core deep learning framework.

Transformers (Hugging Face): For loading the pre-trained GPT-2 model and tokenizer.

Pandas: For loading and manipulating the dataset.

TikToken: Used in the initial GPT implementation for tokenization.

NumPy: For numerical operations.

Create a requirements.txt file with the following content:

Plaintext

torch
transformers
pandas
tiktoken
numpy
accelerate
🛠️ Setup and Installation
Clone the Repository
Bash

git clone https://github.com/Ezcanor1/LLM_Based_English_to_Hindi_Language_Translation.git
cd your-repo-name
Create a Virtual Environment (Recommended)

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
Install Dependencies

Bash

pip install -r requirements.txt
Prepare the Dataset

Make sure you have a CSV file named Dataset_English_Hindi.csv in the root directory.

The CSV file must contain two columns: English for the source sentences and Hindi for the target translations.

▶️ Usage
To start the fine-tuning process, run the main Python script. The script will handle data loading, model setup, training, and will print out sample generations periodically.

Bash

python Code.py
You can adjust hyperparameters like the number of epochs (num_epochs), learning rate (lr), and batch size (batch_size) directly in the script.

At the end of each epoch, the script will use a sample from the test set to generate a translation, allowing you to monitor the model's progress.

📂 Code Structure
GPT from Scratch (GPTModel, MultiHeadAttention, etc.): Classes defining the transformer architecture from the ground up.

Data Loading and Preprocessing:

InstructionDataset: A custom PyTorch Dataset to wrap the pandas DataFrame.

format_input(): A function to structure the data into the instruction format.

custom_collate_fn(): A custom collate function for padding and creating batches.

Model Fine-Tuning:

Loading GPT2LMHeadModel and GPT2Tokenizer from Hugging Face.

train_model_simple(): The main training loop that handles optimization, loss calculation, and evaluation.

Inference:

generate() / generate_text_simple(): Functions for generating text from a starting prompt.

The final section of the script demonstrates how to use the fine-tuned model for translation on unseen data from the test set.<img width="1422" height="757" alt="Screenshot 2025-09-28 005930" src="https://github.com/user-attachments/assets/1f5d1e6b-ef45-4b20-8b05-38302f4b8e7f" />

Test of Translation with 2 epochs were each epoch is of 20000 Steps(Latop wasn't powerfull to train the model fully).
