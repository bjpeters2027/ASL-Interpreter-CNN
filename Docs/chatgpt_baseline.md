# ChatGPT Baseline: ASL Alphabet Classifier

## Prompt Sent to ChatGPT

You are an expert in computer vision and deep learning.

I am taking an undergraduate machine learning course (CS4342), and for my final project my teammate and I want to build a small ASL alphabet classifier. The idea is:

- Input: an RGB image of a hand showing a single static ASL letter (A–Z).
- Output: one of 26 classes (letters A–Z).

Constraints and preferences:

- We must **collect our own dataset**, so we cannot download an existing ASL dataset.
- There are 2 of us; each of us can spend about 1 hour collecting/annotating data.
- We will likely use **Python + TensorFlow/Keras**.
- The whole project is only about 10–12 hours of work per person, so we need something reasonable.

Please give me a **strong but simple baseline plan** that covers:

1. How to collect and organize the dataset (folder structure, number of images per class, train/validation split).
2. A reasonable baseline neural network architecture for this 26-class classification problem in Keras.
3. How to train the model (loss, optimizer, epochs, batch size, basic callbacks).
4. How to evaluate the model (accuracy, confusion matrix) and what plots/visualizations I should produce.
5. Any straightforward improvements I can try if I have extra time (but keep it simple, nothing too researchy).

Please provide concrete, practical instructions and example code snippets where appropriate.


## ChatGPT Baseline Response

Below is a baseline plan you can follow to build a 26-class ASL alphabet classifier in TensorFlow/Keras.

---

### 1. Data Collection and Organization

Because you must collect your own data, a simple and feasible strategy is:

- Each of you photographs your own hand forming each ASL letter A–Z.
- For each letter, take multiple pictures under slightly different conditions:
  - Vary background (desk, wall, etc.).
  - Vary lighting (brighter/dimmer, different angles).
  - Slightly vary orientation and distance.

Aim for at least **20–30 images per class** if possible:

- If you reach 20 images per letter, that is 20 × 26 = 520 images.
- If you reach 30 images per letter, that is 780 images.

Organize them into the following directory structure:

```text
data/
  train/
    A/
    B/
    ...
    Z/
  val/
    A/
    B/
    ...
    Z/





