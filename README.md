# saturnGPT
16Million parameter chatbot built from scratch: own BPE tokenizer, own GPT-style transformer, 3 training phases

## Creation process:
Created around late October and early November 2026.
Created with pytorch and only by hand, and a lot of trial and error to learn how pytorch works.

## The model: 
This is a basic Transformer model in GPT2 Style, cut to 16million parameters due to compute costs(A100).

**Build:**
- It uses BPE tokenizer(~2.2k tokens),Adam,flash attention, bf16,transformer(8 blocks with 6 heads per block), optional dropout, gradient clipping,cosine learn rate, warmup.

**Data:**
- 3 phase training(basic english,chat bot conversations,small talk)
- around 1.2billion characters as data
- around 360million tokens in the data
- around 220million words in the data

## Chat samples:
Temperature is 0.5, each message answered on its own (no memory of earlier messages):

```
You: Hello
Bot: Hello! How can I help you today?

You: Thanks a lot
Bot: You're welcome! I hope you find a great style for your daily routine and overall well-being.

You: What's 3 + 20?
Bot: To find the number of options, we'll divide 2 (30 - 3) and divide 2 (30 - 4).

You: Suggest me a vacation spot
Bot: Suggest is a vacation spot that helps us understand the spot as a vacation. It's a great idea to hear your thoughts and feel free to ask if they could help. This helps us understand the spot as a vacation.

You: I feel sad
Bot: Yes, the Earth is still at the Northern Hemisphere. It's a good idea to have a friend or family member to help you get started.
```

Greetings and thanks work well, bc it saw these often in the data. Math, requests and feelings show the limits:
it sounds like an assistant, but it doesn't really understand the question yet tho kind of gets what the topic is abt(the math question results in a math-looking answer).

## Pipeline:
tokenizer → pretrain → midtrain → SFT

**Datasizes:**
- tokenizer: BPEtokenizer, ~2.2k tokens
- pretrain: TinyStories, ~229M tokens (so it learns English)
- midtrain: smol-smoltalk,~131M tokens (so it learns assistant conversations)
- SFT: everyday-conversations, ~8.6k exchanges (so it learns small talk; the loss counts only the answer tokens)

## Results:
|          | loss | 
|----------|------|
| pretrain | 1.40 |
| midtrain | 2.40 |
| SFT      | 2.16 |

~20 minutes on one A100

## What saturnGPT can and cant do:
Saturn does sound like an assistant and says basic english words in sentence like structures, it learned when to stop, picking up on topics, to copy things it doesnt know. But the sentences still arent coherent and mostly dont make sense, tho a lower temperature solves that problem a bit with the trade off of way more limited abd repetitive sentences.

## How to run:
- This model creates and uses files from your drive to work
- Run [A] - [D],[I] - [X] to create these files, make sure to run [0] - [5] in cell order when executing training or loading data
- skip [_], only use if disconnected
- A100 suggested for computing, tho T4 works too(just slower)

## Agenda:
- [0] - [5] : necessarities and set up to chat
- [A] - [D] : prepare the data(once)
- [I] - [X] : training

## Data credits:
- TinyStories (Eldan & Li, 2023)
- smol-smoltalk and everyday-conversations by Hugging Face (HuggingFaceTB)

## Next steps:
To solve the problems my next step will be scaling to 100M parameters with more data, bigger tokenizer and more features
