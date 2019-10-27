# Neural machine translation
In this repository, I implemented an English-Vietnamese translation model. The concepts behind this implementation are from the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), but with some slight modifications. I used PyTorch as my Deep Learning framework for this repository.

## Overview
### Dataset
To build my model, I used the preprocessed data [IWSLT'15 English-Vietnamese data [Small]](https://nlp.stanford.edu/projects/nmt/) from Stanford NLP Group. Training sentence pairs are store in `train.en` (English) and `train.vi` (Vietnamese), which consist of 133K sentences in each file. The validation set is taken from `tst2012.en` and `tst2012.vi` (1.5K pairs), test set is from `tst2013.en` and `tst2013.vi` (1.2K pairs).

### Seq2seq model
It is a Encoder-Decoder architecture, in which the Encoder takes the input features and passes its output to the first RNN unit of the Decoder. The Decoder will generate sentences word by word. Here is a simple illustration of the Seq2seq model:

<div align="center"><img src="./imgs/seq2seq.png" height="200"></div>

In Seq2seq model, both Encoder and Decoder are Recurrent Neural Networks.

### Attention
At each timestep, the Attention network will compute the weights for all hidden states of the Encoder output. The Decoder will pay more attention to the states with bigger weights.

<div align="center"><img src="./imgs/attn1.png" height="200"></div>

After that, Attention will produce a context vector, which is the weighted sum of the Encoder output

<div align="center"><img src="./imgs/attn2.png" height="100"></div>

## Implementation
### Sentence preprocessing
For any raw text problems, we must firstly preprocessing them. In this project, since there are a lot of sentences with hundreds of words in the dataset, I care about sentences with 20 words and below (both English and Vietnamese sentences). As a familiar process, I remove all punctuation and digits from sentences, then lower all of them. Then add `<START>` and `<END>` tokens to each sentence and extend them to the same size with padding.

### Encoder
I chose to use bi-directional GRU for my Encoder, which takes both forward and backward outputs into account. By this, the annotation for each word is not only from the preceding words, but also from the following words. The **bi-directional RNN** looks like this.

<div align="center"><img src="./imgs/biRNN.png" height="200"></div>

According to the original paper, the last forward hidden state will be fed into the first unit of Decoder. In my project, I modified it a little by combining the last forward hidden state and the last backward hidden state before feeding into the Decoder.

### Decoder
Different than the Encoder, Decoder only uses **uni-directional RNN**, in this project I use GRU for Decoder. At every timestep, the input to Decoder's GRU is the concatenation of embedded input word and the context vector from Attention network.

### Training
I trained my model using CrossEntropyLoss as my loss function, and Adam as the optimizer. And I decrease my learning rate by 0.2 times every epoch using `lr_scheduler` from PyTorch. At training phase, **teacher forcing** is used, which means that the input words fed into Decoder model are taken from the ground-truth train captions, not from our prediction.

### Inference
At test time, **teacher forcing** is not used because we don't know the ground-truth captions yet. So at each timestep, the input to the Decoder are taken from previous prediction.

### BLEU score
This time I use BLEU score as a validation metric, though it doesn't reflect the quality of a sentence completely, it's still a reasonable choice for this type of problem. BLEU score can also be used for *early stopping*, which I added as an option in my `train` function. To compute BLEU score, I choose `nltk` library, which provides function for computing BLEU-4 score.

### Complete model
Putting all together, here is our complete model

<div align="center"><img src="./imgs/completemodel1.png"></div>

Below is the "modified" version

<div align="center"><img src="./imgs/completemodel2.png"></div>

## Results
Since I use Google Colab, I only trained each model for 5 epochs for quick result. Here is the BLEU score on test set after 5 training epochs without early stopping

Model | BLEU
:---: | :---:
Original | 18.524
Modified | 19.283

With this setting, my modified model seems to achieve a better BLEU score than the original. However, there're some times when I set learning rate decay to different values and enable early stopping, this result varied. Sometimes the first model achieves higher BLEU than the latter.

## Examples
With only 5 epochs, the model cannot perform really well. But let's see how it can translate sentences from test set.

**Sentence 1: "and i was very proud"**

Model | Translated sentence
:---: | :---:
Original | và tôi rất tự hào
Modified | và tôi tự hào rất tự hào

**Sentence 2: "but most people don apost agree"

Model | Translated sentence
:---: | :---:
Original | nhưng hầu hết mọi người không đồng ý
Modified | nhưng hầu hết mọi người không đồng ý

**Sentence 3: "i also didn apost know that the second step is to isolate the victim"

Model | Translated sentence
:---: | :---:
Original | tôi cũng không biết rằng thứ hai là để phân loại các nạn nhân
Modified | tôi cũng không biết rằng bước thứ hai là để chuyển nạn nhân

**Sentence 4: "my family was not poor  and myself  i had never experienced hunger"

Model | Translated sentence
:---: | :---:
Original | gia đình tôi không phải là nghèo và tôi không bao giờ hồi phục hồi
Modified | gia đình tôi không nghèo và tôi không bao giờ có thể nhìn qua đói

**Sentence 5: "this was the first time i heard that people in my country were suffering"

Model | Translated sentence
:---: | :---:
Original | lần đầu tiên tôi nghe thấy mọi người ở đất nước của tôi bị đau khổ
Modified | đó là lần đầu tiên tôi nghe thấy mọi người ở đất nước của tôi rất đau khổ
