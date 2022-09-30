# Medical Question Understanding

## 1. Paper

This repository contains the code for the following paper:

<li><b><i>"Medical Question Understanding and Answering with Knowledge Grounding and Semantic Self-Supervision"</i></b> <img height="16" src="https://khalilmrini.github.io/images/nih.png" width="24" style="display: inline-block;"/> <img height="16" src="https://khalilmrini.github.io/images/adobe.png" width="16" style="display: inline-block;"/> <br>
<b>Khalil Mrini</b>, Harpreet Singh, Franck Dernoncourt, Seunghyun Yoon, Trung Bui, Walter Chang, Emilia Farcas, Ndapa Nakashole<br>
COLING 2022<br>
<i><img height="16" src="https://khalilmrini.github.io/images/flag-kr.jpeg" width="24" style="display: inline-block;"/> Gyeongju, Republic of Korea, and Online</i></li>

## 2. Installation

Use the following commands to install the requirements for this repository:

```
pip install -r requirements.txt
```

## 3. Training Commands

The ```main.py``` file contains all the training and testing modes.

For example, use ```python3 main.py train_meqsum``` to train a model for the MeQSum dataset.

## 4. Chatbot

You may train a summarizer for the GAR baseline by running the ```train_summarizer_for_gar.py``` file.

Run the ```app_chatbot.py``` file to start the Chatbot and evaluate answers. The DPR and GAR take about 40 to 50 seconds to retrieve an answer, but our model should be near immediate to retrieve one.
