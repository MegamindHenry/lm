# LM
This is a lm project for hiwi.

## Structures
```bash
├── data (data is not uploaded in GitHub)
│   ├── tasaTrain.txt
│   ├── tasaTest.txt
│   └── *.txt
├── lm_lib
│   ├── math.py
│   ├── process_df.py
│   ├── read.py
│   └── text.py (tasa Class)
├── lstm_model
│   ├── get_file_list.py
│   ├── find_context.py
│   ├── eval.py
│   ├── prepare_sequences.py
│   ├── tokenization.py
│   ├── train.py
│   └── production.py
├── nltk_model(obsolete)
│   ├── prepare_sents.py
│   ├── train.py
│   └── production.py
├── transformer_model(under_construction)
├── lstm_output
│   └── something
├── nltk_output
│   └── something
├── trained (store intermidiate trained model and files)
│   └── something
├── README.md
├── requirements.txt
└── .gitignore
```

## LSTM Instructions
### Train LSTM model with tasa Corpus
1. install all dependencies `pip3 install -r requirements.txt`
2. create data folder and transfer data (by default tasaTrain.txt is needed)
3. run `python3 prepare_sequences.py`
4. run `python3 tokenization.py`
5. run `python3 train.py`
6. create output folder `mkdir lstm_output`
7. run `python3 production.py` (by default tasaTest.txt is needed in data folder)
8. outputs are in lstm_output folder
### Eval model
1. run `python3 get_file_list.py`
1. run `python3 find_context.py`
1. run `python3 eval.py`

## NLTK Tnstructions (obsolete)
1. install all dependencies `pip3 install -r requirements.txt`
2. create data folder and transfer data (by default tasaTrain.txt is needed)
3. run `python3 prepare_sents.py`
4. run `python3 train.py`
5. create output folder `mkdir nltk_output`
6. run `python3 production.py` (by default tasaTest.txt is needed in data folder)
7. outputs are in nltk_output folder