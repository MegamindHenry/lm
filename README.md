# LM
This is a lm project for hiwi.

## Structures
```bash
├── data (data is not uploaded in GitHub)
│   ├── tasaTrain.txt
│   ├── tasaTest.txt
│   └── *.txt
├── lm_lib
│   ├── read.py
│   └── text.py (tasa Class)
├── lstm_model
│   ├── prepare_sequences.py
│   ├── tokenization.py
│   ├── train.py
│   └── production.py
├── nltk_model
├── output (store all output prob tables)
│   └── something
├── trained (store intermidiate trained model and sequences)
│   └── something
├── README.md
├── requirements.txt
└── .gitignore
```

## Instructions
1. install all dependencies `pip3 install -r requirements.txt`
2. create data folder and transfer data (by default tasaTrain.txt is needed)
3. run `python3 prepare_sequences.py`
4. run `python3 tokenization.py`
5. run `python3 train.py`
6. create output folder
7. run `python3 production.py` (by default tasaTest.txt is needed in data folder)
8. outputs are in outputs folder