# seq2seq chatbot
A python based chatbot implemented with TensorFlow. 

You may need an GPU to run the model.

## Dependencies

- Python 3.5+
- TensorFlow 1.1.0
- Flask



## Training

### Specifying the dataset

You should firstly split your dataset into two seperate parts, say `post` and `response`. Put them in the `dataset` folder, with the name set to `post_<dataset_id>.txt` and `response_<dataset_id>.txt` .

### Training the model

After specifying your dataset, you should make some modifications in `train_seq2seq.py` :

```python
FLAGS.dataset=<your_dataset_id>
FLAGS.mode="train"
```

Then you can train the model by running:

```shell
python train_seq2seq.py
```



## Deployment

After training a model, you can deploy the chatbot by setting:

```python
FLAGS.mode="deploy"
```

then deploy the model by running:

```
python train_seq2seq.py
```

Then the chatbot will listen on `http://127.0.0.1:5000` , you can visit `http://127.0.0.1:5000/<input_sentence>` to get the output.

