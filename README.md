# nlu-dataset-creator

## create intents dataset based on custom format
```
python -m create-datasets intents -p data.csv -d , -e csv -o data 
```
## create entities dataset based on custom format
```
python -m create-datasets entities -p data.csv -d , -e csv -o data -s manual
```
## create entities dataset based on doccano format
```
python -m create-datasets entities -p admin.json -d , -e csv -o data -s doccano
```