import sys
import argparse
import pandas as pd
import numpy as np
import json
from pandas import json_normalize
import string


def load_data(file_path: str, delimiter: str, extention: str) -> pd.DataFrame:
    """
    :param file_path: path to a file
    :param delimiter: delimiter for csv files
    :param extention: the extention of the file (csv/json)
    """
    if extention == "csv":
        column_names = ("text", "intent", "entity-label")
        delimiter = delimiter
        data = pd.read_csv(file_path, names=column_names, delimiter=delimiter, keep_default_na=False)
    elif extention == "json":
        data = pd.read_json(path_or_buf="/content/drive/MyDrive/ner/data/admin.json", lines=True)
    else:
        sys.exit(F"The {extention} file extention is not supported")

    return data


def bio_tag(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    :param dataset: dataset to be bio-tagged
    :return: a bio-tagged dataframe
    """
    # actual bio-tagging
    w = dataset["text"].str.replace(r"[^\w\s]+", "", regex=True)
    splitted = dataset.pop("entity").str.split()
    e = dataset.pop("tag")

    m1 = splitted.str[0].eq(w)
    m2 = [b in a for a, b in zip(splitted, w)]

    dataset["tag"] = np.select([m1, m2 & ~m1], ["B-" + e, "I-" + e],  default="O")

    dataset.loc[dataset["tag"] == "B-O", "tag"] = "O"
    for index, row in dataset.iterrows():
        if row.tag == "B-O":
            row["tag"] = "O"

    return dataset


def prepare_intent_dataset(file_path: str,
                           delimiter: str,
                           extention: str,
                           file_name: str,
                           task: str = "intents",
                           save: bool = True) -> pd.DataFrame:
    """
    :param file_path: path to a file
    :param delimiter: delimiter for csv files
    :param extention: the extention of the file (csv/json)
    :param file_name: name for saving the final dataset
    :param task: task, in this case 'intents;
    :param save: flag for saving the dataset if true
    :return: a dataset ready for model training
    """
    data = load_data(file_path=file_path, delimiter=delimiter, extention=extention)
    data = data[["text", "intent"]]
    if save:
        outpath = f"{task}-{file_name}.csv"
        data.to_csv(outpath, header=["text", "label"], index=False)
    return data


def prepare_entities_dataset(file_path: str,
                             delimiter: str,
                             extention: str,
                             source: str,
                             file_name: str,
                             task: str = "entities",
                             save: bool = True) -> pd.DataFrame:
    """
    :param file_path: path to a file
    :param delimiter: delimiter for csv files
    :param extention: the extention of the file (csv/json)
    :param source: the source of the initial dataset (manual/doccano)
    :param file_name: name for saving the final dataset
    :param task: task, in this case 'entities'
    :param save: flag for saving the dataset if true
    :return: a dataset ready for model training
    """
    data = load_data(file_path=file_path, delimiter=delimiter, extention=extention)
    # data = pd.read_csv("animals_cs_entities-full.csv", delimiter=",", names=("text", "entity-label"))
    if source == "manual":
        data = data[["text", "entity-label"]]
        data = data.rename(columns={"index": "Sentence #"})

        data = data.reset_index()
        data = data.rename(columns={"index": "Sentence #"})
        data["Sentence #"] = "Sentence: " + data["Sentence #"].astype(str)
        data["text"] = data["text"].str.replace(r"[^\w\s]+", "", regex=True)
        data["text"] = data["text"].str.strip()
        data["text"] = data["text"].str.split(" ", expand=False)
        data["entity-label"] = data["entity-label"].str.split(",", expand=False)
        data = data.explode("text", ignore_index=True)
        data = data.dropna().reset_index().drop("index", axis=1)
        for index, row in data.iterrows():
        # in case a row (sentence) contains more than one entity-label pair
            for x in row["entity-label"]:
                if x == "": break
                entity, label = x.split(" - ")
                # the below condition has one flaw: eg. the word 'a' is in 'cat', so it
                # gets a label, but it is fixed in bio-tagging
                # TODO: will look again in this algorithm
                if row["text"].strip() in entity.strip():
                    data.loc[index, "tag"] = label.strip()
                    data.loc[index, "entity"] = entity.strip()
                    break
                else:
                    data.loc[index, "tag"] = "O"
                    data.loc[index, "entity"] = row["text"].strip()

        data = data[["Sentence #", "text","entity", "tag"]]
        data = data.dropna()
        data = bio_tag(data)

        if save:
            outpath = f"{task}-{file_name}.csv"
            data.to_csv(outpath, index=False)
        
        return data

    elif source == "doccano":
        data = pd.read_json(path_or_buf=file_path, lines=True)

        for index, row in data.iterrows():
            start = row.label[0][0]
            end = row.label[0][1]
            label = row.label[0][2]
            entity = row.text[start:end]
            data.loc[index, "entity"] = entity
            data.loc[index, "tag"] = label
            data.loc[index, "entity - label"] = [f"{entity} - {label}"]
            
        data = data.rename(columns={"id": "Sentence #"})
        data["Sentence #"] = "Sentence: " + data["Sentence #"].astype(str)
        data = data.drop(["label", "Comments"], axis=1)

        data["text"] = data["text"].str.split(" ", expand=False)
        data = data.explode("text", ignore_index=False)
        data = data.dropna().reset_index().drop("index", axis=1)

        for index, row in data.iterrows():
            text = data.loc[index, "text"].strip().translate(str.maketrans('', '', string.punctuation))
            if text not in data.loc[index, "entity"].strip():
                data.loc[index, "tag"] = "O"
        
        data = bio_tag(data)

        if save:
            outpath = f"{task}-{file_name}.csv"
            data.to_csv(outpath, index=False)

        return data
    else:
        sys.exit(f"Source: '{source}' is not supported")


def intents_args(args):
    """
    :param args: command line arguments
    """
    file_path = args.path
    delimiter = args.delimiter
    extention = args.extention
    file_name = args.out
    prepare_intent_dataset(file_path, delimiter, extention, file_name)




def entities_args(args):
    """
    :param args: command line arguments
    """
    file_path = args.path
    delimiter = args.delimiter
    extention = args.extention
    file_name = args.out
    source = args.source
    prepare_entities_dataset(file_path, delimiter, extention, source, file_name)



def parse_args() -> argparse.Namespace:
    """
    reads all the command line arguments
    :return: Namespace with command line arguments
    """

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-p", "--path", type=str, required=True, help="data path")
    parent_parser.add_argument("-d", "--delimiter", type=str, required=True, help="delimiter: ',' / '\t'")
    parent_parser.add_argument("-e", "--extention", type=str, help="file extention: csv / json")
    parent_parser.add_argument("-o", "--out", type=str, help="final name for the produced file")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    intents_parser = subparsers.add_parser("intents", help="create dataset for classification", aliases=["i"], parents=[parent_parser])
    intents_parser.set_defaults(func=intents_args)

    entities_parser = subparsers.add_parser("entities", help="create dataset for ner", aliases="e", parents=[parent_parser])
    entities_parser.add_argument("-s", "--source", type=str, help="source: manual / doccano")
    entities_parser.set_defaults(func=entities_args)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)

# python -m create-datasets intents -p data.csv -d , -e csv -o data  
# python -m create-datasets entities -p data.csv -d , -e csv -o data -s manual  
# python -m create-datasets entities -p admin.json -d , -e csv -o data -s doccano