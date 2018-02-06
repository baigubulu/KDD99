import tensorflow as tf
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




def main():
    # Dir of total dataset
    total_dataset_dir = "kddcup.data_10_percent_corrected"
    training_set_dir = "train.csv"
    test_set_dir = "test.csv"
    debug_set_dir = "debug.csv"

    #Parse total dataset

    col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

    num_features = [
        "duration","protocol_type","service","flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate","label"
    ]

    print("Pre processor")
    scaler = MinMaxScaler()


    # Change text to numbers
    total_dataset = pandas.read_csv(total_dataset_dir, header=None, names = col_names)
    total_dataset["label"] = pandas.Categorical(total_dataset["label"]).codes
    total_labels = total_dataset["label"]
    total_dataset["protocol_type"] = pandas.Categorical(total_dataset["protocol_type"]).codes
    total_dataset["service"] = pandas.Categorical(total_dataset["service"]).codes
    total_dataset["flag"] = pandas.Categorical(total_dataset["flag"]).codes
    total_dataset_features = total_dataset[num_features].astype(float)

    # Feature scale
    total_dataset_features = scaler.fit_transform(total_dataset_features)
    total_dataset_df = pandas.DataFrame(total_dataset_features)
    total_dataset_df["label"] = total_labels


    # Split dataset

    train, test = train_test_split(total_dataset_df, test_size=0.3 , shuffle=True)
    train.to_csv(training_set_dir, sep=',',header=False, index=False)
    test.to_csv(test_set_dir, sep=',',header=False, index=False)



    # Debug set

    df_01 = total_dataset_df.sample(frac=0.1)
    df_01.to_csv(debug_set_dir, sep=',',header=False, index=False)

    print("Pre processor complete")

if __name__ == '__main__':
    main()