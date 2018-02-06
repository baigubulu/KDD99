import tensorflow as tf
import pandas
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


def main():
    # Dir of total dataset
    total_dataset_dir = "kddcup.data_10_percent_corrected"
    training_set_dir = "train.csv"
    test_set_dir = "test.csv"
    debug_set_dir = "debug.csv"
    neural_network_model_file = "./model.ckpt"

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

    # Training set

    training_set = pandas.read_csv(training_set_dir, names=num_features)
    training_label = training_set["label"]
    training_label = training_label.as_matrix()
    training_set.drop(['label'], axis=1, inplace=True)


    #Test set

    test_set = pandas.read_csv(test_set_dir, names = num_features)
    test_label = test_set["label"]
    test_label = test_label.as_matrix()
    test_set.drop(['label'], axis=1, inplace=True)

    #Debug set

    #debug_set = pandas.read_csv(debug_set_dir,  names = num_features)
    #debug_labels = debug_set["label"]
    #debug_labels = debug_labels.as_matrix()
    #debug_set.drop(['label'], axis=1, inplace=True)


    # Setup input and output
    input_set = training_set.as_matrix()

    # Add ones for bias and x inputs
    N, M = input_set.shape
    input_x = np.ones((N, M + 1))
    input_x[:, :-1] = input_set


    #Sizes
    epoch_num = 50
    dataset_size = training_set["duration"].count()
    input_size = len(num_features)
    output_size = 38
    hidden_size = 300
    batch_size = 200000


    # Setup y outputs
    output_y = np.zeros((dataset_size, output_size), dtype=float)
    for j in range(dataset_size):
        output_y[j, training_label[j]] = 1.0;


    with tf.device("/gpu:0"):
        #Place holder
        tf.reset_default_graph()
        x = tf.placeholder(shape=(None , input_size), dtype=tf.float64, name='X')
        y = tf.placeholder(shape=(None , output_size ), dtype=tf.float64, name='y')

        # Randomize Weights
        W1 = tf.Variable(tf.truncated_normal([input_size, hidden_size ], stddev=0.01, dtype=tf.float64), dtype=tf.float64)
        W2 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size ], stddev=0.01, dtype=tf.float64), dtype=tf.float64)
        W3 = tf.Variable(tf.truncated_normal([hidden_size, output_size ], stddev=0.01, dtype=tf.float64), dtype=tf.float64)


        # Forward propagation
        A1 =  tf.nn.relu (tf.matmul(x, W1))
        A2 =  tf.nn.relu  (tf.matmul(A1, W2))
        yhat = (tf.matmul(A2, W3))

        #Loss function
        #loss = tf.reduce_sum(tf.square(yhat - y))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = yhat, labels =y))


        #Train
        train_op = tf.train.AdamOptimizer(learning_rate=0.007).minimize(loss)
        tf.set_random_seed(1234)



    #sess = tf.Session()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_plot = []
        for i in range(epoch_num):
            print('EPOCH', i)

            #Batch
            for k in range(round(int(dataset_size)/batch_size)):
                start = k*batch_size
                end = start + batch_size
                _, loss_val = sess.run([train_op, loss] , feed_dict={x: input_x[start:end,:], y: output_y[start:end,:]})

            loss_plot.append(loss_val)
            print("Loss: ", loss_val)




        #saver = tf.train.Saver()
        #Save data
        #saver.save(sess, neural_network_model_file)

        #Plot
        plt.plot(loss_plot)
        plt.ylabel('Loss')
        plt.show()

        #Load data
        # sess = tf.Session()
        # saver = tf.train.import_meta_graph(neural_network_model_file+".meta")
        # saver.restore(sess, tf.train.latest_checkpoint('./'))




        #Test data



        # Setup input and output
        test_input_set = test_set.as_matrix()

        # Add ones for bias and x inputs
        N, M = test_input_set.shape
        test_input_x = np.ones((N, M + 1))
        test_input_x[:, :-1] = test_input_set

        test_dataset_size = test_set["duration"].count()

        # Setup y outputs
        test_output_y = np.zeros((test_dataset_size, output_size), dtype=float)
        for j in range(test_dataset_size):
            test_output_y[j, test_label[j]] = 1.0;

        # total_tests = 0
        # correct = 0
        # confusion = np.zeros((19, 19))
        # for k in range(round(int(test_dataset_size) / batch_size)):
        #     start = k * batch_size
        #     end = start + batch_size
        #     test_ = test_output_y[start:end, :]
        #     y_est_np = sess.run(yhat, feed_dict={x: test_input_x[start:end,:] , y: test_ })
        #     batch_correct = [estimate.argmax(axis=0) == target.argmax(axis=0)
        #                for estimate, target in zip(y_est_np, test_ )]
        #     total_tests = total_tests + len(batch_correct)
        #     correct = correct + sum(batch_correct)
        #     #print(correct)
        #     #print(total_tests)
        #     confusion = confusion + confusion_matrix(test_label,  np.argmax(y_est_np, axis=1)   )



        y_est_np = sess.run(yhat, feed_dict={x: test_input_x, y: test_output_y })
        correct = [estimate.argmax(axis=0) == target.argmax(axis=0)
                   for estimate, target in zip(y_est_np, test_output_y )]

        accuracy = 100 * sum(correct) / len(correct)
        confusion = confusion_matrix(test_label,  np.argmax(y_est_np, axis=1)   )
        #f1 = f1_score(test_label, np.argmax(y_est_np, axis=1))
        stats = precision_recall_fscore_support( test_label , np.argmax(y_est_np, axis=1) , average='weighted')

        # Calculate the prediction accuracy
        #accuracy = 100*(correct/total_tests)
        print('Network  accuracy: %.2f%%' % accuracy)
        #print('Number of tests: %d' % total_tests)
        print(stats)

        df = pandas.DataFrame(confusion)
        df.to_csv("confusion.csv" )

        #df = pandas.DataFrame(f1)
        #df.to_csv("f1_score.csv")

        sess.close()






if __name__ == '__main__':
    main()