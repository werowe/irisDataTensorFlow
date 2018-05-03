import tensorflow as tf

FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    print(line)
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    features = dict(zip(COLUMNS,fields))
    label = features.pop('label')

    return features, label
 
feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth']


def parse_line(line):
     parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
     label = parsed_line[-1]  # Last element is the label
     del parsed_line[-1]  # Delete last element
     features = parsed_line  
     d = dict(zip(feature_names, features)), label
     return d

def csv_input_fn(csv_path, batch_size):
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    dataset = dataset.map(parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

ds = tf.data.TextLineDataset("iris_training.csv") 
 
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes=3)


train_path="iris_training.csv"

batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda : csv_input_fn(train_path, batch_size))


