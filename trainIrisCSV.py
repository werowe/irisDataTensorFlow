import tensorflow as tf

FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    features = dict(zip(COLUMNS,fields))
    label = features.pop('label')

    return features, label
 
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']  

CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']

FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('Species')
    return features, label

def csv_input_fn(csv_path, batch_size):
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

ds = tf.data.TextLineDataset("iris_training.csv").skip(1)

ds = ds.map(_parse_line)
print(ds)

feature_columns = COLUMNS

est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes=3)

train_path="iris_training.csv"

batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda : csv_input_fn(train_path, batch_size))


