import random

BINARY_valid_1000 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/valid/combined_outputs-2021-02-08_01-30-23_SOURCE_COLUMN.csv'
BINARY_test_15108 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST/test/combined_outputs-2021-02-08_01-21-46_SOURCE_COLUMN.csv'
# 15108 split between 3000 and 12108:
BINARY_test_3000 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST-COMMITTEE/train/combined_outputs_SOURCE_COLUMN.csv'
BINARY_test_12108 = '/home/peter/covid/datasets/2k-0.5k-rest-BINARY-NEWEST-COMMITTEE/test/combined_outputs_SOURCE_COLUMN.csv'


overview_path = '/home/peter/media/data/covid-19/models-reforged/' \
                'overview-2021-01-14_18-00-08.csv'

current_kernel = 'rbf'

valid_4500 = '/home/peter/media/data/covid-19/models-reforged/' \
             'combined_outputs-2021-01-02_21-48-03.csv'
test_30000 = '/home/peter/media/data/covid-19/models-reforged/' \
             'combined_outputs-2021-01-02_21-50-31.csv'

# valid_4500_18 = '/home/peter/media/data/covid-19/models-reforged/' \
#                 'combined_outputs-2021-01-03_15-56-04.csv'
# test_30000_18 = '/home/peter/media/data/covid-19/models-reforged/' \
#                 'combined_outputs-2021-01-03_15-56-34.csv'

valid_4500_train = '/home/peter/Desktop/Inzynierka/datasets_svm_committee/' \
                   '4500_valid_splitted/train/combined_outputs_softmaxed.csv'
valid_4500_test = '/home/peter/Desktop/Inzynierka/datasets_svm_committee/' \
                  '4500_valid_splitted/test/combined_outputs_softmaxed.csv'

# raw_valid_2250_train = '/home/peter/covid/datasets/svm-datasets-raw/' \
#                        'train/train.csv'
# raw_valid_2250_test = '/home/peter/covid/datasets/svm-datasets-raw/' \
#                       'test/test.csv'

# raw_valid_7500_train = '/home/peter/Desktop/Inzynierka/datasets_svm_committee/' \
#                        '7500_valid_splitted/train/combined_outputs_raw.csv'
# raw_valid_7500_test = '/home/peter/Desktop/Inzynierka/datasets_svm_committee/' \
#                       '7500_valid_splitted/test/combined_outputs_raw.csv'

# raw_test_27000 = '/home/peter/Desktop/datasets_svm_committee/' \
#                  '30000_test_splitted/test/combined_outputs_raw.csv'

valid_7500_train = '/home/peter/covid/Inzynierka/datasets_svm_committee/' \
                   '7500_valid_splitted/train/combined_outputs_softmaxed.csv'
valid_7500_test = '/home/peter/covid/Inzynierka/datasets_svm_committee/' \
                  '7500_valid_splitted/test/combined_outputs_softmaxed.csv'

test_27000 = '/home/peter/covid/Inzynierka/datasets_svm_committee/30000_test_splitted/' \
             'test/combined_outputs_softmaxed.csv'

# new approach 4500/9000/20000:
train_4500 = '/home/peter/covid/Inzynierka/datasets_svm_committee/' \
             '14500_valid-20000_test/4500_valid/' \
             '4500_valid_soft.csv'
valid_9000 = '/home/peter/covid/Inzynierka/datasets_svm_committee/' \
             '14500_valid-20000_test/9000_rest-20000_test/rest_9000/' \
             '9000_test_soft.csv'
test_20000 = '/home/peter/covid/Inzynierka/datasets_svm_committee/' \
             '14500_valid-20000_test/9000_rest-20000_test/test_20000/' \
             '20000_test_soft.csv'

test_20000_NO_DUPLICATES = '/home/peter/covid/Inzynierka/datasets_svm_committee/14500_valid-20000_test/9000_rest-20000_test/test_20000/20000_test_soft_NO_DUPLICATES.csv'

number_of_models = 98
random_committee_len = 15
active_indices = random.sample(range(number_of_models), k=random_committee_len)
# current_mask = [int(ind in active_indices) for ind in range(number_of_models)]
# # # # current_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# current_mask = [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
# current_mask = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# current_mask = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
# current_mask = [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]
# current_mask = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
# current_mask = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
# current_mask = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
# current_mask = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0]
# third party genetic lib:
# current_mask = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
#  1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
#  1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
#  1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
#  1, 1]

# current_mask = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

current_mask2 = [
    1,  # GoogleNet/model-6
    0,  # GoogleNet/model-3
    0,  # GoogleNet/model-4
    0,  # GoogleNet/model-2
    1,  # GoogleNet/model-5
    1,  # GoogleNet/model-1
    0,  # VGG16/model-3
    0,  # VGG16/model-4
    0,  # VGG16/model-2
    0,  # VGG16/model-1
    1,  # DenseNet201/model-6
    0,  # DenseNet201/model-3
    1,  # DenseNet201/model-4
    0,  # DenseNet201/model-2
    0,  # DenseNet201/model-5
    0,  # DenseNet201/model-1
    0,  # DenseNet121/model-4
    0,  # DenseNet121/model-1
    1,  # MobileNet_v2/model-10
    0,  # MobileNet_v2/model-6
    1,  # MobileNet_v2/model-3
    0,  # MobileNet_v2/model-4
    1,  # MobileNet_v2/model-8
    1,  # MobileNet_v2/model-2
    1,  # MobileNet_v2/model-5
    1,  # MobileNet_v2/model-1
    0,  # MobileNet_v2/model-7
    0,  # MobileNet_v2/model-9
    1,  # AlexNet/model-3
    1,  # AlexNet/model-4
    1,  # AlexNet/model-2
    1,  # AlexNet/model-5
    1,  # AlexNet/model-1
    0,  # ResNet-152/model-6
    0,  # ResNet-152/model-4
    0,  # ResNet-152/model-2
    0,  # ResNet-152/model-5
    0,  # ResNet-152/model-1
    0,  # ResNet-152/model-7
    0,  # VGG13/model-3
    0,  # VGG13/model-4
    0,  # VGG13/model-2
    0,  # VGG13/model-1
    0,  # VGG16BN/model-3
    0,  # VGG16BN/model-4
    0,  # VGG16BN/model-1
    0,  # ResNet-50/model-3
    0,  # ResNet-50/model-4
    0,  # VGG19/model-3
    0,  # VGG19/model-4
    0,  # VGG19/model-2
    0,  # VGG19/model-1
    2,  # ShuffleNet_v2_x1_0/model-7
    0,  # ResNet-101/model-6
    0,  # ResNet-101/model-4
    0,  # ResNet-101/model-5
    0,  # ResNet-101/model-7
    0,  # VGG13BN/model-3
    0,  # VGG13BN/model-4
    0,  # VGG13BN/model-2
    0,  # VGG13BN/model-1
    0,  # ResNeXt-50_32x4d/model-3
    0,  # ResNeXt-50_32x4d/model-4
    1,  # ResNeXt-50_32x4d/model-2
    0,  # ResNeXt-50_32x4d/model-1
    0,  # VGG19BN/model-2
    0,  # VGG19BN/model-1
    1,  # ResNeXt-101_32x8d/model-3
    0,  # ResNeXt-101_32x8d/model-4
    0,  # ResNeXt-101_32x8d/model-2
    0,  # ResNeXt-101_32x8d/model-1
    0,  # DenseNet161/model-3
    0,  # DenseNet161/model-4
    0,  # DenseNet161/model-1
    0,  # DenseNet169/model-3
    0,  # DenseNet169/model-5
    0,  # DenseNet169/model-7
    0,  # VGG11/model-3
    0,  # VGG11/model-4
    0,  # VGG11/model-2
    0,  # VGG11/model-1
    1,  # MNASNet1_0/model-4
    0,  # VGG11BN/model-3
    0,  # VGG11BN/model-4
    0,  # VGG11BN/model-2
    0,  # Wide_ResNet-101/model-6
    0,  # Wide_ResNet-101/model-3
    0,  # Wide_ResNet-101/model-4
    0,  # Wide_ResNet-101/model-2
    0,  # Wide_ResNet-101/model-5
    0,  # Wide_ResNet-101/model-1
    0,  # Wide_ResNet-101/model-7
    0,  # Wide_ResNet-50/model-6
    0,  # Wide_ResNet-50/model-3
    0,  # Wide_ResNet-50/model-4
    0,  # Wide_ResNet-50/model-8
    0,  # Wide_ResNet-50/model-5
    0,  # Wide_ResNet-50/model-7
]


current_mask = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

