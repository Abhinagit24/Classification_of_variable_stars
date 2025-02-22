'''
This is the training script of the neural network. The train and validation dataset are specified in the get_train_dataset and get_validation_dataset files.
The CNN model is defined in the Hadryss_new file
'''
from qusi.experimental.metric import CrossEntropyAlt, MulticlassAccuracyAlt, MulticlassAUROCAlt
from qusi.session import TrainHyperparameterConfiguration, train_session
from torch.optim import AdamW

from Hadryss_new import HadryssNew, HadryssMultiClassScoreEndModuleNew
from getting_dataset1 import get_train_dataset, get_validation_dataset


def main():
    train_light_curve_dataset = get_train_dataset()
    validation_light_curve_dataset = get_validation_dataset()
    model = HadryssNew.new(end_module=HadryssMultiClassScoreEndModuleNew(number_of_classes=5))
    optimizer = AdamW(model.parameters(), lr=1e-3)
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=1000, cycles=1000, train_steps_per_cycle=100, validation_steps_per_cycle=50)
    loss_function = CrossEntropyAlt()
    metric_functions = [CrossEntropyAlt(), MulticlassAccuracyAlt(number_of_classes=5),
                        MulticlassAUROCAlt(number_of_classes=5)]
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
                  model=model, optimizer=optimizer, loss_function=loss_function, metric_functions=metric_functions,
                  hyperparameter_configuration=train_hyperparameter_configuration)

if __name__ == '__main__':
    main()
