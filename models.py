class ModelFactory:
    def create_regressor_model(model_type):
        if model_type == 'logistic_regression':
            return LogisticRegression()
        elif model_type == 'decision_tree':
            return DecisionTreeClassifier()
        elif model_type == 'svm':
            return SVC()
        else:
            raise ValueError(f'Model type {model_type} not recognized.')

    def create_mlp_model(self):
        pass


class LogisticRegression:
    def train(self, data):
        print("Training Logistic Regression on data...")


class DecisionTreeClassifier:
    def train(self, data):
        print("Training Decision Tree on data...")


class SVC:
    def train(self, data):
        print("Training SVM on data...")


# Usage:
model_type = 'svm'
model_factory = ModelFactory()
model = model_factory.create_regressor_model(model_type)
model.train(data)
