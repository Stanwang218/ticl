from ticl.prediction.tabflex import TabFlex
import torch
import lightning as L
import pytest


@pytest.fixture(autouse=True)
def set_threads():
    return torch.set_num_threads(1)


def test_predict_tabflex():
    L.seed_everything(42)

    # Generate synthetic dataset
    X_train = torch.randn(300, 20)
    coef = torch.randn(20) 
    y_train = (X_train @ coef > 0).int()

    X_test = torch.randn(50, 20)
    y_test = (X_test @ coef > 0).int()

    # Initialize and train TabFlex model
    tabflex = TabFlex()
    tabflex.fit(X_train, y_train)

    # Make predictions
    y_pred = tabflex.predict(X_test)

    # Evaluate performance
    acc = (torch.tensor(y_pred) == y_test).float().mean().item()
    assert acc > 0.9