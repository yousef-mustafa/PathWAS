## ------------------------------------------------------------------------------------------- ##
## Modeling Subpackage Initialization                                                         ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                       ##
##                                                                                            ##
## @description: Exposes modeling interfaces and factory for SNP-to-PAS models.              ##
##               Supports ridge and can be extended with elastic net, Bayesian, etc.         ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Modeling subpackage for SNP-to-PAS regression models."""

from typing import Dict, Type

from .base import ModelConfig, PathwayModel
from .ridge import RidgePathwayModel

# Registry of available model implementations
_MODEL_REGISTRY: Dict[str, Type[PathwayModel]] = {
    "ridge": RidgePathwayModel,
}


def create_model(model_name: str, **kwargs) -> PathwayModel:
    """Factory function to create pathway models.

    Parameters
    ----------
    model_name : str
        Name of the model type. Currently supported: "ridge".
    **kwargs
        Hyperparameters passed to ModelConfig (e.g., lambda_=0.1).

    Returns
    -------
    PathwayModel
        An instance of the requested model type.

    Raises
    ------
    ValueError
        If model_name is not recognized.

    Examples
    --------
    >>> model = create_model("ridge", lambda_=0.5)
    >>> model.fit(pas_matrix, genotypes, snp_ids)
    """
    model_name_lower = model_name.lower()
    if model_name_lower not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type '{model_name}'. Available: {available}"
        )

    config = ModelConfig(model_name=model_name_lower, **kwargs)
    model_class = _MODEL_REGISTRY[model_name_lower]
    return model_class(config)


def register_model(name: str, model_class: Type[PathwayModel]) -> None:
    """Register a new model type with the factory.

    Parameters
    ----------
    name : str
        Name for the model type.
    model_class : Type[PathwayModel]
        The model class to register.

    Examples
    --------
    >>> from pathwas.modeling import register_model, PathwayModel
    >>> class MyCustomModel(PathwayModel):
    ...     # implementation
    ...     pass
    >>> register_model("custom", MyCustomModel)
    """
    _MODEL_REGISTRY[name.lower()] = model_class


def available_models() -> list:
    """Return list of available model types.

    Returns
    -------
    list of str
        Names of registered model types.
    """
    return list(_MODEL_REGISTRY.keys())


__all__ = [
    "ModelConfig",
    "PathwayModel",
    "RidgePathwayModel",
    "create_model",
    "register_model",
    "available_models",
]
