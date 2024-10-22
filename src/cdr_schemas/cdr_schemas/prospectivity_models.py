from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class NeuralNetUserOptions(BaseModel):
    # data/model inputs processing args
    likely_negative_range: Optional[Tuple[float, float]] = Field(
        default=(0.1, 1.0),
        description="The range of values to consider as likely negatives.",
    )
    fraction_train_split: Optional[float] = Field(
        default=0.8, description="The fraction of the data to use for training."
    )
    upsample_multiplier: Optional[float] = Field(
        default=20.0,
        description="The multiplier for upsampling positives in the training data split.",
    )

    # model args
    dropout_tuple: Optional[Tuple[float, float, float]] = Field(
        default=(0.0, 0.25, 0.25),
        description="Dropout influences variance of network outputs. Low dropout results in deterministic prospectivity map. High dropout results in probabilistic prospectivity map.",
    )

    # model training args
    learning_rate: Optional[float] = Field(
        default=1e-3,
        description="Model learning rate. In machine learning referring to the step size at each iteration while moving toward a minimum of a loss function.",
    )
    weight_decay: Optional[float] = Field(
        default=1e-2,
        description="Model weight decay. A regularization technique that prevents the model weights from growing too large by adding a penalty term to the loss function.",
    )
    smoothing: Optional[float] = Field(
        default=0.3,
        description="Controls certainty of data labels. Low smoothing results in large gradients between low vs high prospectivity areas. High smoothing results in incremental gradients between low vs high prospectivity areas.",
    )


class NeighborhoodFunction(str, Enum):
    GAUSSIAN = "gaussian"
    BUBBLE = "bubble"


class SOMType(str, Enum):
    TOROID = "toroid"
    SHEET = "sheet"


class NeighborhoodDecay(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class LearningRateDecay(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class SOMInitialization(str, Enum):
    RANDOM = "random"
    PCA = "pca"


class SOMGrid(str, Enum):
    HEXAGONAL = "hexagonal"
    RECTANGULAR = "rectangular"


class SOMTrainConfig(BaseModel):
    size: int = Field(default=20, description="Dimension of generated SOM space")
    dimensions_x: Optional[int] = Field(
        default=20, description="Dimension of generated SOM space in x"
    )
    dimensions_y: Optional[int] = Field(
        default=20, description="Dimension of generated SOM space in y"
    )
    num_initializations: Optional[int] = Field(
        default=5, description="Number of initializations to run"
    )
    num_epochs: int = Field(default=10, description="Number of epochs to run")
    grid_type: Optional[SOMGrid] = Field(default=SOMGrid.RECTANGULAR)
    som_type: Optional[SOMType] = Field(default=SOMType.TOROID)
    som_initialization: Optional[SOMInitialization] = Field(
        default=SOMInitialization.RANDOM
    )
    initial_neighborhood_size: Optional[float] = Field(default=0.0)
    final_neighborhood_size: Optional[float] = Field(default=1.0)
    neighborhood_function: Optional[NeighborhoodFunction] = Field(
        default=NeighborhoodFunction.GAUSSIAN
    )
    gaussian_neighborhood_coefficient: Optional[float] = Field(default=0.5)
    learning_rate_decay: Optional[LearningRateDecay] = Field(
        default=LearningRateDecay.LINEAR
    )
    neighborhood_decay: Optional[NeighborhoodDecay] = Field(
        default=NeighborhoodDecay.LINEAR
    )
    initial_learning_rate: Optional[float]
    final_learning_rate: Optional[float]
