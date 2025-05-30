# Introduce
This project is responsible for refactoring the graph encoder part of CogVLM into a rust crate.

`patch_embed.rs`: This module is responsible for dividing the input image into patches of fixed size and mapping them to the embedding space through convolution operations.

`rope.rs`: This module implements RoPE rotational position encoding to introduce position information in the attention mechanism.

`transformer.rs`: This module implements the basic structure of Transformer.

`patch_dropout.rs`: Randomly drop some image patch tokens during the training phase to improve the generalization ability of the model and prevent overfitting.

`glu_projection.rs`: Adds a gating mechanism to image features to improve feature selection capabilities and allow the model to automatically learn which dimensions are more important.

`processor.rs`: Graph encoder pre-processing, including bicubic interpolation.
