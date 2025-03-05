TO DO 

**Nice to have **

- [ ] Precomit der checker om et comit indeholder en binær fil



**Need to have **

**Lektion 1 **

- [ ] Setup version control for your data, or part of your data, and model

      
**Lektion 2 **

- [x] Create config file
- [x] Setup local pre-commit and add some checks
- [x] Add unit tests to the core functionality of your project codebase
- [x] Load the configurations and manage your hyperparameters
- [x] Setup a Github automation procedure to execute the unittest at new commit
- [x] Automate the training of a new model version if all unit tests pass and changes are merged to main branch
- [x] Implement experiment tracking e.g via WandB or MLFlow
- [x] Automatically add the trained model to a model registry e.g. MLFlow
- [ ] Automatically trigger evaluation of the trained model e.g. using Github Actions


**Lektion 3 **


- [x] Implement a training script (train_ddp.py) that scales the training using data parallelism (Bør checkes, tror ikke det virker)
- [x] Implement a memory optimization strategy
- [ ] Scale the training with data parallelism across multiple nodes
- [ ] Implement the ZeRO optimizer using DeepSpeed and experiment with the different stages

**Lektion 4 **

- [ ] Compress your model by post-training quantization e.g. using TensorRT or PyTorch
- [ ] Benchmark your model in terms of inference time and accuracy after compression.
- [ ] Implement an inference script that utilizes batch inference with the compressed model
